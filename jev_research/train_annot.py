"""Training loops for annotators and consumers, plus pseudo-label export.

Transition definition (stride 1, H=1): (f_{t-1}, f_t, f_{t+1}, a_t) where
a_t is the recorded native control that drove state[t] -> state[t+1].
Offline annotation may use f_{t+1}; the deployed consumer may not.
"""
import json
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from . import config, data, models

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load_tokens(demo):
    with h5py.File(config.cache_path(), "r") as f:
        return f[demo]["tokens"][:].astype(np.float32)  # (T, 6, 384) -> flatten


class EpisodeData:
    """Per-episode token arrays + actions in memory.

    Actions are loaded ONLY for gold_ids (training access), select_ids
    (development-selection access), and eval_ids (evaluation-only access);
    roles are recorded in the label ledger. All other episodes' actions
    are never loaded.
    """

    def __init__(self, demos, gold_ids, select_ids=(), eval_ids=(), calib_ids=()):
        self.eps = {}
        for d in demos:
            toks = load_tokens(d)
            self.eps[d] = {"feat": toks.reshape(len(toks), -1)}
        self.gold = set(gold_ids)
        self.select = set(select_ids)
        self.eval_only = set(eval_ids)
        self.calib = set(calib_ids)
        roles = [self.gold, self.select, self.eval_only, self.calib]
        for i in range(len(roles)):
            for j in range(i + 1, len(roles)):
                if roles[i] & roles[j]:
                    raise ValueError("Action access roles must be disjoint")
        if set(demos) & set(data.load_split(config.TASK)["test"]):
            raise ValueError("Locked test access is forbidden in the pilot")
        if self.calib and not self.calib <= set(data.load_split(config.TASK)["calib"]):
            raise ValueError("Calibration actions must come from the reserved calibration partition")
        self.norm = None
        self.acts = {}
        for d in demos:
            if d in self.gold or d in self.select or d in self.eval_only or d in self.calib:
                _, a = data.load_episode(config.TASK, d)
                self.acts[d] = a.astype(np.float32)

    def feature_stats(self, train_ids):
        if not set(train_ids) <= set(data.load_split(config.TASK)["train"]):
            raise ValueError("Feature normalization requires training episodes only")
        allf = np.concatenate([self.eps[d]["feat"] for d in train_ids], 0)
        return {"mean": allf.mean(0), "std": np.maximum(allf.std(0), 1e-4)}

    def norm_feat(self, f):
        return (f - self.norm["f_mean"]) / self.norm["f_std"]

    def norm_act(self, a):
        return (a - np.array(self.norm["a_mean"], dtype=np.float32)) / np.array(
            self.norm["a_std"], dtype=np.float32)

    def denorm_act(self, a):
        return a * np.array(self.norm["a_std"], dtype=np.float32) + np.array(
            self.norm["a_mean"], dtype=np.float32)

    def transitions(self, demos):
        """Yield (ctx_frames(3,2304) raw-normalized, a_t normalized) per step."""
        X, A = [], []
        for d in demos:
            f = self.norm_feat(self.eps[d]["feat"])
            T = f.shape[0]
            if d in self.acts:
                a = self.norm_act(self.acts[d])
                for t in range(T - 1):
                    f_prev = f[t - 1] if t > 0 else np.zeros_like(f[0])
                    X.append(np.stack([f_prev, f[t], f[t + 1]]))
                    A.append(a[t])
            else:
                for t in range(T - 1):
                    f_prev = f[t - 1] if t > 0 else np.zeros_like(f[0])
                    X.append(np.stack([f_prev, f[t], f[t + 1]]))
                    A.append(np.zeros(7, dtype=np.float32))  # placeholder
        return np.stack(X), np.stack(A)


def context_array(eps, demo):
    """Return the shared normalized transition context for one episode."""
    f = eps.norm_feat(eps.eps[demo]["feat"])
    return np.stack([
        np.stack([f[t - 1] if t > 0 else np.zeros_like(f[0]), f[t], f[t + 1]])
        for t in range(len(f) - 1)
    ])


def _forward_dist(annot, contexts):
    """Run B2, B3, or J0 and return continuous and gripper distributions."""
    if annot["type"] == "j0":
        enc, _, dec = annot["model"]
        mu_z, _ = enc(contexts)
        return dec(mu_z, contexts[:, 1])
    return annot["model"](contexts)


def predict_dist(annot, eps, demos):
    """Predict native-unit action distributions without sampling."""
    models_ = annot["model"] if isinstance(annot["model"], tuple) else (annot["model"],)
    for model in models_:
        model.eval()
    out = {}
    with torch.no_grad():
        for demo in demos:
            contexts = torch.from_numpy(context_array(eps, demo)).to(DEV)
            mu, logvar, grip_logit = _forward_dist(annot, contexts)
            mu_norm = mu.cpu().numpy()
            mu_raw = mu_norm * np.asarray(eps.norm["a_std"][:6]) + np.asarray(
                eps.norm["a_mean"][:6])
            logvar_raw = logvar.cpu().numpy() + 2.0 * np.log(
                np.asarray(eps.norm["a_std"][:6]))
            out[demo] = {
                "mu": mu_raw.astype(np.float32),
                "logvar": logvar_raw.astype(np.float32),
                "p_grip": torch.sigmoid(grip_logit).cpu().numpy().astype(np.float32),
            }
    return out


def train_annotator(arm, eps, labeled_ids, train_ids, steps, log, seed):
    """Train B2/B3/J0. J0 first learns unsupervised on train-pool RGB only,
    then its decoder on the labeled subset only."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    ck = config.SHARED["annotator"]
    Xg, Ag = eps.transitions(labeled_ids)
    Xg_t = torch.from_numpy(Xg).to(DEV)
    Ag_t = torch.from_numpy(Ag).to(DEV)
    a_cont = Ag_t[:, :6]
    a_grip = torch.from_numpy(np.concatenate([eps.acts[d][:-1, 6] for d in labeled_ids])).to(DEV)

    if arm in ("b2", "b3"):
        cls = {"b2": models.B2Inverse, "b3": models.B3FeatureDiff}[arm]
        model = cls(**(ck["kwargs"].get(arm) or {})).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=ck["lr"], weight_decay=1e-4)
        n = Xg_t.shape[0]
        for it in range(steps):
            idx = torch.randint(0, n, (ck["batch"],), device=DEV)
            mu, logvar, lg = model(Xg_t[idx])
            loss = F.gaussian_nll_loss(mu, a_cont[idx], logvar.exp(),
                                       full=True, eps=1e-6) + F.binary_cross_entropy_with_logits(
                lg, (a_grip[idx] > 0).float())
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 500 == 0:
                log(f"  [annot {arm}] step {it} nll {loss.item():.4f}")
        return {"model": model, "type": arm}

    # J0: unsupervised representation on train-pool transitions only
    pool_ids = list(train_ids)
    Xu, _ = eps.transitions(pool_ids)
    Xu_t = torch.from_numpy(Xu).to(DEV)
    enc = models.JevEncoder(**ck["kwargs"]["j0_enc"]).to(DEV)
    pred = models.JevPredictor(**ck["kwargs"]["j0_pred"]).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=ck["lr"], weight_decay=1e-4)
    beta = ck["kl_beta"]
    n = Xu_t.shape[0]
    for it in range(steps):
        idx = torch.randint(0, n, (ck["batch"],), device=DEV)
        frames = Xu_t[idx]
        mu, logvar = enc(frames)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        f_hat = pred(frames[:, 1], z)
        rec = F.mse_loss(f_hat, frames[:, 2])
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(-1).mean()
        loss = rec + beta * kl
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            log(f"  [annot j0] step {it} rec {rec.item():.4f} kl {kl.item():.3f}")
    # grounding decoder on labeled subset only
    dec = models.JevDecoder(**ck["kwargs"]["j0_dec"]).to(DEV)
    opt = torch.optim.AdamW(dec.parameters(), lr=ck["lr"], weight_decay=1e-4)
    enc.eval()
    with torch.no_grad():
        mu_all, _ = enc(Xg_t)
    n = Xg_t.shape[0]
    for it in range(ck["decoder_steps"]):
        idx = torch.randint(0, n, (ck["batch"],), device=DEV)
        mu, logvar, lg = dec(mu_all[idx], Xg_t[idx][:, 1])
        loss = F.gaussian_nll_loss(mu, a_cont[idx], logvar.exp(),
                                   full=True, eps=1e-6) + F.binary_cross_entropy_with_logits(
            lg, (a_grip[idx] > 0).float())
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            log(f"  [annot j0 dec] step {it} nll {loss.item():.4f}")
    return {"model": (enc, pred, dec), "type": "j0"}


def annotate_episodes(annot, eps, demos, log, seed):
    """Export pseudo native actions for @demos (unlabeled train episodes).

    Returns {demo: actions_hat (T-1, 7) in RAW action units}.
    Continuous dims: posterior mean. Gripper: Bernoulli sample (recorded RNG).
    """
    rng = np.random.default_rng(1000 + seed)
    enc, pred, dec = (None, None, None)
    if annot["type"] == "j0":
        enc, pred, dec = annot["model"]
    else:
        model = annot["model"]
    for m in (annot["model"] if isinstance(annot["model"], tuple) else (annot["model"],)):
        m.eval()
    if not set(demos) <= set(data.pilot_subset(data.load_split(config.TASK))["train_pool"]) or set(demos) & eps.gold:
        raise ValueError("Pseudo-label targets must be unlabeled training episodes")
    out = {}
    with torch.no_grad():
        for d in demos:
            f = eps.norm_feat(eps.eps[d]["feat"])
            T = f.shape[0]
            X = np.stack([np.stack([f[t - 1] if t > 0 else np.zeros_like(f[0]),
                                    f[t], f[t + 1]]) for t in range(T - 1)])
            Xt = torch.from_numpy(X).to(DEV)
            if annot["type"] == "j0":
                mu_z, logvar_z = enc(Xt)
                mu, logvar, lg = dec(mu_z, Xt[:, 1])
            elif annot["type"] == "b2":
                mu, logvar, lg = model(Xt)
            else:
                mu, logvar, lg = model(Xt)
            p_grip = torch.sigmoid(lg).cpu().numpy()
            g_raw = np.where(rng.random(len(mu)) < p_grip, 1.0, -1.0).astype(np.float32)
            g_norm = (g_raw - float(eps.norm["a_mean"][6])) / float(eps.norm["a_std"][6])
            a_norm = np.concatenate([mu.cpu().numpy(),
                                     g_norm.reshape(-1, 1)], axis=1)
            out[d] = eps.denorm_act(a_norm)
    return out


def export_bundle(annot, pseudo, ledger_hash, out_dir):
    """Versioned annotation bundle per protocol section 10 (scaled down)."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    arr_dir = os.path.join(out_dir, "arrays")
    os.makedirs(arr_dir, exist_ok=True)
    for d, acts in pseudo.items():
        p = os.path.join(arr_dir, f"{d}_pseudo_actions.npy")
        np.save(p, acts)
        manifest.append({
            "schema_version": "jev-research-v0",
            "source_dataset": "robomimic_v1.5_lift_ph_raw",
            "episode_id": d,
            "split": "train",
            "control_horizon": 1,
            "embodiment": "panda",
            "controller_spec": "OSC_POSE delta world-frame [-1,1]^6 + gripper {-1,1}",
            "native_action_source": "inferred",
            "annotator": annot["type"],
            "pseudo_action_ref": p,
            "lineage": {"ledger_hash": ledger_hash},
        })
    mp = os.path.join(out_dir, "manifest.jsonl")
    with open(mp, "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    return mp
