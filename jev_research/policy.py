"""E2 policy-utility consumer: standard behavior-cloning policy + closed-loop eval.

This is the PRIMARY downstream metric for Jev v0 (B2 + calibration +
filtering). The next-feature world-model consumer (train_consumer.py) is
retained as a secondary diagnostic only.

Policy: small deterministic MLP behavior-cloning policy on the standard
robomimic low-dim Lift observation set, predicting the native 7D action.
Identical architecture / optimizer / update budget / observation
representation / evaluation protocol across arms; the ONLY intended
difference between arms is the training data:

    p0       : 24 gold demos
    pjev     : 24 gold + 16 B2 pseudo-labeled demos (all)
    pjev_filt: 24 gold + 8 lowest-mean-risk B2 pseudo demos (top-50% episodes)
    pb3      : 24 gold + 16 B3 pseudo-labeled demos (strong simple baseline)
    pref     : 40 gold demos (upper reference, NOT budget-matched)

Hygiene enforced here (fail-closed):
  - policy inputs are obs[t] only: no future frames, ever;
  - training ids may only be gold train-pool episodes or pseudo-labeled
    unlabeled train-pool episodes: dev / calib / test ids are rejected;
  - pseudo length must equal obs length - 1 (transition alignment);
  - observation normalization uses train-pool observations only (label-free,
    same precedent as feature_stats);
  - evaluation uses fresh randomized environment resets with a shared,
    predeclared start-seed list so every arm-seed combo is paired.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config, data
from .evaluate import bootstrap_ci

# Standard robomimic low-dim Lift observation set (19 dims total).
POLICY_OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat",
                   "robot0_gripper_qpos", "object"]
OBS_DIM = 3 + 4 + 2 + 10
ACT_DIM = 7


class BCMLP(nn.Module):
    """Deterministic MLP behavior-cloning policy: obs[t] -> 7D action."""

    def __init__(self, hidden=256, layers=2):
        super().__init__()
        seq = []
        d = OBS_DIM
        for _ in range(layers):
            seq += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        self.net = nn.Sequential(*seq)
        self.cont = nn.Linear(hidden, 6)
        self.grip = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.net(obs)
        return self.cont(h), self.grip(h).squeeze(-1)


def obs_vector(lowdim_obs):
    return np.concatenate([np.asarray(lowdim_obs[k], dtype=np.float32)
                           for k in POLICY_OBS_KEYS], axis=-1)


def obs_stats(train_ids, lowdim):
    """Train-pool-only observation normalization (label-free inputs)."""
    if not set(train_ids) <= set(data.load_split(config.TASK)["train"]):
        raise ValueError("Policy obs normalization requires training episodes only")
    allx = np.concatenate(
        [obs_vector({k: v[:] for k, v in lowdim[d]["obs"].items()}) for d in train_ids], 0)
    return {"mean": allx.mean(0), "std": np.maximum(allx.std(0), 1e-4)}


def norm_obs(x, stats):
    return (x - stats["mean"].astype(np.float32)) / stats["std"].astype(np.float32)


def build_policy_dataset(gold_ids, pseudo, lowdim, train_pool, split):
    """Assemble (X, A) with obs[t] -> a[t] for t in 0..L-2.

    The last step is dropped for gold AND pseudo alike so both arms share
    identical transition support (pseudo bundles hold L-1 rows by the H=1
    transition contract). Returns float32 arrays.
    """
    forbidden = set(split["dev"]) | set(split["calib"]) | set(split["test"])
    if set(gold_ids) & forbidden:
        raise ValueError("Policy gold ids must be train-pool episodes")
    if not set(gold_ids) <= set(train_pool):
        raise ValueError("Policy gold ids must be train-pool episodes")
    pseudo = pseudo or {}
    if set(gold_ids) & set(pseudo):
        raise ValueError("Gold and pseudo episodes overlap")
    if set(pseudo) & (forbidden | set(gold_ids)):
        raise ValueError("Pseudo episodes must be unlabeled train-pool episodes")
    if not set(pseudo) <= set(train_pool):
        raise ValueError("Pseudo episodes must be train-pool episodes")
    X, A = [], []
    for d in list(gold_ids) + sorted(pseudo):
        obs = obs_vector(lowdim[d]["obs"])
        L = obs.shape[0]
        if d in pseudo:
            pa = np.asarray(pseudo[d], dtype=np.float32)
            if pa.shape != (L - 1, ACT_DIM) or not np.isfinite(pa).all():
                raise ValueError(f"Invalid pseudo-label shape/values: {d}")
            acts = pa
        else:
            acts = np.asarray(lowdim[d]["actions"], dtype=np.float32)[: L - 1]
            if acts.shape != (L - 1, ACT_DIM) or not np.isfinite(acts).all():
                raise ValueError(f"Invalid gold action shape/values: {d}")
        X.append(obs[: L - 1])
        A.append(acts)
    return np.concatenate(X, 0).astype(np.float32), np.concatenate(A, 0).astype(np.float32)


def train_policy(X, A, stats, steps, batch, lr, hidden, layers, log, seed, device):
    """Fixed-budget BC training. No early stopping: identical updates per arm."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xn = torch.from_numpy(norm_obs(X, stats)).to(device)
    Ac = torch.from_numpy(A[:, :6]).to(device)
    Ag = torch.from_numpy((A[:, 6] > 0).astype(np.float32)).to(device)
    model = BCMLP(hidden=hidden, layers=layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = Xn.shape[0]
    curve = []
    for it in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        mc, lg = model(Xn[idx])
        loss = F.mse_loss(mc, Ac[idx]) + F.binary_cross_entropy_with_logits(lg, Ag[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0 or it == steps - 1:
            curve.append({"step": it, "loss": float(loss.item())})
            log(f"  [policy] step {it} loss {loss.item():.5f}")
    model.eval()
    return model, curve


def _make_rollout_env():
    import json as _json
    import robomimic.utils.env_utils as EnvUtils
    from . import env_setup  # noqa: F401
    import h5py
    with h5py.File(config.DATASETS[config.TASK]["raw"], "r") as src:
        env_meta = _json.loads(src["data"].attrs["env_args"])
    return EnvUtils.create_env_for_data_processing(
        env_meta=env_meta, camera_names=[], camera_height=84, camera_width=84,
        reward_shaping=False)


def rollout_once(env, model, stats, horizon, device, reset_obs):
    """One closed-loop rollout from an already-reset env. Returns a record."""
    import torch

    model.eval()
    heights, sats = [], []
    obs = reset_obs
    with torch.no_grad():
        for t in range(horizon):
            xn = torch.from_numpy(norm_obs(obs_vector(obs)[None], stats)).to(device)
            mc, lg = model(xn)
            a = np.zeros(ACT_DIM, dtype=np.float64)
            a[:6] = np.clip(mc[0].cpu().numpy(), -1, 1)
            a[6] = 1.0 if float(lg[0].cpu().numpy()) > 0 else -1.0
            sats.append(float((np.abs(a[:6]) > 0.99).mean()))
            obs = env.step(a)[0]
            h = float(env.env.sim.data.body_xpos[env.env.cube_body_id][2])
            heights.append(h)
            if env.env._check_success():
                return {"success": True, "steps": t + 1, "heights": heights,
                        "saturation": float(np.mean(sats))}
    return {"success": False, "steps": horizon, "heights": heights,
            "saturation": float(np.mean(sats))}


def failure_mode(record, table_z, contact_eps=0.01, lift_margin=0.04):
    """success / contacted-not-lifted / no-contact from cube-height trace."""
    if record["success"]:
        return "success"
    hs = np.asarray(record["heights"], dtype=float) - table_z
    if (hs.max() - hs[0]) > contact_eps or (hs.max() > lift_margin):
        return "contacted-not-lifted"
    return "no-contact"


def evaluate_policy(model, stats, start_seeds, horizon, device, log):
    """Paired closed-loop evaluation on shared randomized starts."""
    from .encode_cache import close_env
    env = _make_rollout_env()
    table_z = float(env.env.model.mujoco_arena.table_offset[2])
    records = {}
    try:
        for i, s in enumerate(start_seeds):
            np.random.seed(int(s))
            reset_obs = env.reset()
            rec = rollout_once(env, model, stats, horizon, device, reset_obs)
            rec["mode"] = failure_mode(rec, table_z)
            rec["max_lift_above_table"] = float(
                (np.asarray(rec["heights"]) - table_z).max()) if rec["heights"] else None
            heights = rec.pop("heights")
            rec["start_seed"] = int(s)
            records[i] = rec
            if (i + 1) % 10 == 0:
                log(f"  [eval] {i + 1}/{len(start_seeds)} starts")
    finally:
        close_env(env)
    return records


def summarize_policy(records):
    """Episode/start-level aggregation (starts are the independent units)."""
    succ = np.array([1.0 if r["success"] else 0.0 for r in records.values()])
    modes = {}
    for r in records.values():
        modes[r["mode"]] = modes.get(r["mode"], 0) + 1
    steps = np.array([r["steps"] for r in records.values() if r["success"]])
    lifts = np.array([r["max_lift_above_table"] for r in records.values()
                      if r["max_lift_above_table"] is not None])
    sats = np.array([r["saturation"] for r in records.values()])
    return {
        "n_starts": len(records),
        "success_rate": bootstrap_ci(list(succ)),
        "failure_modes": modes,
        "success_steps_mean": float(steps.mean()) if len(steps) else None,
        "max_lift_mean": float(lifts.mean()) if len(lifts) else None,
        "saturation_mean": float(sats.mean()) if len(sats) else None,
    }
