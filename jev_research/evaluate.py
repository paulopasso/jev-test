"""Evaluation: primary rollout error, controls, annotation accuracy,
intervention bank (candidate command execution vs prediction), calibration
diagnostics, bootstrap CIs at the episode level.

Hard rules enforced here:
  - rollout gives the model ONLY past features + supplied true actions;
    predicted frames feed the context autoregressively (no future peeking);
  - locked-test episodes are never loaded in phase `arm`.
"""
import numpy as np
import torch

from .train_annot import DEV


def rollout_error(model, eps, demos, horizon, k_ctx, control=None):
    """Fixed episode-prefix rollout: context ends at t, action[t] predicts t+1.

    All episodes must support the complete requested horizon. The primary
    endpoint is episode-macro MSE averaged across steps 1..10.
    """
    if control not in (None, "zero", "shuffle"):
        raise ValueError("Unknown control")
    horizon = list(horizon)
    if not horizon or min(horizon) < 1 or k_ctx < 1:
        raise ValueError("Positive context and horizons required")
    per_ep = {}
    rng = np.random.default_rng(7)
    model.eval()
    with torch.no_grad():
        for d in demos:
            f = eps.norm_feat(eps.eps[d]["feat"])
            raw = eps.acts[d]
            if len(f) < k_ctx + max(horizon):
                raise ValueError(f"{d}: insufficient frames for complete horizon")
            if control == "zero":
                raw = np.zeros_like(raw)
            elif control == "shuffle":
                raw = raw[rng.permutation(len(raw))]
            a = eps.norm_act(raw)
            ctx = list(f[:k_ctx])
            errors = []
            for step in range(1, max(horizon) + 1):
                t = k_ctx - 1 + step - 1
                cur = torch.from_numpy(np.stack(ctx[-k_ctx:])).unsqueeze(0).float().to(DEV)
                act = torch.from_numpy(a[t]).unsqueeze(0).float().to(DEV)
                pred = model(cur, act)[0].cpu().numpy()
                err = float(np.mean((pred - f[t + 1]) ** 2))
                if not np.isfinite(err):
                    raise ValueError(f"Nonfinite rollout error: {d}, step {step}")
                errors.append(err)
                ctx.append(pred)
            per_ep[d] = {h: errors[h-1] for h in horizon}
            if max(horizon) >= 10:
                per_ep[d]["mean_1_10"] = float(np.mean(errors[:10]))
    if not per_ep:
        raise ValueError("Empty evaluation")
    macro = {key: float(np.mean([ep[key] for ep in per_ep.values()]))
             for key in next(iter(per_ep.values()))}
    return macro, per_ep


def control_rollouts(model, eps, demos, k_ctx, mode):
    return rollout_error(model, eps, demos, [1, 5, 10], k_ctx,
                         control=mode)[0]["mean_1_10"]


def bootstrap_ci(values, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite bootstrap samples")
    if len(values) < 2:
        return {"mean": float(values.mean()) if len(values) else None,
                "lo": None, "hi": None, "n": len(values)}
    idx = rng.integers(0, len(values), (n_boot, len(values)))
    means = values[idx].mean(1)
    return {"mean": float(values.mean()), "lo": float(np.quantile(means, 0.025)),
            "hi": float(np.quantile(means, 0.975)), "n": len(values)}


def paired_improvement(per_ep_a, per_ep_b, horizon):
    """Paired episode-level difference a-b for each h (negative = a better)."""
    out = {}
    for h in horizon:
        diffs = []
        for d in per_ep_a:
            if d in per_ep_b and per_ep_a[d][h] is not None and per_ep_b[d][h] is not None:
                diffs.append(per_ep_a[d][h] - per_ep_b[d][h])
        out[h] = bootstrap_ci(diffs)
    return out


def annotation_accuracy(annot, eps, demos, k_ctx=3):
    """Annotator native-action accuracy on eval-only episodes.
    Reports raw-unit position/rotation/gripper errors separately."""
    model = annot["model"]
    enc = pred = dec = None
    if annot["type"] == "j0":
        enc, pred, dec = model
    for m in (model if isinstance(model, tuple) else (model,)):
        m.eval()
    res = {"pos_mae": [], "rot_mae": [], "grip_err": [], "nll": []}
    with torch.no_grad():
        for d in demos:
            f = eps.norm_feat(eps.eps[d]["feat"])
            a_true = eps.acts[d]  # raw units
            T = f.shape[0]
            X = np.stack([np.stack([f[t - 1] if t > 0 else np.zeros_like(f[0]),
                                    f[t], f[t + 1]]) for t in range(T - 1)])
            Xt = torch.from_numpy(X).to(DEV)
            if annot["type"] == "j0":
                mu_z, _ = enc(Xt)
                mu, logvar, lg = dec(mu_z, Xt[:, 1])
            else:
                mu, logvar, lg = model(Xt)
            p_grip = torch.sigmoid(lg).cpu().numpy()
            g_raw = 2.0 * p_grip - 1.0
            g_norm = (g_raw - float(eps.norm["a_mean"][6])) / float(eps.norm["a_std"][6])
            mu_raw = eps.denorm_act(np.concatenate(
                [mu.cpu().numpy(), g_norm.reshape(-1, 1)], 1))
            pos = np.abs(mu_raw[:, :3] - a_true[: len(mu_raw), :3]).mean(1)
            rot = np.abs(mu_raw[:, 3:6] - a_true[: len(mu_raw), 3:6]).mean(1)
            grip = (mu_raw[:, 6] > 0) != (a_true[: len(mu_raw), 6] > 0)
            res["pos_mae"].append(float(pos.mean()))
            res["rot_mae"].append(float(rot.mean()))
            res["grip_err"].append(float(grip.mean()))
            # gaussian nll in raw units per dim
            var = np.maximum(np.exp(logvar.cpu().numpy()), 1e-6) * (np.array(eps.norm["a_std"][:6]) ** 2)
            se = ((mu_raw[:, :6] - a_true[: len(mu_raw), :6]) ** 2 / var)
            res["nll"].append(float((0.5 * (6 * np.log(2 * np.pi) + np.log(var).sum(1) + se.sum(1))).mean()))
    out = {k: float(np.mean(v)) for k, v in res.items() if v}
    out["nll_definition"] = "continuous-only raw-unit Gaussian joint NLL, episode macro; J0 conditional at posterior mean, NOT marginal predictive NLL"
    return out


def _candidate_bank(a_t, rng, n_candidates):
    """Bounded candidate commands near the demonstration action support."""
    if n_candidates < 6:
        raise ValueError("Candidate bank requires at least six actions including zero")
    cands = [a_t.copy()]
    c = a_t.copy(); c[:3] = -c[:3]; cands.append(c)              # mirrored pos
    c = a_t.copy(); c[:3] *= 0.5; cands.append(c)                 # damped pos
    c = a_t.copy(); c[:3] *= 1.5; cands.append(np.clip(c, -1, 1)) # boosted pos
    c = a_t.copy(); c[6] = -c[6]; cands.append(c)                  # flipped grip
    cands.append(np.zeros_like(a_t))                              # zero action
    while len(cands) < n_candidates:
        c = a_t.copy()
        c[:6] += rng.normal(0, 0.25, 6)
        cands.append(np.clip(c, -1, 1))
    return cands[:n_candidates]


def intervention_bank(model, eps, demos, n_states=8, n_candidates=8, seed=3):
    """One-step effect diagnostic, NOT task success or closed-loop planning.

    Sample states without replacement; bootstrap episode-level aggregates.
    Every candidate restores scene XML, simulator and controller reset state.
    Regret targets the recorded next feature and is only a feature-space proxy.
    """
    from . import config
    from .encode_cache import (_encode_batch, _encoder, reset_episode,
                               reset_controller_goals, close_env)
    import h5py
    rng = np.random.default_rng(seed)
    k = config.SHARED["consumer"]["kwargs"]["ctx"]
    eligible = [(d, t) for d in demos for t in range(k-1, len(eps.eps[d]["feat"])-1)]
    if n_states < 1 or n_states > len(eligible):
        raise ValueError("Invalid intervention state count")
    selected = [eligible[i] for i in rng.choice(len(eligible), n_states, replace=False)]
    enc, dev = _encoder()
    rows, env = [], None
    model.eval()
    try:
        with h5py.File(config.DATASETS[config.TASK]["raw"], "r") as src, torch.no_grad():
            env = _make_env(src)
            for d, t in selected:
                group = src["data"][d]
                cands = _candidate_bank(eps.acts[d][t], rng, n_candidates)
                zi = 5  # explicit canonical zero, even if another candidate is zero
                f = eps.norm_feat(eps.eps[d]["feat"])
                cur = torch.from_numpy(f[t-k+1:t+1]).unsqueeze(0).float().to(DEV)
                actual, predicted = [], []
                for action in cands:
                    reset_episode(env, group, t)
                    reset_controller_goals(env)
                    obs = env.step(action)[0]
                    tok = _encode_batch(enc, dev, obs["agentview_image"][None])[0].cpu().numpy().reshape(-1)
                    actual.append(eps.norm_feat(tok[None])[0])
                    act = torch.from_numpy(eps.norm_act(action[None])).float().to(DEV)
                    predicted.append(model(cur, act)[0].cpu().numpy())
                actual, predicted = np.stack(actual), np.stack(predicted)
                cos, l2 = [], []
                for i in range(len(cands)):
                    if i == zi:
                        continue
                    truth, pred = actual[i]-actual[zi], predicted[i]-predicted[zi]
                    norm = np.linalg.norm(truth)*np.linalg.norm(pred)
                    if norm > 1e-12:
                        cos.append(float(truth @ pred / norm))
                    l2.append(float(np.linalg.norm(truth-pred)))
                actual_cost = np.mean((actual-f[t+1])**2, axis=1)
                chosen = int(np.argmin(np.mean((predicted-f[t+1])**2, axis=1)))
                rows.append({"demo": d, "t": t,
                             "cosine": float(np.mean(cos)) if cos else None,
                             "l2": float(np.mean(l2)),
                             "zero_mse": float(np.mean((predicted[zi]-actual[zi])**2)),
                             "feature_goal_regret": float(actual_cost[chosen]-actual_cost.min())})
    finally:
        close_env(env)
    def episode_values(key):
        out = []
        for d in sorted({r["demo"] for r in rows}):
            vals = [r[key] for r in rows if r["demo"] == d and r[key] is not None]
            if vals:
                out.append(float(np.mean(vals)))
        return out
    ci = bootstrap_ci(episode_values("cosine"))
    return {"n_states_evaluated": len(rows), "n_candidates": n_candidates,
            "effect_cosine_mean": ci["mean"], "effect_cosine_ci": ci,
            "effect_l2_mean": float(np.mean(episode_values("l2"))),
            "zero_action_pred_mse": float(np.mean(episode_values("zero_mse"))),
            "feature_goal_regret_ci": bootstrap_ci(episode_values("feature_goal_regret")),
            "bootstrap_unit": "episode", "states": rows,
            "scope": "one-step reset interventions; feature-space regret, not physical task success"}


def _make_env(src):
    import json as _json
    import robomimic.utils.env_utils as EnvUtils
    from . import env_setup  # noqa: F401

    env_meta = _json.loads(src["data"].attrs["env_args"])
    return EnvUtils.create_env_for_data_processing(
        env_meta=env_meta, camera_names=["agentview"],
        camera_height=84, camera_width=84, reward_shaping=False,
    )
