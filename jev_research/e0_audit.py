"""E0 integrity audit: the evaluation must measure the intended transition.

Checks (thresholds predeclared in shared.json):
  1. dataset inventory + hashes + env_args pin
  2. timing: control rate uniformity from embedded state time
  3. action semantics: OSC delta world-frame, gripper bimodality
  4. cross-modal alignment: recorded EE-delta commands vs stored EE position deltas
  5. dynamics replay audit on a subset (robosuite version caveat recorded)
  6. split isolation + label ledger
  7. encoder causality: framewise invariance to neighboring frames
  8. one-batch fit: consumer can overfit a single batch
"""
import json
import os

import h5py
import numpy as np

from . import config, data
from .encode_cache import replay_audit


def run_e0(log):
    S = config.SHARED
    task = config.TASK
    report = {"task": task, "checks": {}, "pass": True}
    th = S["e0_thresholds"]

    split = data.load_split(task)
    pool = data.pilot_subset(split)
    audit_demos = pool["labeled_train"]
    report["audit_label_ids"] = audit_demos
    raw = config.DATASETS[task]["raw"]
    lowdim = config.DATASETS[task]["lowdim"]
    report["checks"]["hashes"] = {
        "raw": data.sha256_file(raw),
        "lowdim": data.sha256_file(lowdim),
    }

    with h5py.File(raw, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
        demos = sorted(f["data"].keys(), key=data.demo_num)
        n = len(demos)
        lens = [int(f["data"][d].attrs["num_samples"]) for d in demos]
        a0 = f["data"][audit_demos[0]]["actions"][:]
        st0 = f["data"][audit_demos[0]]["states"][:]
        g0 = f["data"][audit_demos[0]]["controller_info"]
        ts = g0["user_timestamps"][audit_demos[0].split("_")[-1]][:] if audit_demos[0].split("_")[-1] in g0["user_timestamps"] else None
    report["checks"]["inventory"] = {
        "episodes": n, "total_transitions": int(sum(lens)),
        "len_mean": float(np.mean(lens)), "len_min": int(min(lens)), "len_max": int(max(lens)),
        "env": {"name": env_meta["env_name"], "version": env_meta.get("env_version")},
        "controller": env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"]["type"],
        "control_freq": env_meta["env_kwargs"]["control_freq"],
    }
    # timing from embedded state time (column 0)
    dts = st0[1:, 0] - st0[:-1, 0]
    ok_timing = bool(np.allclose(dts, 1.0 / env_meta["env_kwargs"]["control_freq"], atol=1e-6))
    report["checks"]["timing"] = {
        "state_dt_mean": float(dts.mean()), "expected": 1.0 / env_meta["env_kwargs"]["control_freq"],
        "uniform": ok_timing,
        "user_timestamps_available": ts is not None,
    }

    # action semantics: input space, gripper bimodality
    with h5py.File(raw, "r") as f:
        acts = np.concatenate([f["data"][d]["actions"][:] for d in audit_demos])
    grip = acts[:, 6]
    bimodal = float(np.minimum((grip > 0).mean(), (grip < 0).mean()))
    report["checks"]["action_semantics"] = {
        "dim": acts.shape[1], "range": [float(acts.min()), float(acts.max())],
        "osc_output_max": env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"]["output_max"],
        "control_delta": env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"].get("control_delta"),
        "input_ref_frame": env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"].get("input_ref_frame"),
        "gripper_binary_ok": bool(np.allclose(np.abs(grip), 1)),
        "gripper_frac_positive": float((grip > 0).mean()),
        "gripper_bimodal_score": bimodal,
        "gripper_bimodal_ok": bimodal > th["gripper_bimodal_min"],
    }

    # cross-modal alignment: command EE delta vs low-dim EE position delta
    demo = audit_demos[0]
    ld = data.load_lowdim_episodes(task, [demo])[demo]
    eef = ld["obs"]["robot0_eef_pos"][:]
    act = ld["actions"][:]
    out_max = np.array(env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"]["output_max"][:3])
    cmd_delta = act[:-1, :3] * out_max
    pos_delta = eef[1:] - eef[:-1]
    num = (cmd_delta * pos_delta).sum(-1)
    den = np.linalg.norm(cmd_delta, axis=-1) * np.linalg.norm(pos_delta, axis=-1)
    cos = num / np.maximum(den, 1e-8)
    report["checks"]["alignment"] = {
        "n_steps": int(len(cos)),
        "cos_mean": float(cos.mean()),
        "cos_frac_positive": float((cos > 0).mean()),
        "ok": bool((cos > 0).mean() > th["alignment_cos_frac_min"]),
    }

    # split isolation + ledger
    split = data.load_split(task)
    parts = list(split.values())
    disjoint = all(set(parts[i]).isdisjoint(parts[j]) for i in range(4) for j in range(i + 1, 4))
    assert len(parts[0]) + len(parts[1]) + len(parts[2]) + len(parts[3]) == n
    ledger = data.label_ledger(task, split)
    data.write_split_manifest(task, split)
    report["checks"]["split"] = {
        "sizes": {k: len(v) for k, v in split.items()},
        "disjoint": bool(disjoint),
        "ledger_counts": data.ledger_counts(ledger),
        "ledger_ids_hash": {k: data.ids_hash(ledger[k]) for k in
                            ["annotator_train_labels", "selection_labels", "calibration_labels",
                             "consumer_gold_default", "ref_extra_gold"]},
        "manifest": os.path.join(config.CACHE_DIR, f"{task}_split_manifest.json"),
    }

    # dynamics replay audit (subset)
    audit = replay_audit(audit_demos, max_episodes=th["replay_audit_episodes"])
    qpos_max = max(e["qpos_max"] for e in audit["episodes"])
    one_step_qpos_max = max(e["one_step_qpos_max"] for e in audit["episodes"])
    report["checks"]["replay"] = {
        "episodes": audit["episodes"],
        "image_state_consistency_mae": audit["image_state_consistency_mae"],
        "qpos_max_over_subset": qpos_max,
        "one_step_qpos_max_over_subset": one_step_qpos_max,
        "one_step_qpos_ok": bool(one_step_qpos_max < th["replay_qpos_max"]),
        "image_ok": bool(audit["image_state_consistency_mae"] < th["replay_image_mae_max"]),
        "caveat": "The predeclared H=1 gate uses independently reset transitions. Free-run drift is reported separately and is not silently waived.",
    }

    # encoder causality: framewise invariance (encode same frame with different neighbors)
    from .encode_cache import _encode_batch, _encoder
    model, dev = _encoder()
    rng = np.random.default_rng(0)
    frame = (rng.random((1, 84, 84, 3)) * 255).astype(np.uint8)
    with h5py.File(config.cache_path(), "r") as cf:
        real = cf[demo]["tokens"][10:11]
    t1 = _encode_batch(model, dev, frame).cpu().numpy()
    t2 = _encode_batch(model, dev, np.concatenate([frame, rng.integers(0, 256, (2,84,84,3), dtype=np.uint8)])).cpu().numpy()
    report["checks"]["encoder_causality"] = {
        "framewise_invariance_max_abs_diff": float(np.abs(t1[0] - t2[0]).max()),
        "ok": bool(np.abs(t1[0]-t2[0]).max() < th["encoder_invariance_max_abs"]),
    }

    # one-batch fit with the consumer on cached features (labeled train only)
    import torch
    import torch.nn.functional as F
    from . import models
    from .train_annot import DEV, EpisodeData
    pool = data.pilot_subset(split)
    stats = data.action_stats(task, pool["labeled_train"])
    eps = EpisodeData(pool["labeled_train"], pool["labeled_train"])
    fs = eps.feature_stats(pool["labeled_train"])
    eps.norm = {"f_mean": fs["mean"], "f_std": fs["std"],
                "a_mean": np.array(stats["mean"]), "a_std": np.array(stats["std"])}
    torch.manual_seed(0)
    m = models.ConsumerW(**S["consumer"]["kwargs"]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=S["consumer"]["lr"])
    from .train_consumer import _build_supervised
    X, A, Y = [x[:32].float().to(DEV) for x in _build_supervised(eps, pool["labeled_train"], None)]
    m.eval()
    with torch.no_grad():
        initial_loss = F.mse_loss(m(X, A), Y).item()
    m.train()
    losses = []
    for it in range(300):
        pred = m(X, A)
        loss = F.mse_loss(pred, Y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    m.eval()
    with torch.no_grad():
        final_loss = F.mse_loss(m(X, A), Y).item()
    fit_frac = initial_loss / max(final_loss, 1e-12)
    report["checks"]["one_batch_fit"] = {
        "loss_first": initial_loss, "loss_last": final_loss,
        "reduction_factor": float(fit_frac),
        "ok": bool(fit_frac > th["one_batch_fit_reduction_min"]),
    }

    # aggregate pass/fail
    fails = []
    for name, ok in [("encoder_causality", report["checks"]["encoder_causality"]["ok"]),
                     ("replay_image", report["checks"]["replay"]["image_ok"]),
                     ("gripper_binary", report["checks"]["action_semantics"]["gripper_binary_ok"])]:
        if not ok:
            fails.append(name)
    if not ok_timing:
        fails.append("timing")
    if bimodal <= th["gripper_bimodal_min"]:
        fails.append("gripper_bimodal")
    if not report["checks"]["alignment"]["ok"]:
        fails.append("alignment")
    if not report["checks"]["replay"]["one_step_qpos_ok"]:
        fails.append("one_step_replay_qpos")
    if not report["checks"]["one_batch_fit"]["ok"]:
        fails.append("one_batch_fit")
    if not disjoint:
        fails.append("split_disjoint")
    report["pass"] = not fails
    report["fails"] = fails
    return report
