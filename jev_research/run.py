"""Fixed run entrypoint: `python -m jev_research.run`.

Reads the committed variant.json (phase/arm/seed) and dispatches:
  phase=e0  : integrity audit + feature cache build
  phase=arm : annotator (if any) -> pseudo-labels -> consumer W -> evaluation
Prints a final summary block with all metrics; writes JSON artifacts.
"""
import json
import os
import platform
import sys
import time

import numpy as np
import torch

from . import config, data
from .train_annot import DEV, EpisodeData


def log(msg):
    print(msg, flush=True)


def main():
    t_start = time.time()
    v = config.VARIANT
    log("=" * 70)
    log(f"JEV RUN  task={config.TASK} phase={v['phase']} arm={v.get('arm')} seed={v.get('seed', 0)}")
    log(f"shared config hash inputs: encoder={config.SHARED['encoder']['name']}, "
        f"annotator_steps={config.SHARED['annotator']['steps']}, "
        f"consumer_steps={config.SHARED['consumer']['steps']}")
    log(f"torch {torch.__version__} device={DEV} python={sys.version.split()[0]} "
        f"machine={platform.mac_ver()[0] if sys.platform == 'darwin' else platform.platform()}")
    log("=" * 70)

    split = data.load_split(config.TASK)
    pool = data.pilot_subset(split)
    ledger = data.label_ledger(config.TASK, split)
    lc = data.ledger_counts(ledger)
    log("LABEL LEDGER (upstream-of-eval, task=%s):" % config.TASK)
    log(f"  labeled episodes upstream: {lc['labeled_episodes_upstream_of_eval']} "
        f"(annotator_train={lc['breakdown']['annotator_train']}, "
        f"selection={lc['breakdown']['selection']}, calibration={lc['breakdown']['calibration']})")
    log(f"  ledger id-list hashes: {data.ids_hash(ledger['annotator_train_labels'])[:8]}...")
    log(f"  locked test episodes: {len(split['test'])} (access: none)")

    if v["phase"] == "e0":
        receipt_path = os.path.join(config.CACHE_DIR, f"{config.TASK}_e0_receipt.json")
        with open(receipt_path, "w") as f:
            json.dump({"pass": False, "reason": "E0 in progress or interrupted"}, f)
        from .encode_cache import build_cache
        needed = sorted(set(pool["train_pool"] + pool["dev_labeled"] + pool["eval_holdout"]))
        cache_info = build_cache(needed_demos=needed)
        log(f"[cache] total episodes in cache build scope: {len(needed)}")
        from .e0_audit import run_e0
        report = run_e0(log)
        report["fingerprint"] = config.integrity_fingerprint()
        report["cache_build"] = cache_info
        report["wall_seconds"] = time.time() - t_start
        out = os.path.join(config.run_dir(), "e0_report.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=1)
        log("E0 SUMMARY:")
        for k, val in report["checks"].items():
            log(f"  {k}: {json.dumps(val)[:400]}")
        log(f"E0 PASS={report['pass']} fails={report['fails']}")
        log(f"E0 report written to {out}")
        log(f"Wall time: {report['wall_seconds']:.1f}s")
        with open(receipt_path, "w") as f:
            json.dump(report, f, indent=1)
        if not report["pass"]:
            raise SystemExit(1)
        return

    config.require_integrity()
    if v["phase"] in ("e3vlm", "e3ground", "e3cal", "e3pol"):
        # E3 visual-Jev experiment: frontier VLM transition understanding +
        # small embodiment grounding vs local B2. Locked test untouched; see
        # research/jev/E3_PLAN.md (frozen before results).
        from . import e3 as e3mod
        seed = v.get("seed", 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        if v["phase"] == "e3vlm":
            res = e3mod.run_e3vlm(log)
        elif v["phase"] == "e3ground":
            res = e3mod.run_e3ground(log)
        elif v["phase"] == "e3cal":
            res = e3mod.run_e3cal(log)
        else:
            res = e3mod.run_e3pol(log)
        log("=" * 70)
        log(f"FINAL SUMMARY phase={v['phase']} seed={seed} wall={time.time()-t0:.1f}s")
        log(f"  metrics dir: {config.run_dir()}")
        log("=" * 70)
        return
    if v["phase"] in ("e2cal", "e2pol"):
        # E2 policy-utility experiment. Jev v0 = B2 direct inverse dynamics +
        # calibrated risk + selective pseudo-labels + BC policy success.
        # Locked test untouched; see research/jev/E2_PLAN.md for the
        # predeclared advancement criterion.
        from . import e2 as e2mod
        seed = v.get("seed", 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        if v["phase"] == "e2cal":
            res = e2mod.run_e2cal(log)
        else:
            res = e2mod.run_e2pol(log)
            from . import plots as plotsmod
            cal_dir = os.path.join(
                config.RUNS_DIR, f"{config.TASK}_e2cal_none_seed{seed}")
            with open(os.path.join(cal_dir, "e2cal.json")) as f:
                e2cal = json.load(f)
            arr = np.load(os.path.join(cal_dir, "eval_arrays.npz"))
            made = plotsmod.make_all_plots(
                e2cal, res, {"eval_true": arr["eval_true"],
                             "eval_pred_b2": arr["eval_pred_b2"]}, config.run_dir())
            log(f"[plots] wrote {len(made)} PNGs to {config.run_dir()}")
        log("=" * 70)
        log(f"FINAL SUMMARY phase={v['phase']} seed={seed} wall={time.time()-t0:.1f}s")
        log(f"  metrics dir: {config.run_dir()}")
        log("=" * 70)
        return
    # phase=arm
    arm = v["arm"]
    seed = v.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    stats = data.action_stats(config.TASK, pool["labeled_train"])
    if arm not in ("b0", "b2", "b3", "j0", "ref"):
        raise ValueError("Unknown arm")
    gold_all = pool["train_pool"] if arm == "ref" else pool["labeled_train"]
    lc["arm_total_reserved_labels"] = lc["labeled_episodes_upstream_of_eval"] + (len(ledger["ref_extra_gold"]) if arm == "ref" else 0)
    demos_all = pool["train_pool"] + pool["eval_holdout"] + pool["dev_labeled"]
    eps = EpisodeData(demos_all, gold_all,
                      select_ids=pool["dev_labeled"],
                      eval_ids=pool["eval_holdout"])
    fs = eps.feature_stats(pool["train_pool"])
    eps.norm = {"f_mean": fs["mean"], "f_std": fs["std"],
                "a_mean": np.array(stats["mean"]), "a_std": np.array(stats["std"])}

    pseudo = None
    annot_acc = None
    if arm in ("b2", "b3", "j0"):
        from .train_annot import train_annotator, annotate_episodes, export_bundle
        ck = config.SHARED["annotator"]
        log(f"[annotator] training {arm} ...")
        t0 = time.time()
        annot = train_annotator(arm, eps, pool["labeled_train"], pool["train_pool"],
                                ck["steps"], log, seed)
        log(f"[annotator] done in {time.time()-t0:.1f}s")
        targets = [d for d in pool["train_pool"] if d not in gold_all]
        if arm == "ref":
            targets = []
        log(f"[annotate] pseudo-labeling {len(targets)} episodes: {targets}")
        pseudo = annotate_episodes(annot, eps, targets, log, seed)
        lh = data.ids_hash(ledger["annotator_train_labels"])
        mp = export_bundle(annot, pseudo, lh,
                           os.path.join(config.run_dir(), f"bundle_{arm}"))
        log(f"[annotate] bundle manifest: {mp}")
        # eval-only diagnostic: pseudo-label error on eval-holdout (disclosed)
        from .evaluate import annotation_accuracy
        annot_acc = annotation_accuracy(annot, eps, pool["eval_holdout"])
        log(f"[annotate] accuracy on eval-holdout (eval-only labels): {annot_acc}")
    elif arm == "b0":
        log("[annotator] none (gold-only arm)")
    elif arm == "ref":
        log("[annotator] none (all train-pool actions gold)")

    from .train_consumer import train_consumer
    ck = config.SHARED["consumer"]
    log("[consumer] training ...")
    t0 = time.time()
    model, best_dev = train_consumer(eps, gold_all, pseudo, pool["dev_labeled"], log,
                                     seed, ck["steps"])
    log(f"[consumer] done in {time.time()-t0:.1f}s best_dev_rollout10={best_dev:.5f}")

    from .evaluate import (rollout_error, control_rollouts, bootstrap_ci,
                           paired_improvement, intervention_bank, annotation_accuracy)
    horizon = [1, 5, 10, 20]
    ctx = ck["kwargs"]["ctx"]
    eval_ids = pool["eval_holdout"]
    macro, per_ep = rollout_error(model, eps, eval_ids, horizon, ctx)
    c1 = bootstrap_ci([per_ep[d][10] for d in per_ep])
    c5 = bootstrap_ci([per_ep[d][5] for d in per_ep])
    ctrl_zero = control_rollouts(model, eps, eval_ids, ctx, "zero")
    ctrl_shuffle = control_rollouts(model, eps, eval_ids, ctx, "shuffle")
    log(f"[eval] rollout macro: {macro}")
    log(f"[eval] h10 episode-bootstrap CI: {c1}")
    log(f"[eval] controls: zero-action {ctrl_zero:.5f}, shuffled-action {ctrl_shuffle:.5f}")
    bank = intervention_bank(model, eps, eval_ids,
                             n_states=config.SHARED["intervention"]["n_states"],
                             n_candidates=config.SHARED["intervention"]["n_candidates"])
    log(f"[eval] intervention bank: {bank}")

    out = {
        "arm": arm, "seed": seed, "task": config.TASK,
        "rollout_macro": macro, "rollout_per_episode": per_ep,
        "primary_mean_1_10_ci": bootstrap_ci([ep["mean_1_10"] for ep in per_ep.values()]),
        "primary_endpoint": "episode macro normalized MSE averaged steps 1..10 from first complete context",
        "h10_ci": c1, "h5_ci": c5,
        "controls": {"zero_action": ctrl_zero, "shuffled_action": ctrl_shuffle},
        "intervention_bank": bank,
        "annotation_accuracy_eval_only": annot_acc,
        "best_dev_rollout": best_dev,
        "ledger_counts": lc,
        "gold_episodes": len(gold_all), "pseudo_episodes": len(pseudo or {}),
        "consumer_params": sum(p.numel() for p in model.parameters()),
        "wall_seconds": time.time() - t_start,
    }
    path = os.path.join(config.run_dir(), f"metrics_{arm}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    log("=" * 70)
    log(f"FINAL SUMMARY arm={arm} seed={seed}")
    log(f"  rollout_1 {macro.get(1):.5f} | rollout_5 {macro.get(5):.5f} | "
        f"rollout_10 {macro.get(10):.5f} | rollout_20 {macro.get(20):.5f}")
    log(f"  h10 CI {c1}")
    log(f"  controls zero {ctrl_zero:.5f} shuffle {ctrl_shuffle:.5f} "
        f"(metric must be BELOW both for action-conditioning to matter)")
    log(f"  intervention effect_cos {bank['effect_cosine_mean']} "
        f"zero_pred_mse {bank['zero_action_pred_mse']}")
    if annot_acc:
        log(f"  annot pos_mae {annot_acc['pos_mae']:.4f} rot_mae {annot_acc['rot_mae']:.4f} "
            f"grip_err {annot_acc['grip_err']:.4f}")
    log(f"  gold_eps {len(gold_all)} pseudo_eps {len(pseudo or {})} wall {out['wall_seconds']:.1f}s")
    log(f"  metrics: {path}")
    log("=" * 70)


if __name__ == "__main__":
    main()
