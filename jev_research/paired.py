"""Phase `paired`: the powered E1 pilot decision run.

All arms x all seeds under one code version, one split, one evaluation set.
Per (arm, seed): train annotator (if any) + consumer; evaluate on the same
12 dev-partition holdout episodes. Statistics: per-episode values are averaged
over seeds (clustering), then bootstrapped over episodes — preserving pairing
and seed clustering per the protocol.

Primary contrast (preregistered pilot form): J0 vs the best non-Jev baseline
selected on DEV-labeled episodes only (never on eval-holdout).
"""
import json
import os
import time

import numpy as np
import torch

from . import config, data
from .train_annot import DEV, EpisodeData
from .evaluate import rollout_error, control_rollouts, bootstrap_ci, intervention_bank


def run_paired(v, split, pool, ledger, lc, log, t_start):
    arms = v["arms"]
    seeds = v["seeds"]
    ck_a = config.SHARED["annotator"]
    ck_c = config.SHARED["consumer"]
    ctx = ck_c["kwargs"]["ctx"]
    horizon = [1, 5, 10, 20]
    eval_ids = pool["eval_holdout"]

    stats = data.action_stats(config.TASK, pool["labeled_train"])
    demos_all = pool["train_pool"] + pool["eval_holdout"] + pool["dev_labeled"]

    results = {}
    banks = {}
    first_models = {}
    for arm in arms:
        if arm not in ("b0", "b2", "b3", "j0", "ref"):
            raise ValueError(f"Unknown arm {arm}")
        gold_all = pool["train_pool"] if arm == "ref" else pool["labeled_train"]
        eps = EpisodeData(demos_all, gold_all,
                          select_ids=pool["dev_labeled"],
                          eval_ids=pool["eval_holdout"])
        fs = eps.feature_stats(pool["train_pool"])
        eps.norm = {"f_mean": fs["mean"], "f_std": fs["std"],
                    "a_mean": np.array(stats["mean"]), "a_std": np.array(stats["std"])}
        slog = lambda m: log(f"  [{arm}] {m}")
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            t0 = time.time()
            pseudo, annot_acc, annot_params = None, None, 0
            if arm in ("b2", "b3", "j0"):
                from .train_annot import (train_annotator, annotate_episodes,
                                          export_bundle)
                annot = train_annotator(arm, eps, pool["labeled_train"],
                                        pool["train_pool"], ck_a["steps"],
                                        slog, seed)
                targets = [d for d in pool["train_pool"] if d not in gold_all]
                pseudo = annotate_episodes(annot, eps, targets, log, seed)
                annot_params = sum(p.numel() for p in annot["model"][0].parameters()) \
                    if arm == "j0" else sum(p.numel() for p in annot["model"].parameters())
                from .evaluate import annotation_accuracy
                annot_acc = annotation_accuracy(annot, eps, eval_ids)
            from .train_consumer import train_consumer
            model, best_dev = train_consumer(eps, gold_all, pseudo,
                                             pool["dev_labeled"],
                                             slog, seed, ck_c["steps"])
            if seed == seeds[0]:
                first_models[arm] = (model, pseudo)
            macro, per_ep = rollout_error(model, eps, eval_ids, horizon, ctx)
            ctrl_zero = control_rollouts(model, eps, eval_ids, ctx, "zero")
            ctrl_shuffle = control_rollouts(model, eps, eval_ids, ctx, "shuffle")
            results[(arm, seed)] = {
                "macro": macro, "per_ep": per_ep, "best_dev": best_dev,
                "zero": ctrl_zero, "shuffle": ctrl_shuffle,
                "annot_acc": annot_acc, "annot_params": annot_params,
                "consumer_params": sum(p.numel() for p in model.parameters()),
                "wall": time.time() - t0,
            }
            log(f"[paired] {arm} seed {seed}: roll10 {macro[10]:.4f} "
                f"mean1_10 {np.mean([ep['mean_1_10'] for ep in per_ep.values()]):.4f} "
                f"zero {ctrl_zero:.4f} shuffle {ctrl_shuffle:.4f} "
                f"({results[(arm, seed)]['wall']:.0f}s)")
        # intervention bank once per arm (diagnostic; reuses first-seed consumer)
        model, pseudo = first_models[arm]
        banks[arm] = intervention_bank(model, eps, eval_ids,
                                       n_states=config.SHARED["intervention"]["n_states"],
                                       n_candidates=config.SHARED["intervention"]["n_candidates"])
        log(f"[paired] {arm} bank: {banks[arm]}")

    # ---- aggregate: episode-level clustering over seeds ----
    def ep_mean(arm, key="mean_1_10"):
        per_ep_by_seed = [results[(arm, s)]["per_ep"] for s in seeds]
        return {d: float(np.mean([pe[d][key] for pe in per_ep_by_seed]))
                for d in eval_ids if all(d in pe for pe in per_ep_by_seed)}

    agg = {}
    for arm in arms:
        vals = list(ep_mean(arm).values())
        agg[arm] = {
            "mean_1_10_episode_macro": float(np.mean(vals)),
            "ci": bootstrap_ci(vals),
            "zero": float(np.mean([results[(arm, s)]["zero"] for s in seeds])),
            "shuffle": float(np.mean([results[(arm, s)]["shuffle"] for s in seeds])),
            "annot_pos_mae": (float(np.mean([results[(arm, s)]["annot_acc"]["pos_mae"]
                                             for s in seeds if results[(arm, s)]["annot_acc"]]))
                              if results[(arm, seeds[0])]["annot_acc"] else None),
            "annot_params": results[(arm, seeds[0])]["annot_params"],
            "consumer_params": results[(arm, seeds[0])]["consumer_params"],
            "wall_mean": float(np.mean([results[(arm, s)]["wall"] for s in seeds])),
        }

    # dev-based selection of best non-Jev baseline (selection data only)
    dev_sel = {arm: float(np.mean([results[(arm, s)]["best_dev"] for s in seeds]))
               for arm in arms if arm not in ("j0",)}
    best_nonjev = min(dev_sel, key=dev_sel.get)

    def paired_diff(a, b):
        va, vb = ep_mean(a), ep_mean(b)
        diffs = [va[d] - vb[d] for d in va if d in vb]
        return bootstrap_ci(diffs)

    contrasts = {f"{a}_minus_{b}": paired_diff(a, b) for a, b in
                 [("b2", "b0"), ("b3", "b0"), ("j0", "b0"), ("ref", "b0"),
                  ("j0", "b2"), ("j0", "b3"), ("j0", "ref"),
                  (best_nonjev, "b0"), ("ref", best_nonjev)]}

    out = {
        "arms": arms, "seeds": seeds, "task": config.TASK,
        "aggregate": agg, "contrasts": contrasts,
        "dev_selection": {"values": dev_sel, "best_nonjev": best_nonjev},
        "intervention_banks": banks,
        "per_run": {f"{a}_s{s}": {k: (v if not isinstance(v, dict) else v)
                                  for k, v in results[(a, s)].items() if k != "per_ep"}
                    for a in arms for s in seeds},
        "per_episode_mean_over_seeds": {a: ep_mean(a) for a in arms},
        "primary_endpoint": "episode-macro normalized MSE, steps 1-10, "
                            "12 dev-partition holdout episodes, seeds clustered",
        "ledger_counts": lc,
        "wall_seconds": time.time() - t_start,
    }
    path = os.path.join(config.run_dir(), "paired_metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)

    log("=" * 70)
    log("PAIRED COMPARISON FINAL TABLE (mean_1_10, lower better, episode-clustered CI)")
    for arm in arms:
        a = agg[arm]
        log(f"  {arm:4s} {a['mean_1_10_episode_macro']:.4f} CI [{a['ci']['lo']:.4f}, "
            f"{a['ci']['hi']:.4f}] zero {a['zero']:.4f} shuffle {a['shuffle']:.4f} "
            f"annot_mae {a['annot_pos_mae']} wall {a['wall_mean']:.0f}s")
    log(f"  dev-selected best non-Jev baseline: {best_nonjev} (dev values {dev_sel})")
    log("PAIRED CONTRASTS (negative = first arm better; CI excluding 0 = decisive)")
    for k, c in contrasts.items():
        lo = c['lo'] if c['lo'] is not None else float('nan')
        hi = c['hi'] if c['hi'] is not None else float('nan')
        log(f"  {k}: {c['mean']:+.4f} CI [{lo:+.4f}, {hi:+.4f}] n={c['n']}")
    for arm in arms:
        b = banks[arm]
        log(f"  bank[{arm}] effect_cos {b['effect_cosine_mean']} "
            f"zero_pred_mse {b['zero_action_pred_mse']}")
    log(f"  metrics: {path}")
    log(f"  total wall: {out['wall_seconds']:.0f}s")
    log("=" * 70)
    return out
