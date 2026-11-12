"""E2 policy-utility experiment: does Jev v0 pseudo-labeling reduce the true
action-labeled data needed for a useful closed-loop policy?

Jev v0 (MVP) = B2 direct probabilistic inverse dynamics + calibrated risk
(calibrate.py) + selective pseudo-labeling + BC policy utility (policy.py).
J0 is retained as an experimental latent arm and is NOT optimized here.

Phase e2cal (this module: run_e2cal):
  train B2/B3 on 24 gold -> pseudo-label 16 unlabeled -> predict_dist on
  train/calib/eval-holdout/unlabeled -> median-train-error thresholds ->
  fit risk maps on RESERVED calib episodes -> reliability + coverage curves
  (risk vs random) on eval-holdout (eval-only labels, disclosed) ->
  episode ranking of the 16 unlabeled -> top-K filter list ->
  B2-vs-B3 motion ablation numbers (dev_labeled selection + eval diagnostic).

Phase e2pol (run_e2pol): policy arms x policy seeds, paired closed-loop eval.

Locked test is never touched: every id below comes from train/dev/calib.
"""
import json
import os

import numpy as np
import torch

from . import config, data
from .train_annot import DEV, EpisodeData


def spearman(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    rx = np.argsort(np.argsort(x, kind="stable")).astype(float)
    ry = np.argsort(np.argsort(y, kind="stable")).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float(rx @ ry / den) if den > 0 else 0.0


def _e2_cache_scope(pool):
    """Feature-cache episodes E2 may render: everything except locked test."""
    return sorted(set(pool["train_pool"] + pool["dev_labeled"]
                      + pool["eval_holdout"] + pool["calib"]))


def _setup_episodes(load_cache=False, log=None):
    split = data.load_split(config.TASK)
    pool = data.pilot_subset(split)
    if load_cache:
        # E2 additionally needs cached features for the reserved calib
        # episodes (rendered from recorded states only; no action access).
        # Incremental: episodes already in the versioned cache are skipped.
        from .encode_cache import build_cache
        needed = _e2_cache_scope(pool)
        info = build_cache(needed_demos=needed)
        if log:
            log(f"[e2cal] cache build scope {len(needed)} episodes: {info}")
    demos = pool["train_pool"] + pool["dev_labeled"] + pool["eval_holdout"] + pool["calib"]
    eps = EpisodeData(demos, pool["labeled_train"],
                      select_ids=pool["dev_labeled"],
                      eval_ids=pool["eval_holdout"],
                      calib_ids=pool["calib"])
    stats = data.action_stats(config.TASK, pool["labeled_train"])
    fs = eps.feature_stats(pool["train_pool"])
    eps.norm = {"f_mean": fs["mean"], "f_std": fs["std"],
                "a_mean": np.array(stats["mean"]), "a_std": np.array(stats["std"])}
    return split, pool, eps


def _map_pred(dist):
    """Posterior-mean cont + MAP gripper prediction in raw units."""
    mu = np.asarray(dist["mu"], dtype=float)
    g = np.where(np.asarray(dist["p_grip"], dtype=float) >= 0.5, 1.0, -1.0)
    return np.concatenate([mu, g.reshape(-1, 1)], axis=1)


def _concat_dists(dists, demos):
    mu = np.concatenate([dists[d]["mu"] for d in demos], 0)
    lv = np.concatenate([dists[d]["logvar"] for d in demos], 0)
    pg = np.concatenate([dists[d]["p_grip"] for d in demos], 0)
    return mu, lv, pg


def run_e2cal(log):
    from .train_annot import train_annotator, annotate_episodes, export_bundle, predict_dist
    from . import calibrate as cal
    from .evaluate import annotation_accuracy

    t_cfg = config.SHARED["e2"]
    seed = config.VARIANT.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = config.run_dir()
    split, pool, eps = _setup_episodes(load_cache=True, log=log)
    unlabeled = [d for d in pool["train_pool"] if d not in pool["labeled_train"]]
    log(f"[e2cal] unlabeled targets ({len(unlabeled)}): {unlabeled}")

    annots, pseudos = {}, {}
    for arm in ("b2", "b3"):
        ck = config.SHARED["annotator"]
        log(f"[e2cal] training {arm} ...")
        annots[arm] = train_annotator(arm, eps, pool["labeled_train"],
                                      pool["train_pool"], ck["steps"], log, seed)
        pseudos[arm] = annotate_episodes(annots[arm], eps, unlabeled, log, seed)
        lh = data.ids_hash(data.label_ledger(config.TASK, split)["annotator_train_labels"])
        mp = export_bundle(annots[arm], pseudos[arm], lh,
                           os.path.join(out_dir, f"bundle_e2_{arm}"))
        log(f"[e2cal] {arm} bundle: {mp}")

    # --- thresholds: median B2 error on dev_labeled (SELECTION role) ---
    # The ledger reserves dev_labeled exactly for such selection choices;
    # calib stays a clean fitting set and eval-holdout a clean evaluation.
    # (v1 rule "median in-sample train error" was abandoned: B2 overfits its
    # 24 train demos ~7x, giving 99%+ "bad" rates and a degenerate map.)
    from . import calibrate as calmod
    dev_dist = predict_dist(annots["b2"], eps, pool["dev_labeled"])
    derr = {"pos": [], "rot": []}
    for d in pool["dev_labeled"]:
        a_true = eps.acts[d][: len(dev_dist[d]["mu"])]
        ev = calmod.error_events(a_true, _map_pred(dev_dist[d]), 1e9, 1e9)
        derr["pos"].append(ev["pos_err"])
        derr["rot"].append(ev["rot_err"])
    taus = calmod.median_thresholds(np.concatenate(derr["pos"]), np.concatenate(derr["rot"]))
    log(f"[e2cal] thresholds (median dev-selection error): {taus}")
    for arm in ("b2", "b3"):
        ckpt = os.path.join(out_dir, f"annot_e2_{arm}_seed{seed}.pt")
        m = annots[arm]["model"]
        torch.save(m.state_dict() if not isinstance(m, tuple) else
                   [x.state_dict() for x in m], ckpt)

    # --- fit risk maps on RESERVED calib episodes ---
    calib_dist = predict_dist(annots["b2"], eps, pool["calib"])
    csc, cev_pos, cev_rot, cev_grip = [], [], [], []
    for d in pool["calib"]:
        a_true = eps.acts[d][: len(calib_dist[d]["mu"])]
        ev = calmod.error_events(a_true, _map_pred(calib_dist[d]),
                                 taus["tau_pos"], taus["tau_rot"])
        n = len(ev["pos_err"])
        lv = np.asarray(calib_dist[d]["logvar"], dtype=float)
        pg = np.asarray(calib_dist[d]["p_grip"], dtype=float)
        for t in range(n):
            csc.append({"score_cont": float(lv[t].mean()),
                        "score_grip": float(calmod.bernoulli_entropy(pg[t]))})
        cev_pos.append(ev["bad_pos"]); cev_rot.append(ev["bad_rot"]); cev_grip.append(ev["bad_grip"])
    calib_events = {"bad_pos": np.concatenate(cev_pos), "bad_rot": np.concatenate(cev_rot),
                    "bad_grip": np.concatenate(cev_grip)}
    risk = calmod.fit_risk_maps(csc, calib_events)
    risk.update({"thresholds": taus, "threshold_rule": "median in-sample B2 train error",
                 "n_calib_episodes": len(pool["calib"]),
                 "calib_event_rates": {k: float(v.mean()) for k, v in calib_events.items()}})
    log(f"[e2cal] calib event rates: {risk['calib_event_rates']}")

    def risks_for(dists, demos):
        out = {}
        for d in demos:
            lv = np.asarray(dists[d]["logvar"], dtype=float)
            pg = np.asarray(dists[d]["p_grip"], dtype=float)
            out[d] = calmod.predict_risk(
                {"score_cont": lv.mean(1), "score_grip": calmod.bernoulli_entropy(pg)}, risk)
        return out

    # --- honest evaluation on eval-holdout (eval-only labels, disclosed) ---
    eval_dist = predict_dist(annots["b2"], eps, pool["eval_holdout"])
    eval_risks = risks_for(eval_dist, pool["eval_holdout"])
    etrue = {}
    for d in pool["eval_holdout"]:
        a_true = eps.acts[d][: len(eval_dist[d]["mu"])]
        etrue[d] = calmod.error_events(a_true, _map_pred(eval_dist[d]),
                                       taus["tau_pos"], taus["tau_rot"])
    flat_risk = np.concatenate([eval_risks[d]["risk_any"] for d in pool["eval_holdout"]])
    flat_any = np.concatenate([etrue[d]["bad_any"] for d in pool["eval_holdout"]]).astype(int)
    flat_pos = np.concatenate([etrue[d]["pos_err"] for d in pool["eval_holdout"]])
    flat_rot = np.concatenate([etrue[d]["rot_err"] for d in pool["eval_holdout"]])
    flat_gripbad = np.concatenate([etrue[d]["bad_grip"] for d in pool["eval_holdout"]]).astype(int)
    auc_any = calmod.rank_auc(flat_risk, flat_any)
    log(f"[e2cal] eval-holdout risk AUC(any-bad)={auc_any}, base rate={flat_any.mean():.3f}")

    # --- coverage curves: risk selection vs random at identical coverage ---
    rng = np.random.default_rng(t_cfg["risk"]["random_seed"])
    coverages = t_cfg["risk"]["coverages"]
    R = t_cfg["risk"]["random_baseline_resamples"]
    curves = {"risk": {}, "random": {}}
    for c in coverages:
        m = calmod.select_lowest_risk(flat_risk, c)
        curves["risk"][str(c)] = {
            "pos_mae": float(flat_pos[m].mean()), "rot_mae": float(flat_rot[m].mean()),
            "grip_err": float(flat_gripbad[m].mean()), "any_bad": float(flat_any[m].mean()),
            "n": int(m.sum())}
        rr = {"pos_mae": [], "rot_mae": [], "grip_err": [], "any_bad": []}
        for _ in range(R):
            mr = calmod.random_mask(len(flat_risk), c, rng)
            rr["pos_mae"].append(float(flat_pos[mr].mean()))
            rr["rot_mae"].append(float(flat_rot[mr].mean()))
            rr["grip_err"].append(float(flat_gripbad[mr].mean()))
            rr["any_bad"].append(float(flat_any[mr].mean()))
        curves["random"][str(c)] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                                    for k, v in rr.items()}
    log(f"[e2cal] coverage any_bad risk vs random: " +
        ", ".join(f"{c}: {curves['risk'][str(c)]['any_bad']:.3f} vs "
                  f"{curves['random'][str(c)]['any_bad']['mean']:.3f}" for c in coverages))

    # --- episode ranking: 16 unlabeled by mean risk; validate on eval-holdout ---
    unl_dist = predict_dist(annots["b2"], eps, unlabeled)
    unl_risks = risks_for(unl_dist, unlabeled)
    ep_mean_risk = {d: float(np.asarray(unl_risks[d]["risk_any"]).mean()) for d in unlabeled}
    ranked = sorted(unlabeled, key=lambda d: ep_mean_risk[d])
    keep = ranked[: t_cfg["risk"]["filter_keep_episodes"]]
    log(f"[e2cal] episode risk ranking (low->high): {[(d, round(ep_mean_risk[d], 3)) for d in ranked]}")
    log(f"[e2cal] filter keep ({len(keep)}): {keep}")
    ep_r, ep_e = [], []
    for d in pool["eval_holdout"]:
        ep_r.append(float(np.asarray(eval_risks[d]["risk_any"]).mean()))
        ep_e.append(float(etrue[d]["pos_err"].mean()))
    ep_spearman = spearman(ep_r, ep_e)
    log(f"[e2cal] eval-holdout episode mean-risk vs mean-pos-err spearman={ep_spearman:.3f}")

    # --- motion ablation diagnostic: B2 (learned temporal) vs B3 (explicit diffs) ---
    # selection-role numbers on dev_labeled; eval-holdout disclosed as diagnostic.
    abl = {}
    for arm in ("b2", "b3"):
        for part, ids in (("dev_labeled", pool["dev_labeled"]),
                          ("eval_holdout", pool["eval_holdout"])):
            acc = annotation_accuracy(annots[arm], eps, ids)
            abl[f"{arm}_{part}"] = acc
    log(f"[e2cal] ablation B2 vs B3 dev_labeled: "
        f"B2 pos {abl['b2_dev_labeled']['pos_mae']:.4f} / B3 pos {abl['b3_dev_labeled']['pos_mae']:.4f}")

    # --- reliability tables on eval-holdout ---
    flat_rp = np.concatenate([eval_risks[d]["risk_pos"] for d in pool["eval_holdout"]])
    flat_bp = np.concatenate([etrue[d]["bad_pos"] for d in pool["eval_holdout"]]).astype(int)
    flat_rg = np.concatenate([eval_risks[d]["risk_grip"] for d in pool["eval_holdout"]])
    flat_bg = np.concatenate([etrue[d]["bad_grip"] for d in pool["eval_holdout"]]).astype(int)
    reliability = {"any": calmod.reliability_table(flat_risk, flat_any),
                   "pos": calmod.reliability_table(flat_rp, flat_bp),
                   "grip": calmod.reliability_table(flat_rg, flat_bg)}

    e2cal = {
        "phase": "e2cal", "seed": seed, "task": config.TASK,
        "thresholds": taus, "auc_any": auc_any,
        "base_any_bad_rate": float(flat_any.mean()),
        "curves": curves, "episode_spearman": ep_spearman,
        "episode_mean_risk": ep_mean_risk, "episode_ranking": ranked,
        "filter_keep": keep, "ablation": abl, "reliability": reliability,
        "risk_artifact": risk,
        "pseudo_counts": {a: {d: [int(np.asarray(p).shape[0]), int(np.asarray(p).shape[1])]
                              for d, p in pseudos[a].items()} for a in pseudos},
        "unlabeled": unlabeled,
    }
    with open(os.path.join(out_dir, "risk_maps.json"), "w") as f:
        json.dump(risk, f, indent=1)
    with open(os.path.join(out_dir, "e2cal.json"), "w") as f:
        json.dump(e2cal, f, indent=1, default=float)
    eval_true = np.concatenate([eps.acts[d][: len(eval_dist[d]["mu"])]
                                for d in pool["eval_holdout"]]).astype(np.float32)
    eval_pred = np.concatenate([_map_pred(eval_dist[d])
                                for d in pool["eval_holdout"]]).astype(np.float32)
    np.savez(os.path.join(out_dir, "eval_arrays.npz"), eval_true=eval_true,
             eval_pred_b2=eval_pred, eval_risk_any=flat_risk.astype(np.float32))
    log(f"[e2cal] wrote e2cal.json + risk_maps.json to {out_dir}")
    return e2cal


def run_e2pol(log):
    from . import policy as pol

    t_cfg = config.SHARED["e2"]
    p_cfg = t_cfg["policy"]
    cal_dir = os.path.join(config.RUNS_DIR,
                           f"{config.TASK}_e2cal_none_seed{config.VARIANT.get('seed', 0)}")
    with open(os.path.join(cal_dir, "e2cal.json")) as f:
        e2cal = json.load(f)
    out_dir = config.run_dir()
    split = data.load_split(config.TASK)
    pool = data.pilot_subset(split)
    train_pool = pool["train_pool"]
    lowdim = data.load_lowdim_episodes(config.TASK, train_pool)
    stats = pol.obs_stats(train_pool, lowdim)
    log(f"[e2pol] obs stats from {len(train_pool)} train-pool episodes (label-free)")

    def load_pseudo(arm):
        arr_dir = os.path.join(cal_dir, f"bundle_e2_{arm}", "arrays")
        out = {}
        for d in e2cal["unlabeled"]:
            p = os.path.join(arr_dir, f"{d}_pseudo_actions.npy")
            out[d] = np.load(p).astype(np.float32)
        return out

    pseudo_b2 = load_pseudo("b2")
    pseudo_b3 = load_pseudo("b3")
    filt = [d for d in e2cal["filter_keep"] if d in pseudo_b2]
    arms = {
        "p0": (pool["labeled_train"], {}),
        "pjev": (pool["labeled_train"], pseudo_b2),
        "pjev_filt": (pool["labeled_train"], {d: pseudo_b2[d] for d in filt}),
        "pb3": (pool["labeled_train"], pseudo_b3),
        "pref": (train_pool, {}),
    }
    log(f"[e2pol] arms: " + ", ".join(
        f"{a}(gold={len(g)},pseudo={len(p)})" for a, (g, p) in arms.items()))

    datasets = {}
    for a, (g, p) in arms.items():
        X, A = pol.build_policy_dataset(g, p, lowdim, train_pool, split)
        datasets[a] = (X, A)
        log(f"[e2pol] {a}: {X.shape[0]} transitions")

    starts = [p_cfg["start_seed_base"] + i for i in range(p_cfg["n_starts"])]
    results = {}
    for a, (X, A) in datasets.items():
        for ps in p_cfg["seeds"]:
            key = f"{a}_pseed{ps}"
            log(f"[e2pol] training {key} ...")
            model, curve = pol.train_policy(
                X, A, stats, p_cfg["steps"], p_cfg["batch"], p_cfg["lr"],
                p_cfg["hidden"], p_cfg["layers"], log, ps, DEV)
            ckpt = os.path.join(out_dir, f"{key}.pt")
            torch.save(model.state_dict(), ckpt)
            model = model.to("cpu")
            log(f"[e2pol] evaluating {key} on {len(starts)} paired starts ...")
            recs = pol.evaluate_policy(model, stats, starts, p_cfg["horizon"], "cpu", log)
            summ = pol.summarize_policy(recs)
            log(f"[e2pol] {key}: success {summ['success_rate']['mean']:.3f} "
                f"[{summ['success_rate']['lo']}, {summ['success_rate']['hi']}] "
                f"modes {summ['failure_modes']}")
            results[key] = {"summary": summ,
                            "records": {str(k): {kk: v for kk, v in r.items()} for k, r in recs.items()},
                            "curve": curve, "ckpt": ckpt}

    # --- paired aggregation: Δ vs p0 per (policy seed, start) cell ---
    from .evaluate import bootstrap_ci
    per_seed_delta = {}
    for a in arms:
        if a == "p0":
            continue
        per_seed_delta[a] = {}
        for ps in p_cfg["seeds"]:
            ra = results[f"{a}_pseed{ps}"]["records"]
            r0 = results[f"p0_pseed{ps}"]["records"]
            d = [float(ra[str(i)]["success"]) - float(r0[str(i)]["success"])
                 for i in range(len(starts))]
            per_seed_delta[a][f"pseed{ps}"] = {"mean": float(np.mean(d)), "ci": bootstrap_ci(d)}
    arm_means = {}
    for a in arms:
        vals = [results[f"{a}_pseed{ps}"]["summary"]["success_rate"]["mean"] for ps in p_cfg["seeds"]]
        arm_means[a] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "per_seed": vals}
    d_ref = arm_means["pref"]["mean"] - arm_means["p0"]["mean"]
    rlv = {}
    for a in ("pjev", "pjev_filt", "pb3"):
        d = arm_means[a]["mean"] - arm_means["p0"]["mean"]
        rlv[a] = float(d / d_ref) if d_ref > 0 else None
    e2pol = {"phase": "e2pol", "arms": list(arms), "policy_seeds": p_cfg["seeds"],
             "n_starts": len(starts), "arm_means": arm_means,
             "per_seed_delta_vs_p0": per_seed_delta,
             "recovered_label_value": rlv,
             "denominator_pref_minus_p0": float(d_ref),
             "results": {k: {"summary": v["summary"], "curve": v["curve"], "ckpt": v["ckpt"]}
                         for k, v in results.items()}}
    with open(os.path.join(out_dir, "e2pol.json"), "w") as f:
        json.dump(e2pol, f, indent=1, default=float)
    full = os.path.join(out_dir, "e2pol_records.json")
    with open(full, "w") as f:
        json.dump({k: v["records"] for k, v in results.items()}, f, default=float)
    log(f"[e2pol] wrote e2pol.json to {out_dir}")
    log(f"[e2pol] arm means: {json.dumps(arm_means, indent=1)}")
    log(f"[e2pol] RLV: {json.dumps(rlv)}")
    return e2pol
