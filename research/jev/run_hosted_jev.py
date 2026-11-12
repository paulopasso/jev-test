"""Frozen E3b follow-up. Reuses E3 caches; never calls Gemini.

Run with the project's conda Python. --decisions-only skips local evaluation.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from hosted_jev import Client, MODEL, RESOLVED_MODEL, VERSION, parse_response
from jev_research import config, data, calibrate as cal, e3, policy as pol
from jev_research.e2 import _setup_episodes
from jev_research.train_annot import DEV, predict_dist
from jev_research.vlm import load_key

OUT = Path(config.RUNS_DIR) / "lift_e3b_hosted_jev_seed0"
E3POL = Path(config.RUNS_DIR) / "lift_e3pol_none_seed0"
E3CAL = Path(config.RUNS_DIR) / "lift_e3cal_none_seed0"


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def save(name, obj):
    p = OUT / name
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(temp, p)


def decisions(pool):
    unlabeled = [d for d in pool["train_pool"] if d not in pool["labeled_train"]]
    demos = unlabeled + pool["calib"] + pool["eval_holdout"]
    counts = e3.transition_counts(demos)
    source = Path(config.DATA_ROOT) / "e3_vlm_cache/semantic_v2_episode_xml.jsonl"
    source_hash = data.sha256_file(source)
    records = {}
    for line in source.open():
        r = json.loads(line)
        if r["status"] == "ok" and r["demo"] in demos:
            assert r["model"] == "google/gemini-3.6-flash"
            assert r["prompt_version"] == "semantic_v2_episode_xml"
            records[(r["demo"], r["tids"][0])] = r
    expected = [(d, t) for d in demos for t in range(counts[d])]
    if set(records) != set(expected):
        raise RuntimeError("Incomplete or unexpected Gemini cache coverage")
    client = Client(OUT / "decisions.jsonl", load_key(), budget=0.25)
    rows = []
    for i, (d, t) in enumerate(expected):
        src = records[(d, t)]
        raw = src["payload"]["parsed"]["raw"]
        rec = client.decide(raw)
        parsed = parse_response(rec["response"])
        rows.append({"demo": d, "t": t, "source_key": src["key"],
                     "request_hash": rec["request_hash"],
                     "hosted_raw": parsed["raw_risk"],
                     "gemini_raw": 1.0 - float(raw["confidence"])})
        if (i + 1) % 100 == 0:
            log(f"Jev {i+1}/{len(expected)} transitions, {len(client.records)} unique calls, ${client.spent:.5f}")
    if data.sha256_file(source) != source_hash:
        raise RuntimeError("Gemini source changed during run")
    manifest = {"requested_model": MODEL, "resolved_model": RESOLVED_MODEL,
                "prompt_version": VERSION, "source_path": str(source),
                "source_sha256": source_hash, "n_transitions": len(rows),
                "unique_calls_in_cache": len(client.records), "cost_usd": client.spent,
                "gemini_calls": 0, "locked_test_access": False,
                "counts": counts, "unlabeled": unlabeled,
                "calib": pool["calib"], "eval_holdout": pool["eval_holdout"],
                "rows": rows}
    save("decisions_manifest.json", manifest)
    return manifest


def hybrid_predictions(eps, demos):
    from jev_research import models
    counts = e3.transition_counts(demos)
    z = e3.z_matrix("semantic", demos, counts)
    ck = config.SHARED["annotator"]
    b2 = models.B2Inverse(**(ck["kwargs"].get("b2") or {}))
    bpath = Path(config.RUNS_DIR) / "lift_e2cal_none_seed0/annot_e2_b2_seed0.pt"
    b2.load_state_dict(torch.load(bpath, map_location="cpu"))
    b2 = b2.to(DEV).eval()
    dd = predict_dist({"model": b2, "type": "b2"}, eps, demos)
    x = np.concatenate([np.concatenate([
        dd[d]["mu"][:counts[d]], dd[d]["logvar"][:counts[d]].mean(1, keepdims=True),
        dd[d]["p_grip"][:counts[d]].reshape(-1, 1), z[d],
        eps.norm_feat(eps.eps[d]["feat"])[:counts[d]]], 1) for d in demos], 0)
    h = e3.make_grounding(x.shape[1], 0)
    h.load_state_dict(torch.load(Path(config.RUNS_DIR) / "lift_e3ground_none_seed0/ground_e3_hybrid.pt", map_location="cpu"))
    mu, lv, pg = e3.predict_head(h, x.astype(np.float32))
    pred = e3.map_actions(e3.denorm6(mu, eps.norm["a_mean"], eps.norm["a_std"]), pg)
    return pred, lv, pg, counts


def metrics(risk, y):
    y = np.asarray(y, dtype=float)
    risk = np.asarray(risk, dtype=float)
    return {"auc": cal.rank_auc(risk, y), "brier": float(np.mean((risk-y)**2)),
            "coverage": {str(c): float(y[cal.select_lowest_risk(risk, c)].mean())
                         for c in [1.0, 0.8, 0.6, 0.4, 0.2]},
            "reliability": cal.reliability_table(risk, y)}


def fit_map(scores, labels):
    # Jev probabilities have many ties. Pool identical scores correctly;
    # the legacy PAVA helper does not pool all tied-score observations.
    from sklearn.isotonic import IsotonicRegression
    fitted = IsotonicRegression(out_of_bounds="clip").fit(scores, labels)
    return {"xs": fitted.X_thresholds_.tolist(), "ps": fitted.y_thresholds_.tolist()}


def apply_map(mapping, scores):
    return np.interp(scores, mapping["xs"], mapping["ps"])


def calibrate_and_select(manifest, pool):
    _, _, eps = _setup_episodes(load_cache=False, log=log)
    demos = pool["calib"] + pool["eval_holdout"]
    pred, lv, pg, counts = hybrid_predictions(eps, demos)
    labels = np.concatenate([eps.acts[d][:counts[d]].astype(np.float32) for d in demos], 0)
    old = json.loads((E3CAL / "e3cal.json").read_text())
    taus = old["thresholds"]
    y = cal.error_events(labels, pred, taus["tau_pos"], taus["tau_rot"])["bad_any"].astype(float)
    ncal = sum(counts[d] for d in pool["calib"])
    rows = {(r["demo"], r["t"]): r for r in manifest["rows"]}
    out = {"thresholds": taus, "n_calib": ncal, "n_eval": len(y)-ncal,
           "eval_bad_rate": float(y[ncal:].mean()), "arms": {}, "filter_keep": {}}
    diagnostics = {"demo": [d for d in demos for _ in range(counts[d])],
                   "t": [t for d in demos for t in range(counts[d])],
                   "bad_any": y.tolist(), "ncal": ncal}
    for key in ["hosted_raw", "gemini_raw"]:
        score = np.array([rows[(d,t)][key] for d in demos for t in range(counts[d])])
        mapping = fit_map(score[:ncal], y[:ncal])
        risk = apply_map(mapping, score)
        out["arms"][key] = {"raw": metrics(score[ncal:], y[ncal:]),
                             "calibrated": metrics(risk[ncal:], y[ncal:]), "mapping": mapping}
        diagnostics[key] = score.tolist()
        diagnostics[key + "_calibrated"] = risk.tolist()
        ep_risk = {d: float(np.mean(apply_map(mapping,
                   [rows[(d,t)][key] for t in range(manifest["counts"][d])])))
                   for d in manifest["unlabeled"]}
        ranked = sorted(ep_risk, key=lambda d: (ep_risk[d], data.demo_num(d)))
        out["arms"][key]["episode_mean_risk"] = ep_risk
        out["filter_keep"][key] = ranked[:8]
    existing_map = json.loads((E3CAL / "risk_maps_jv-hybrid.json").read_text())
    oldrisk = cal.predict_risk({"score_cont": lv.mean(1), "score_grip": cal.bernoulli_entropy(pg)}, existing_map)["risk_any"]
    out["arms"]["existing_hybrid"] = metrics(oldrisk[ncal:], y[ncal:])
    out["calib_constant_brier"] = float(np.mean((y[:ncal].mean()-y[ncal:])**2))
    diagnostics["existing_hybrid"] = oldrisk.tolist()
    if not np.isclose(out["arms"]["existing_hybrid"]["auc"], old["arms"]["jv-hybrid"]["auc_any"], atol=1e-7):
        raise RuntimeError("Hybrid evaluation replication failed")
    save("calibration.json", out)
    save("calibration_records.json", diagnostics)
    for k in ["hosted_raw", "gemini_raw"]:
        log(f"{k}: AUC {out['arms'][k]['calibrated']['auc']:.4f}; keep {out['filter_keep'][k]}")
    return out


def paired(a, b, seeds, starts):
    from jev_research.evaluate import bootstrap_ci
    matrix = np.array([[float(a[str(ps)][str(i)]["success"])-float(b[str(ps)][str(i)]["success"])
                        for i in range(starts)] for ps in seeds])
    for ps in seeds:
        for i in range(starts):
            assert a[str(ps)][str(i)]["start_seed"] == b[str(ps)][str(i)]["start_seed"]
    rng = np.random.default_rng(731)
    draws = [float(matrix[np.ix_(rng.integers(len(seeds),size=len(seeds)),
                                rng.integers(starts,size=starts))].mean()) for _ in range(10000)]
    return {"mean": float(matrix.mean()), "per_seed": matrix.mean(1).tolist(),
            "pooled_paired_ci": bootstrap_ci(matrix.ravel()),
            "two_way_seed_start_ci": np.quantile(draws,[.025,.975]).tolist()}


def policies(selection, split, pool):
    import h5py
    p_cfg = config.SHARED["e2"]["policy"]
    pseudo = {d: np.load(E3POL / "bundle_e3_hyb/arrays" / f"{d}_pseudo_actions.npy")
              for d in pool["train_pool"] if d not in pool["labeled_train"]}
    # Load labels only for the 24 gold episodes. Native unlabeled actions
    # are not read even though the source file contains them.
    lowdim = {}
    with h5py.File(config.DATASETS[config.TASK]["lowdim"], "r") as f:
        for d in pool["train_pool"]:
            g = f["data"][d]
            lowdim[d] = {"obs": {k: g["obs"][k][:] for k in pol.POLICY_OBS_KEYS}}
            if d in pool["labeled_train"]:
                lowdim[d]["actions"] = g["actions"][:]
    stats = pol.obs_stats(pool["train_pool"], lowdim)
    arms = {"phosted_jev_filt": selection["filter_keep"]["hosted_raw"],
            "pgemini_conf_filt": selection["filter_keep"]["gemini_raw"]}
    old = json.loads((E3POL / "e3pol_records.json").read_text())
    results = {arm: {str(ps): old[f"{arm}_pseed{ps}"] for ps in p_cfg["seeds"]}
               for arm in ["p0", "pb2_filt", "phybrid", "phybrid_filt", "pref"]}
    seen = {tuple(sorted(json.loads((E3CAL/"e3cal.json").read_text())["arms"]["jv-hybrid"]["filter_keep"])): "phybrid_filt"}
    starts = [p_cfg["start_seed_base"]+i for i in range(p_cfg["n_starts"])]
    for arm, kept in arms.items():
        same = seen.get(tuple(sorted(kept)))
        if same:
            results[arm] = results[same]
            log(f"{arm}: same selected data as {same}; reusing policy results")
            continue
        results[arm] = {}
        X,A = pol.build_policy_dataset(pool["labeled_train"],{d:pseudo[d] for d in kept},lowdim,pool["train_pool"],split)
        for ps in p_cfg["seeds"]:
            result_path = OUT / f"{arm}_pseed{ps}.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                if saved["filter_keep"] != kept:
                    raise RuntimeError("Existing policy result has a different selection")
                results[arm][str(ps)] = saved["records"]
                continue
            log(f"Training {arm} seed {ps}")
            model,curve = pol.train_policy(X,A,stats,p_cfg["steps"],p_cfg["batch"],p_cfg["lr"],p_cfg["hidden"],p_cfg["layers"],log,ps,DEV)
            torch.save(model.state_dict(),OUT/f"{arm}_pseed{ps}.pt")
            recs = pol.evaluate_policy(model.to("cpu"),stats,starts,p_cfg["horizon"],"cpu",log)
            records = {str(k):v for k,v in recs.items()}
            saved = {"filter_keep":kept,"records":records,"summary":pol.summarize_policy(recs),"curve":curve}
            save(result_path.name,saved)
            results[arm][str(ps)] = records
            log(f"{arm} seed {ps}: success {saved['summary']['success_rate']['mean']:.3f}")
        seen[tuple(sorted(kept))] = arm
    means = {arm: [float(np.mean([r["success"] for r in byseed[str(ps)].values()])) for ps in p_cfg["seeds"]] for arm,byseed in results.items()}
    comparisons = {base: paired(results["phosted_jev_filt"],results[base],p_cfg["seeds"],len(starts))
                   for base in ["phybrid_filt", "pgemini_conf_filt", "p0", "pref"]}
    summary = {"arm_means":{a: {"mean":float(np.mean(v)),"per_seed":v} for a,v in means.items()},
               "hosted_minus_comparisons":comparisons,"policy_config":p_cfg,"filter_keep":arms}
    save("policy_records.json",results)
    save("policy_results.json",summary)
    log(json.dumps(summary["arm_means"]))
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--decisions-only",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    config.require_integrity()
    split=data.load_split(config.TASK);pool=data.pilot_subset(split)
    save("provenance.json", {"started":time.time(), "plan_sha256":data.sha256_file(ROOT/"research/jev/E3B_HOSTED_JEV_PLAN.md"),
         "client_sha256":data.sha256_file(Path(__file__).with_name("hosted_jev.py")),
         "runner_sha256":data.sha256_file(__file__),"shared_config":config.SHARED})
    manifest=decisions(pool)
    if args.decisions_only: return
    selection=calibrate_and_select(manifest,pool)
    summary=policies(selection,split,pool)
    save("complete.json",{"completed":time.time(),"cost_usd":manifest["cost_usd"],"gemini_calls":0,"locked_test_access":False,"policy_results":"policy_results.json"})
    log("Hosted Jev follow-up complete")


if __name__=="__main__":
    main()
