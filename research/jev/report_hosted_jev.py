"""Generate the E3b report and figures from completed, saved artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN = Path.home()/".local/share/jev_test/runs/lift_e3b_hosted_jev_seed0"
DEST = ROOT/"research/jev/e3b_figures"


def main():
    if not (RUN/"complete.json").exists():
        raise RuntimeError("E3b has not completed")
    policy=json.loads((RUN/"policy_results.json").read_text())
    calib=json.loads((RUN/"calibration.json").read_text())
    manifest=json.loads((RUN/"decisions_manifest.json").read_text())
    DEST.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    order=["p0","phybrid","phybrid_filt","pgemini_conf_filt","phosted_jev_filt","pref"]
    labels=["24 gold","Hybrid\nunfiltered","Hybrid\nexisting\nfilter","Hybrid\nGemini\nconfidence","Hybrid\nhosted Jev\nfilter","40 gold\nreference"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw={"width_ratios":[1.4,1]},layout="constrained")
    ax=axes[0]
    values=[policy["arm_means"][a]["mean"]*100 for a in order]
    ax.bar(range(len(order)),values,color=["#a3a3a3","#839db7","#516e91","#e6a34d","#3b8d78","#a3a3a3"],alpha=.85)
    for i,a in enumerate(order):
        ax.scatter(i+np.array([-.12,0,.12]),np.array(policy["arm_means"][a]["per_seed"])*100,color="black",s=20,zorder=3)
    ax.set(xticks=range(len(order)),xticklabels=labels,ylabel="Lift success (%)",ylim=(0,100),title="Policy utility: three seeds × 50 shared starts")
    ax.tick_params(axis="x",labelsize=8)
    ax=axes[1]
    for i,(base,label) in enumerate([("phybrid_filt","vs existing hybrid filter"),("pgemini_conf_filt","vs Gemini-confidence filter")]):
        c=policy["hosted_minus_comparisons"][base];m=c["mean"]*100
        lo,hi=np.array(c["two_way_seed_start_ci"])*100
        ax.plot([lo,hi],[i,i],color="#3b8d78",linewidth=3)
        ax.scatter([m],[i],color="#214e43",zorder=3)
        ax.text(m,i+.16,f"{m:+.1f} pp",ha="center",fontsize=9)
    ax.axvline(0,color="gray",linestyle="--",linewidth=1)
    ax.set(yticks=[0,1],yticklabels=["vs existing hybrid filter","vs Gemini-confidence filter"],ylim=(-.5,1.5),xlabel="Hosted Jev success difference (percentage points)",title="Descriptive 95% paired intervals")
    fig.savefig(DEST/"policy_comparison.png",dpi=200)
    fig.savefig(DEST/"policy_comparison.svg")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,4),layout="constrained")
    for key,label in [("hosted_raw","Hosted Jev + isotonic"),("gemini_raw","Gemini confidence + isotonic"),("existing_hybrid","Existing hybrid risk")]:
        metrics=calib["arms"][key] if key=="existing_hybrid" else calib["arms"][key]["calibrated"]
        items=sorted((float(k),v) for k,v in metrics["coverage"].items())
        ax.plot([k*100 for k,v in items],[v*100 for k,v in items],marker="o",label=f"{label} (AUC {metrics['auc']:.3f})")
    ax.axhline(calib["eval_bad_rate"]*100,color="gray",linestyle="--",label="All-transition error rate")
    ax.set(xlabel="Transitions retained (%)",ylabel="Bad-action event rate (%)",ylim=(0,100),title="Heldout action-error ranking: lower is better")
    ax.legend(fontsize=8)
    fig.savefig(DEST/"risk_coverage.png",dpi=200);fig.savefig(DEST/"risk_coverage.svg")
    plt.close(fig)
    lines=["# Hosted TypeSafe Jev follow-up (E3b)","",
           "Completed follow-up using `typesafe/jev-1.13-20260917` through OpenRouter's Decisions API. This is the first experiment here that actually calls hosted TypeSafe Jev. Original E3 did not.","",
           f"Evaluated {manifest['n_transitions']:,} transition descriptions using {manifest['unique_calls_in_cache']} cached unique API requests, at provider-reported cost **${manifest['cost_usd']:.8f}**. No new Gemini calls. The 40-episode locked test was not accessed.","",
           "## Method","",
           "Hosted Jev judged motion ambiguity, contact ambiguity and internal inconsistency in cached Gemini semantic JSON. The maximum of these three Noul probabilities was calibrated to hybrid action-error events using only the 8 calibration episodes. A matched Gemini-confidence selector used the same calibration method. Each selected 8 of the same 16 unlabeled demonstrations. Both reused the exact saved hybrid pseudo-actions, combined with 24 gold demonstrations; policies used the frozen E3 training settings, 3 seeds and 50 paired starts per seed. Existing E3 baseline policy results were reused.","",
           "## Results","","| Policy | Mean success | Seeds 0 / 1 / 2 |","|---|---:|---|" ]
    for a in order:
        v=policy["arm_means"][a]
        lines.append(f"| {a} | {v['mean']*100:.1f}% | {' / '.join(f'{s*100:.1f}%' for s in v['per_seed'])} |")
    lines += ["","Primary comparisons were specified before E3b outputs, after E3 results were already known:",""]
    for base in ["phybrid_filt","pgemini_conf_filt"]:
        c=policy["hosted_minus_comparisons"][base];lo,hi=c["two_way_seed_start_ci"]
        lines.append(f"- Hosted Jev minus `{base}`: **{c['mean']*100:+.1f} pp**, two-way seed/start bootstrap 95% interval **[{lo*100:+.1f}, {hi*100:+.1f}] pp**. Per-seed differences: {', '.join(f'{v*100:+.1f}' for v in c['per_seed'])} pp. Pooled paired-start interval is also saved in the result JSON.")
    lines += ["","| Error-risk selector | Heldout AUC | Brier score |","|---|---:|---:|"]
    for key in ["hosted_raw","gemini_raw","existing_hybrid"]:
        m=calib["arms"][key] if key=="existing_hybrid" else calib["arms"][key]["calibrated"]
        label={"hosted_raw":"Hosted Jev, calibrated","gemini_raw":"Gemini confidence, calibrated","existing_hybrid":"Existing hybrid risk"}[key]
        lines.append(f"| {label} | {m['auc']:.4f} | {m['brier']:.4f} |")
    wins=all(policy['hosted_minus_comparisons'][b]['two_way_seed_start_ci'][0]>0 and sum(v>=0 for v in policy['hosted_minus_comparisons'][b]['per_seed'])>=2 for b in ['phybrid_filt','pgemini_conf_filt'])
    lines += ["","## Interpretation and limits","",
              ("This pilot supports further investigation of hosted Jev's selection layer against both specified controls." if wins else "This pilot does not establish an improvement from hosted Jev over both specified controls. Successful API integration must not be confused with a positive research result."),"",
              "Jev receives only Gemini's compressed textual descriptions. It cannot recover visual evidence that those descriptions omit. Repeated inputs were deduplicated by exact request hash; the transition count is not the number of independent Jev judgments. These are text-based uncertainty judgments calibrated to local errors, not direct continuous controls or a pretrained world model.","",
              "The 3 training seeds give limited uncertainty resolution. Paired intervals resample training seeds and shared simulator starts; they are descriptive, with no multiple-comparison adjustment. This reuses a pilot evaluation population seen in E3. An independent confirmatory evaluation remains necessary. Calibration-error events use MAP gripper predictions, while policy pseudo-actions reuse the previously sampled gripper bundle, matching E3's convention.","",
              "An existing tied-score issue in the legacy isotonic helper was found during integration. Both new selectors use sklearn isotonic regression; the historical hybrid selector is unchanged and its AUC was replicated. Consequently the matched Jev-versus-Gemini comparison isolates the new scoring method more cleanly than comparisons with the historical selector.","",
              "## Figures","",f"![Policy utility]({DEST/'policy_comparison.png'})","",f"![Risk–coverage]({DEST/'risk_coverage.png'})","",
              "Policy dots show each seed, not confidence intervals. Forest intervals use paired two-way seed/start bootstrap with 10,000 draws and RNG seed 731. Risk–coverage shows transition selection, whereas policies use episode-level selection. Ties in risk coverage retain stable transition order.","",
              "## Reproducibility","",f"Source artifacts: `{RUN}`. Protocol: `E3B_HOSTED_JEV_PLAN.md`. Client: `hosted_jev.py`. Runner: `run_hosted_jev.py`. Figures and this report: `report_hosted_jev.py`. See `HOSTED_JEV_API.md` for primary documentation and verified API details.",""]
    (ROOT/"research/jev/E3B_HOSTED_JEV_REPORT.md").write_text("\n".join(lines))
    print(ROOT/"research/jev/E3B_HOSTED_JEV_REPORT.md")


if __name__=="__main__": main()
