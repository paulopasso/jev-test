"""Regenerate paper figures from completed E3/E3b JSON. No model calls."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"figs"
PACKAGED_RUNS=ROOT/"results/runs"
RUNS=PACKAGED_RUNS if PACKAGED_RUNS.exists() else Path.home()/".local/share/jev_test/runs"
def read(path): return json.loads(path.read_text())
e3=read(RUNS/"lift_e3pol_none_seed0/e3pol.json")
e3r=read(RUNS/"lift_e3pol_none_seed0/e3pol_records.json")
e3b=read(RUNS/"lift_e3b_hosted_jev_seed0/policy_results.json")
e3br=read(RUNS/"lift_e3b_hosted_jev_seed0/policy_records.json")
ground=read(RUNS/"lift_e3ground_none_seed0/e3ground.json")
risk=read(RUNS/"lift_e3b_hosted_jev_seed0/calibration.json")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "savefig.bbox":"tight"})

def records(arm):
    return e3br[arm] if arm in e3br else {str(s):e3r[f"{arm}_pseed{s}"] for s in range(3)}

def paired(arm,base):
    a,b=records(arm),records(base)
    m=np.array([[float(a[str(s)][str(i)]["success"])-float(b[str(s)][str(i)]["success"])
                 for i in range(50)] for s in range(3)])
    for s in range(3):
        for i in range(50):
            assert a[str(s)][str(i)]["start_seed"]==b[str(s)][str(i)]["start_seed"]
    rng=np.random.default_rng(731)
    draws=[m[np.ix_(rng.integers(3,size=3),rng.integers(50,size=50))].mean() for _ in range(10000)]
    return float(m.mean()),np.quantile(draws,[.025,.975])

def save(fig,name):
    fig.savefig(OUT/f"{name}.pdf")
    fig.savefig(OUT/f"{name}.svg")
    plt.close(fig)

arms=[("p0","Gold actions only (24 episodes)"),
      ("pb2","Inverse-dynamics actions (16 added)"),
      ("pb2_filt","Inverse-dynamics actions (8 selected)"),
      ("pjv","Gemini-grounded actions (16 added)"),
      ("pjv_filt","Gemini-grounded actions (8 selected)"),
      ("phybrid","IDM + Gemini hybrid actions (16 added)"),
      ("phybrid_filt","Hybrid actions (8; original risk filter)"),
      ("pgemini_conf_filt","Hybrid actions (8; Gemini-confidence filter)"),
      ("phosted_jev_filt","Hybrid actions (8; Jev decision filter)"),
      ("pref","Gold actions only (40 episodes)")]
colors={"p0":"#888888","pb2":"#7192b0","pb2_filt":"#416884",
        "pjv":"#d9a15b","pjv_filt":"#b87831","phybrid":"#a48ab2",
        "phybrid_filt":"#765496","pgemini_conf_filt":"#e1ba72",
        "phosted_jev_filt":"#3d866e","pref":"#777777"}
fig,ax=plt.subplots(figsize=(7.2,4.3),layout="constrained")
for j,(arm,label) in enumerate(arms):
    y=len(arms)-j-1;v=e3b["arm_means"].get(arm) or e3["arm_means"][arm]
    ax.barh(y,100*v["mean"],color=colors[arm],height=.64,alpha=.9)
    ax.scatter(100*np.array(v["per_seed"]),[y]*3,s=17,facecolors="white",edgecolors="black",linewidths=.7,zorder=3)
    ax.text(63,y,f"{100*v['mean']:.1f}",va="center",fontsize=7)
ax.set(yticks=range(len(arms)),yticklabels=[a[1] for a in arms[::-1]],xlabel="Lift success (%)",xlim=(0,69),title="Policy success by pseudo-action source and selection method")
ax.grid(axis="x",linestyle=":",alpha=.35);ax.set_axisbelow(True)
save(fig,"policy_results")

contrasts=[("pjv","p0","Gemini-grounded actions vs 24 gold only"),
           ("pjv","pb2","Gemini-grounded vs inverse-dynamics actions"),
           ("phybrid","pb2","Hybrid vs inverse-dynamics actions; all 16"),
           ("phybrid_filt","pb2_filt","Hybrid vs inverse-dynamics; selected 8"),
           ("phosted_jev_filt","pgemini_conf_filt","Jev vs Gemini-confidence episode filter"),
           ("phosted_jev_filt","phybrid_filt","Jev vs original-risk episode filter")]
fig,ax=plt.subplots(figsize=(7.2,2.9),layout="constrained")
for j,(arm,base,label) in enumerate(contrasts):
    y=len(contrasts)-j-1;mean,ci=paired(arm,base)
    col="#3d866e" if arm=="phosted_jev_filt" else "#416884"
    ax.plot(100*ci,[y,y],color=col,lw=2);ax.scatter(100*mean,y,color=col,s=23,zorder=3)
ax.axvline(0,color="black",lw=.7,linestyle="--")
ax.set(yticks=range(len(contrasts)),yticklabels=[x[2] for x in contrasts[::-1]],
       xlabel="Success difference (percentage points)",xlim=(-30,47),
       title="Paired differences; two-way seed/start bootstrap intervals")
ax.grid(axis="x",linestyle=":",alpha=.35);ax.set_axisbelow(True)
save(fig,"policy_contrasts")

rec=ground["reconstruction_eval_holdout"]
items=[("B3","Feature-difference inverse dynamics"),
       ("B2","DINOv2 inverse dynamics"),
       ("JV-HYBRID","Inverse dynamics + Gemini hybrid"),
       ("G-state+z","Gemini descriptions + grounding net"),
       ("G-state+shuffled","Grounding net + shuffled descriptions"),
       ("G-state-only","State-only grounding control"),
       ("JV-DIRECT","Gemini direct seven-action prompt")]
fig,ax=plt.subplots(figsize=(7.2,3.1),layout="constrained")
for j,(key,label) in enumerate(items):
    y=len(items)-j-1;v=rec[key]["pos_mae"]
    ax.barh(y,v,color="#db9855" if key.startswith("JV") or key=="G-state+z" else "#6689a9",height=.7)
    ax.text(v+.004,y,f"{v:.3f}",va="center",fontsize=7)
ax.set(yticks=range(len(items)),yticklabels=[x[1] for x in items[::-1]],xlim=(0,.28),
       xlabel="Position-action MAE (native action units)",title="Held-out action reconstruction; lower is better")
ax.grid(axis="x",linestyle=":",alpha=.35);ax.set_axisbelow(True)
save(fig,"action_error")

fig,ax=plt.subplots(figsize=(6.8,2.9),layout="constrained")
for key,label,col in [("hosted_raw","Jev decision score (calibrated)","#3d866e"),
                      ("gemini_raw","Gemini uncertainty (calibrated)","#c88735"),
                      ("existing_hybrid","Original hybrid risk score","#416884")]:
    m=risk["arms"][key]
    if key!="existing_hybrid":m=m["calibrated"]
    vals=sorted((float(c),v) for c,v in m["coverage"].items())
    ax.plot([100*c for c,v in vals],[100*v for c,v in vals],marker="o",ms=3,color=col,
            label=f"{label} (AUC {m['auc']:.3f})")
ax.axhline(100*risk["eval_bad_rate"],color="0.45",linestyle="--",lw=1,label="All transitions")
ax.set(xlabel="Transitions retained (%)",ylabel="Bad-action rate (%)",ylim=(0,100),xlim=(17,103),
       title="Held-out error selection; lower event rate is better")
ax.legend(loc="upper left",fontsize=7,frameon=False,ncol=2)
save(fig,"risk_coverage")
print("Wrote four data-derived figures to",OUT)
