"""E3 visual-Jev experiment: pretrained VLM transition understanding +
small embodiment grounding, tested against local B2 inverse dynamics.

Phases (fixed run command; variant.json selects):
  e3vlm    : VLM annotation passes (SEMANTIC schema on gold/unlabeled/calib/
             eval-holdout; DIRECT schema on unlabeled/calib/eval; 2-frame
             ablation subset). Cached, resumable, fail-closed.
  e3ground : grounding G + ablations + probes + native-action reconstruction.
  e3cal    : JV risk maps on reserved calib + coverage on eval-holdout.
  e3pol    : 8-arm BC policy utility, closed-loop Lift success (PRIMARY).

Locked test is never touched. See research/jev/E3_PLAN.md (frozen).
"""
import hashlib
import io
import json
import os

import h5py
import numpy as np

from . import config, data

SEMANTIC_VERSION = "semantic_v2_episode_xml"
DIRECT_VERSION = "direct_v2_episode_xml"
RENDER_SIZE = 256
# Primary temporal context: (t-1, t, t+1, t+2) with edge duplication.
# Ablation context: (t, t+1).
PRIMARY_OFFSETS = (-1, 0, 1, 2)
ABLATION_OFFSETS = (0, 1)

INTERACTIONS = ["approach", "grasp", "lift", "transport", "place", "retreat", "other"]
MAG = {"none": 0, "small": 1, "medium": 2, "large": 3}
GRIP_EV = {"open": [1, 0, 0], "close": [0, 1, 0], "none": [0, 0, 1]}
CONTACT = {"none": [1, 0, 0, 0], "made": [0, 1, 0, 0],
           "broken": [0, 0, 1, 0], "maintained": [0, 0, 0, 1]}
AMB = {"low": 0, "medium": 1, "high": 2}
Z_DIM = 24

SEMANTIC_PROMPT = """You observe 4 consecutive frames from a fixed robot camera \
(frame 0 oldest, frame 3 newest) showing a Franka Panda arm manipulating a \
small cube on a table. Describe ONLY the physical transformation between the \
middle frames (frame 1 -> frame 2); use outer frames for motion context.

Reply with ONLY this JSON object (no other text):
{"ee_motion": {"x": -1|0|1, "y": -1|0|1, "z": -1|0|1, \
"mag": "none|small|medium|large"}, \
"rotation": {"roll": -1|0|1, "pitch": -1|0|1, "yaw": -1|0|1}, \
"gripper_event": "open|close|none", "object_motion": "none|small|large", \
"contact_change": "none|made|broken|maintained", \
"interaction": "approach|grasp|lift|transport|place|retreat|other", \
"confidence": 0.0-1.0, "ambiguity": "low|medium|high"}

Rules: x/y/z are end-effector motion DIRECTION signs in the camera view \
(right/up/toward-viewer positive); use 0 when no visible motion on that axis. \
Report what is VISIBLE, not what you infer about motor commands. If the \
gripper or cube is occluded, say so via high ambiguity, do not guess motion."""


def direct_prompt(examples_text):
    return """You observe 2 consecutive frames (BEFORE, AFTER) from a fixed \
camera of a Franka Panda arm with parallel gripper. Predict the robot action \
that caused this transition as 7 numbers in the controller input units:
dx,dy,dz in [-1,1] (end-effector position deltas, +5cm per unit at 20Hz), \
droll,dpitch,dyaw in [-1,1] (+0.5rad per unit), gripper in {-1 for close, +1 \
for open}. Also report your confidence 0.0-1.0. Reply with ONLY this JSON: \
{"dx": f, "dy": f, "dz": f, "droll": f, "dpitch": f, "dyaw": f, \
"gripper": -1|1, "confidence": 0.0-1.0}.

Labeled examples (frame pairs like yours, with TRUE actions):
""" + examples_text


def _sgn(v):
    v = int(v)
    if v not in (-1, 0, 1):
        raise ValueError(f"sign not in -1/0/1: {v}")
    return float(v)


def parse_semantic(content):
    o = json.loads(content)
    ee = o["ee_motion"]
    z = [_sgn(ee["x"]), _sgn(ee["y"]), _sgn(ee["z"]),
         MAG[ee["mag"]] / 3.0]
    r = o["rotation"]
    z += [_sgn(r["roll"]), _sgn(r["pitch"]), _sgn(r["yaw"])]
    z += [float(x) for x in GRIP_EV[o["gripper_event"]]]
    z += [({"none": 0, "small": 1, "large": 2}[o["object_motion"]]) / 2.0]
    z += [float(x) for x in CONTACT[o["contact_change"]]]
    inter = [0.0] * len(INTERACTIONS)
    inter[INTERACTIONS.index(o["interaction"])] = 1.0
    z += inter
    conf = float(o["confidence"])
    if not 0.0 <= conf <= 1.0:
        raise ValueError("confidence out of range")
    z += [conf, AMB[o["ambiguity"]] / 2.0]
    assert len(z) == Z_DIM, len(z)
    return {"z": [float(x) for x in z], "raw": o}


def parse_direct(content):
    o = json.loads(content)
    vals = [float(o[k]) for k in ("dx", "dy", "dz", "droll", "dpitch", "dyaw")]
    if any(not -1.0 <= v <= 1.0 for v in vals):
        raise ValueError("direct action out of [-1,1]")
    g = int(o["gripper"])
    if g not in (-1, 1):
        raise ValueError("gripper not in {-1,1}")
    conf = float(o["confidence"])
    if not 0.0 <= conf <= 1.0:
        raise ValueError("confidence out of range")
    return {"a": vals + [float(g)], "confidence": conf, "raw": o}


def few_shot_text():
    """3 FIXED in-context examples from in-budget gold labels (ids logged)."""
    from . import data as _data
    split = _data.load_split(config.TASK)
    pool = _data.pilot_subset(split)
    demos = pool["labeled_train"][:3]
    t = 5
    low = _data.load_lowdim_episodes(config.TASK, demos)
    lines = []
    for d in demos:
        a = np.asarray(low[d]["actions"], dtype=float)[t]
        lines.append(
            f"example {d} transition {t}: TRUE action "
            f"[dx={a[0]:.3f}, dy={a[1]:.3f}, dz={a[2]:.3f}, droll={a[3]:.3f}, "
            f"dpitch={a[4]:.3f}, dyaw={a[5]:.3f}, gripper={int(a[6])}]")
    return "\n".join(lines), [{"demo": d, "t": t} for d in demos]


def render_pngs(demo, idxs, size=RENDER_SIZE, env=None):
    """Render agentview RGB PNG bytes for recorded state indices.

    Images only. The caller must never load actions for VLM-target demos;
    this function reads STATES (observation-generation path, same as the
    E0-audited feature cache) and returns pixels.
    """
    from PIL import Image
    own = env is None
    if own:
        env = _make_render_env(size)
    from .encode_cache import close_env, reset_episode
    out = []
    try:
        with h5py.File(config.DATASETS[config.TASK]["raw"], "r") as src:
            group = src["data"][demo]
            for n, i in enumerate(idxs):
                # Episode-specific scene XML is required: physics state alone
                # does not reconstruct the recorded camera/scene reliably.
                o = (reset_episode(env, group, int(i)) if n == 0 else
                     env.reset_to({"states": group["states"][int(i)]}))
                buf = io.BytesIO()
                Image.fromarray(o["agentview_image"]).save(buf, format="PNG")
                out.append(buf.getvalue())
    finally:
        if own:
            close_env(env)
    return out


def _make_render_env(size):
    import json as _json
    import robomimic.utils.env_utils as EnvUtils
    from . import env_setup  # noqa: F401
    with h5py.File(config.DATASETS[config.TASK]["raw"], "r") as src:
        env_meta = _json.loads(src["data"].attrs["env_args"])
    return EnvUtils.create_env_for_data_processing(
        env_meta=env_meta, camera_names=["agentview"],
        camera_height=size, camera_width=size, reward_shaping=False)


def frame_hashes(pngs):
    return [hashlib.sha256(p).hexdigest()[:16] for p in pngs]


def transitions_of(demo):
    """Transition indices t=0..L-2 for a demo (L = recorded state count)."""
    with h5py.File(config.DATASETS[config.TASK]["raw"], "r") as src:
        n_states = src["data"][demo]["states"].shape[0]
    return list(range(n_states - 1))


def context_idxs(t, n_states, offsets):
    return [min(max(t + o, 0), n_states - 1) for o in offsets]


def assert_no_action_leak(target_demos, eps):
    """Fail-closed: VLM targets must have no loaded actions anywhere."""
    leaked = [d for d in target_demos
              if d in getattr(eps, "acts", {})]
    if leaked:
        raise RuntimeError(f"ACTION LEAK into VLM targets: {leaked}")


def vlm_target_sets():
    """Frozen demo sets for VLM annotation (same split as E2)."""
    split = data.load_split(config.TASK)
    pool = data.pilot_subset(split)
    unlabeled = [d for d in pool["train_pool"] if d not in pool["labeled_train"]]
    return {"gold": pool["labeled_train"], "unlabeled": unlabeled,
            "calib": pool["calib"], "eval": pool["eval_holdout"],
            "dev_labeled": pool["dev_labeled"]}


# ---------------------------------------------------------------------------
# VLM annotation passes (phase e3vlm)
# ---------------------------------------------------------------------------

def build_jobs(demos, offsets, schema_tag):
    """Render transition contexts. Returns jobs + render stats.

    One job per transition t: {demo, t, tids, frames, frame_hashes}.
    Rendering is per-demo (each demo rendered once); images only.
    """
    import time as _time
    t0 = _time.time()
    env = _make_render_env(RENDER_SIZE)
    jobs = []
    n_frames = 0
    try:
        for d in demos:
            ts = transitions_of(d)
            n_states = len(ts) + 1
            need = sorted({i for t in ts for i in context_idxs(t, n_states, offsets)})
            pngs = render_pngs(d, need, size=RENDER_SIZE, env=env)
            n_frames += len(pngs)
            art = {i: p for i, p in zip(need, pngs)}
            fh = {i: h for i, h in zip(need, frame_hashes(pngs))}
            for t in ts:
                idxs = context_idxs(t, n_states, offsets)
                jobs.append({"demo": d, "t": t, "tids": [t],
                             "frames": [art[i] for i in idxs],
                             "frame_hashes": [fh[i] for i in idxs]})
    finally:
        from .encode_cache import close_env
        close_env(env)
    return jobs, {"demos": len(demos), "jobs": len(jobs),
                  "frames_rendered": n_frames,
                  "seconds": _time.time() - t0}


def run_e3vlm(log):
    """Production VLM passes. Fail-closed. Writes e3vlm.json."""
    import time as _time
    from . import vlm as _vlm
    t_start = _time.time()
    sets = vlm_target_sets()
    log(f"[e3vlm] model={_vlm.MODEL} prompt_versions={SEMANTIC_VERSION},{DIRECT_VERSION}")
    log(f"[e3vlm] sets: " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()))

    few_text, few_ids = few_shot_text()
    log(f"[e3vlm] DIRECT few-shot examples (in-budget gold): {few_ids}")

    # Smoke validation BEFORE production (fail-closed gate).
    sj = build_jobs([sets["gold"][0]], PRIMARY_OFFSETS, "smoke")[0][:5]
    n_parse = 0
    key = _vlm.load_key()
    for j in sj:
        msgs = _vlm.user_content(SEMANTIC_PROMPT, j["frames"])
        content, _, _ = _vlm.chat(msgs, temperature=0,
                                  response_format={"type": "json_object"},
                                  max_tokens=2000, key=key)
        parse_semantic(content)
        n_parse += 1
    assert n_parse == 5, "smoke parse gate failed"
    log("[e3vlm] smoke gate 5/5 parses OK (no production data yet)")

    dmsg = direct_prompt(few_text)

    def sem_messages(job):
        return _vlm.user_content(SEMANTIC_PROMPT, job["frames"])

    def dir_messages(job):
        return _vlm.user_content(dmsg, job["frames"])

    results = {"model": _vlm.MODEL,
               "semantic_version": SEMANTIC_VERSION,
               "direct_version": DIRECT_VERSION,
               "render_size": RENDER_SIZE,
               "primary_offsets": list(PRIMARY_OFFSETS),
               "ablation_offsets": list(ABLATION_OFFSETS),
               "few_shot": few_ids}
    def chunked(schema, version, demos, offsets, messages, parse_fn,
                  max_tokens, chunk_demos=6):
        """Render + request one demo-chunk at a time (memory-bounded).
        The shared JSONL cache makes every chunk independently resumable."""
        import gc as _gc
        merged = None
        for i in range(0, len(demos), chunk_demos):
            part = demos[i:i + chunk_demos]
            jobs, rinfo = build_jobs(part, offsets, schema)
            log(f"[e3vlm:{schema}] chunk {i // chunk_demos + 1}: "
                f"{rinfo['demos']} demos, {rinfo['jobs']} jobs, "
                f"{rinfo['frames_rendered']} frames")
            s = _vlm.run_pass(schema, version, _vlm.MODEL, jobs, messages,
                              parse_fn, max_tokens, log)
            if merged is None:
                merged = dict(s)
            else:
                for k in ("requests", "ok", "parse_fail", "http_fail",
                          "retries", "prompt_tokens", "completion_tokens",
                          "jobs", "ok_total", "rate_limit_waits"):
                    merged[k] = merged.get(k, 0) + s.get(k, 0)
                merged["cache_sha256"] = s["cache_sha256"]
                merged["cache_path"] = s["cache_path"]
            del jobs
            _gc.collect()
        merged["est_cost_usd"] = (merged["prompt_tokens"] * 0.75e-6
                                  + merged["completion_tokens"] * 3.75e-6)
        return merged

    def require_full(key):
        r = results[key]
        if r["ok_total"] != r["jobs"]:
            raise RuntimeError(
                f"E3 INVALID: VLM pass {key} incomplete "
                f"({r['ok_total']}/{r['jobs']} ok). Cache preserves progress; "
                f"relaunch resumes the missing records. No fallback.")

    # 1. SEMANTIC primary: gold + unlabeled + calib + eval
    sem_demos = sets["gold"] + sets["unlabeled"] + sets["calib"] + sets["eval"]
    results["semantic"] = chunked("semantic", SEMANTIC_VERSION, sem_demos,
                                  PRIMARY_OFFSETS, sem_messages,
                                  parse_semantic, 2000)
    require_full("semantic")
    # 2. DIRECT: unlabeled + calib + eval (no grounding training needed)
    dir_demos = sets["unlabeled"] + sets["calib"] + sets["eval"]
    results["direct"] = chunked("direct", DIRECT_VERSION, dir_demos,
                                ABLATION_OFFSETS, dir_messages,
                                parse_direct, 800)
    require_full("direct")
    # 3. SEMANTIC 2-frame ablation on fixed stratified subset of unlabeled
    unl = sets["unlabeled"]
    sub = []
    for d in unl:
        ts = transitions_of(d)
        sub += [(d, t) for t in ts[::4]]
    log(f"[e3vlm] ablation subset: {len(sub)} transitions")
    env = _make_render_env(RENDER_SIZE)
    ajobs = []
    try:
        for d in sorted({x[0] for x in sub}):
            ts = [t for (dd, t) in sub if dd == d]
            n_states = len(transitions_of(d)) + 1
            need = sorted({i for t in ts for i in context_idxs(t, n_states, ABLATION_OFFSETS)})
            pngs = render_pngs(d, need, size=RENDER_SIZE, env=env)
            art = {i: p for i, p in zip(need, pngs)}
            fh = {i: h for i, h in zip(need, frame_hashes(pngs))}
            for t in ts:
                idxs = context_idxs(t, n_states, ABLATION_OFFSETS)
                ajobs.append({"demo": d, "t": t, "tids": [t],
                              "frames": [art[i] for i in idxs],
                              "frame_hashes": [fh[i] for i in idxs]})
    finally:
        from .encode_cache import close_env
        close_env(env)
    results["ablation_2frame"] = _vlm.run_pass(
        "semantic_abl2", SEMANTIC_VERSION + "+2frame", _vlm.MODEL, ajobs,
        sem_messages, parse_semantic, 2000, log)

    total_req = (results["semantic"]["requests"] + results["direct"]["requests"]
                 + results["ablation_2frame"]["requests"])
    if total_req == 0:
        raise RuntimeError("E3 INVALID: OpenRouter requests = 0 (fail-closed)")
    # Full coverage is required: pseudo bundles need complete (L-1, 7)
    # support and grounding needs complete gold pairs. Anything missing
    # fails closed here (cache preserves progress for investigation/retry).
    for key in ("semantic", "direct", "ablation_2frame"):
        r = results[key]
        if r["ok_total"] != r["jobs"]:
            raise RuntimeError(
                f"E3 INVALID: VLM pass {key} incomplete "
                f"({r['ok_total']}/{r['jobs']} ok). No fallback.")
    results["total_requests"] = total_req
    results["total_est_cost_usd"] = round(
        results["semantic"]["est_cost_usd"] + results["direct"]["est_cost_usd"]
        + results["ablation_2frame"]["est_cost_usd"], 2)
    results["wall_seconds"] = _time.time() - t_start
    out = os.path.join(config.run_dir(), "e3vlm.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=float)
    log(f"[e3vlm] wrote {out}: {total_req} requests, "
        f"${results['total_est_cost_usd']:.2f} est, {results['wall_seconds']:.0f}s")
    return results


# ---------------------------------------------------------------------------
# Cached VLM representation access
# ---------------------------------------------------------------------------

def load_z_records(schema):
    """{ (demo, t): parsed_payload } for ok records of a VLM schema pass."""
    from . import vlm as _vlm
    recs, _ = _vlm.load_cache(schema)
    out = {}
    for r in recs.values():
        if r.get("status") == "ok" and r.get("payload"):
            out[(r["demo"], tuple(r["tids"])[0])] = r["payload"]["parsed"]
    return out


def z_matrix(schema, demos, nt):
    """(sum L-1, 24) z arrays per demo. Raises on any missing transition."""
    recs = load_z_records(schema)
    out = {}
    for d in demos:
        rows = []
        for t in range(nt[d]):
            p = recs.get((d, t))
            if p is None:
                raise RuntimeError(f"missing VLM {schema} record for {(d, t)}")
            rows.append(p["z"])
        out[d] = np.asarray(rows, dtype=np.float32)
    return out


def direct_matrix(demos, nt):
    """{demo: (L-1, 7) raw-unit direct VLM actions}. Raises on missing."""
    recs = load_z_records("direct")
    out = {}
    for d in demos:
        rows = []
        for t in range(nt[d]):
            p = recs.get((d, t))
            if p is None:
                raise RuntimeError(f"missing VLM direct record for {(d, t)}")
            rows.append(p["a"])
        out[d] = np.asarray(rows, dtype=np.float32)
    return out


def transition_counts(demos):
    return {d: len(transitions_of(d)) for d in demos}


# ---------------------------------------------------------------------------
# Grounding models (small MLPs + ActionHead; 24-gold training only)
# ---------------------------------------------------------------------------

def make_grounding(in_dim, seed):
    import torch
    from . import models
    torch.manual_seed(seed)
    g = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 256), torch.nn.GELU(),
        models.ActionHead(256, 256))
    return g


def train_head(model, X, Yc, Yg, seed, log, tag, steps=3000, batch=64, lr=1e-3):
    import torch
    import torch.nn.functional as F
    from .train_annot import DEV
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X).to(DEV)
    Yc_t = torch.from_numpy(Yc).to(DEV)
    Yg_t = torch.from_numpy(Yg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model = model.to(DEV)
    n = Xt.shape[0]
    for it in range(steps):
        idx = torch.randint(0, n, (batch,), device=DEV)
        mu, logvar, lg = model(Xt[idx])
        loss = F.gaussian_nll_loss(mu, Yc_t[idx], logvar.exp(), full=True,
                                   eps=1e-6) + F.binary_cross_entropy_with_logits(
            lg, Yg_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0:
            log(f"  [ground:{tag}] step {it} nll {loss.item():.4f}")
    model.eval()
    return model


def predict_head(model, X):
    import torch
    from .train_annot import DEV
    model = model.to(DEV)
    model.eval()
    out_mu, out_lv, out_pg = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            Xt = torch.from_numpy(X[i:i + 512]).to(DEV)
            mu, logvar, lg = model(Xt)
            out_mu.append(mu.cpu().numpy())
            out_lv.append(logvar.cpu().numpy())
            out_pg.append(torch.sigmoid(lg).cpu().numpy())
    return (np.concatenate(out_mu, 0).astype(np.float32),
            np.concatenate(out_lv, 0).astype(np.float32),
            np.concatenate(out_pg, 0).astype(np.float32))


def denorm6(mu_norm, a_mean, a_std):
    return mu_norm * np.asarray(a_std[:6], dtype=np.float32) + np.asarray(
        a_mean[:6], dtype=np.float32)


def map_actions(mu_raw6, p_grip):
    g = np.where(np.asarray(p_grip) >= 0.5, 1.0, -1.0)
    return np.concatenate([np.asarray(mu_raw6, dtype=np.float32),
                           g.reshape(-1, 1).astype(np.float32)], 1)


def sample_grip(p_grip, seed):
    rng = np.random.default_rng(seed)
    return np.where(rng.random(len(p_grip)) < np.asarray(p_grip), 1.0, -1.0).astype(np.float32)


def shuffled_z(Z, seed=999):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(Z.shape[0])
    return Z[perm]


# ---------------------------------------------------------------------------
# Phase e3ground: grounding + ablations + probes + reconstruction
# ---------------------------------------------------------------------------

def run_e3ground(log):
    import time as _time
    import torch
    from .e2 import _setup_episodes
    from .train_annot import train_annotator, DEV
    from . import calibrate as calmod
    t_start = _time.time()
    seed = config.VARIANT.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = config.run_dir()
    split, pool, eps = _setup_episodes(load_cache=False, log=log)
    sets = vlm_target_sets()
    gold, unlabeled = sets["gold"], sets["unlabeled"]
    a_mean = np.asarray(eps.norm["a_mean"], dtype=np.float32)
    a_std = np.asarray(eps.norm["a_std"], dtype=np.float32)

    nt_gold = transition_counts(gold)
    for d in gold:
        nf = eps.eps[d]["feat"].shape[0]
        # Each robomimic episode stores L actions for L states. The final
        # action has no successor state, so only actions 0..L-2 pair with
        # the L-1 rendered transitions and VLM annotations below.
        if nf - 1 != nt_gold[d] or eps.acts[d].shape[0] != nf:
            raise RuntimeError(f"transition misalignment for {d}: "
                               f"frames={nf} states-derived={nt_gold[d] + 1} "
                               f"actions={eps.acts[d].shape[0]}")
    Z_train = z_matrix("semantic", gold, nt_gold)
    Zg = np.concatenate([Z_train[d] for d in gold], 0)
    Fg = np.concatenate([eps.norm_feat(eps.eps[d]["feat"])[:nt_gold[d]] for d in gold], 0)
    Ag = np.concatenate([eps.norm_act(eps.acts[d][:nt_gold[d]]) for d in gold], 0)
    Yc, Yg = Ag[:, :6].astype(np.float32), (Ag[:, 6] > 0).astype(np.float32)
    log(f"[e3ground] grounding pairs: {len(Yc)} (24 gold demos)")

    FeatureDim = Fg.shape[1]
    variants = {
        "state+z": np.concatenate([Zg, Fg], 1),
        "state-only": Fg,
        "state+shuffled": np.concatenate([shuffled_z(Zg), Fg], 1),
    }
    grounds = {}
    for tag, X in variants.items():
        log(f"[e3ground] training G({tag}) in_dim={X.shape[1]} ...")
        grounds[tag] = train_head(make_grounding(X.shape[1], seed),
                                  X.astype(np.float32), Yc, Yg, seed, log, tag)

    # B2/B3 reload (E2 checkpoints) + replication check; J0 single retrain.
    from . import models as _models
    e2cal_dir = os.path.join(config.RUNS_DIR, f"{config.TASK}_e2cal_none_seed{seed}")
    ck = config.SHARED["annotator"]
    loaded = {}
    for arm in ("b2", "b3"):
        cls = {"b2": _models.B2Inverse, "b3": _models.B3FeatureDiff}[arm]
        m = cls(**(ck["kwargs"].get(arm) or {}))
        m.load_state_dict(torch.load(os.path.join(e2cal_dir, f"annot_e2_{arm}_seed{seed}.pt"),
                                     map_location="cpu"))
        m = m.to(DEV)
        m.eval()
        loaded[arm] = {"model": m, "type": arm}
    from .evaluate import annotation_accuracy
    rep_check = {}
    for arm in ("b2", "b3"):
        acc = annotation_accuracy(loaded[arm], eps, pool["dev_labeled"])
        rep_check[arm] = acc["pos_mae"]
    log(f"[e3ground] B2/B3 reload replication dev pos-MAE: {rep_check} "
        f"(E2 refs 0.1247 / 0.1239)")
    for arm, ref in (("b2", 0.1247), ("b3", 0.1239)):
        if abs(rep_check[arm] - ref) / ref > 0.05:
            raise RuntimeError(f"{arm} replication drift: {rep_check[arm]} vs {ref}")
    log("[e3ground] retraining J0 annotator (single seed-0, no tuning) ...")
    j0 = train_annotator("j0", eps, pool["labeled_train"], pool["train_pool"],
                         ck["steps"], log, seed)
    torch.save([x.state_dict() for x in j0["model"]],
               os.path.join(out_dir, "annot_e3_j0_seed0.pt"))

    # HYBRID fusion: B2 frozen outputs + z + state, trained on 24 gold.
    # Rows built demo-by-demo to keep transition alignment exact.
    from .train_annot import predict_dist
    b2gold = predict_dist(loaded["b2"], eps, gold)
    Hrows = []
    for d in gold:
        n = nt_gold[d]
        Hrows.append(np.concatenate(
            [b2gold[d]["mu"][:n], b2gold[d]["logvar"][:n].mean(1, keepdims=True),
             b2gold[d]["p_grip"][:n].reshape(-1, 1), Z_train[d],
             eps.norm_feat(eps.eps[d]["feat"])[:n]], 1))
    Hf = np.concatenate(Hrows, 0).astype(np.float32)
    log(f"[e3ground] training HYBRID fusion in_dim={Hf.shape[1]} ...")
    hybrid = train_head(make_grounding(Hf.shape[1], seed), Hf, Yc, Yg, seed, log, "hybrid")

    for tag, m in list(grounds.items()) + [("hybrid", hybrid)]:
        torch.save(m.state_dict(), os.path.join(out_dir, f"ground_e3_{tag.replace('+', '_')}.pt"))

    # --- reconstruction on eval-holdout (eval-only labels, disclosed) ---
    eval_ids = pool["eval_holdout"]
    nt_eval = transition_counts(eval_ids)
    Z_eval = z_matrix("semantic", eval_ids, nt_eval)
    Fe = {d: eps.norm_feat(eps.eps[d]["feat"])[:nt_eval[d]] for d in eval_ids}
    A_true = {d: eps.acts[d][:nt_eval[d]].astype(np.float32) for d in eval_ids}
    e2cal = json.load(open(os.path.join(e2cal_dir, "e2cal.json")))
    taus = e2cal["thresholds"]

    # Shuffled control for eval: one fixed global permutation over the
    # stacked eval z (mirrors the seed-999 training control); sliced back
    # per demo so shapes align exactly.
    Zstack = np.concatenate([Z_eval[d] for d in eval_ids], 0)
    Zs = shuffled_z(Zstack, seed=999)
    offs = np.cumsum([0] + [nt_eval[d] for d in eval_ids])
    Xe = {
        "state+z": np.concatenate(
            [np.concatenate([Z_eval[d], Fe[d]], 1) for d in eval_ids], 0),
        "state-only": np.concatenate([Fe[d] for d in eval_ids], 0),
        "state+shuffled": np.concatenate(
            [np.concatenate([Zs[offs[i]:offs[i + 1]], Fe[d]], 1)
             for i, d in enumerate(eval_ids)], 0),
    }

    Ae = np.concatenate([A_true[d] for d in eval_ids], 0)
    recon = {}
    for tag in ("state+z", "state-only", "state+shuffled"):
        mu_n, lv, pg = predict_head(grounds[tag], Xe[tag].astype(np.float32))
        pred = map_actions(denorm6(mu_n, a_mean, a_std), pg)
        ev = calmod.error_events(Ae, pred, taus["tau_pos"], taus["tau_rot"])
        recon[f"G-{tag}"] = _recon_stats(Ae, pred, ev, lv, pg, a_std)
    # B2 / B3 / J0 via predict_dist
    for arm, am in (("b2", loaded["b2"]), ("b3", loaded["b3"]), ("j0", j0)):
        dd = predict_dist(am, eps, eval_ids)
        pred = np.concatenate(
            [np.concatenate([dd[d]["mu"][:nt_eval[d]],
                             np.where(dd[d]["p_grip"][:nt_eval[d]] >= 0.5, 1.0, -1.0
                                      ).reshape(-1, 1)], 1) for d in eval_ids], 0)
        lv = np.concatenate([dd[d]["logvar"][:nt_eval[d]] for d in eval_ids], 0)
        pg = np.concatenate([dd[d]["p_grip"][:nt_eval[d]] for d in eval_ids], 0)
        ev = calmod.error_events(Ae, pred, taus["tau_pos"], taus["tau_rot"])
        recon[arm.upper()] = _recon_stats(Ae, pred, ev, lv, pg, a_std)
    # DIRECT VLM outputs
    D_eval = direct_matrix(eval_ids, nt_eval)
    pred = np.concatenate([D_eval[d] for d in eval_ids], 0)
    ev = calmod.error_events(Ae, pred, taus["tau_pos"], taus["tau_rot"])
    recon["JV-DIRECT"] = _recon_stats(Ae, pred, ev, None, None, a_std)
    # HYBRID
    b2eval = predict_dist(loaded["b2"], eps, eval_ids)
    Hrows = [np.concatenate(
        [b2eval[d]["mu"][:nt_eval[d]], b2eval[d]["logvar"][:nt_eval[d]].mean(1, keepdims=True),
         b2eval[d]["p_grip"][:nt_eval[d]].reshape(-1, 1), Z_eval[d], Fe[d]], 1)
        for d in eval_ids]
    mu_n, lv, pg = predict_head(hybrid, np.concatenate(Hrows, 0).astype(np.float32))
    pred = map_actions(denorm6(mu_n, a_mean, a_std), pg)
    ev = calmod.error_events(Ae, pred, taus["tau_pos"], taus["tau_rot"])
    recon["JV-HYBRID"] = _recon_stats(Ae, pred, ev, lv, pg, a_std)
    # JV-SEMANTIC pseudo path = G(state+z) predictions (same as above)
    recon["JV-SEMANTIC"] = recon["G-state+z"]
    log("[e3ground] reconstruction (eval-holdout, disclosed eval-only labels):")
    for k, v in recon.items():
        log(f"  {k}: pos {v['pos_mae']:.4f} rot {v['rot_mae']:.4f} grip {v['grip_err']:.4f}")

    # --- representation probes (gold-train / eval-test) ---
    probes = _run_probes(Zg, Ag, Zstack, Ae, log)

    out = {"phase": "e3ground", "seed": seed,
           "replication_dev_pos_mae": rep_check,
           "reconstruction_eval_holdout": recon, "probes": probes,
           "taus": taus, "wall_seconds": _time.time() - t_start}
    with open(os.path.join(out_dir, "e3ground.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    log(f"[e3ground] wrote e3ground.json to {out_dir}")
    return out


def _recon_stats(Ae, pred, ev, lv, pg, a_std):
    pe = np.abs(Ae[:, :3] - pred[:, :3])
    re_ = np.abs(Ae[:, 3:6] - pred[:, 3:6])
    mag = np.abs(Ae[:, :6]).mean(1)
    tert = np.quantile(mag, [1 / 3, 2 / 3])
    by_mag = {}
    for i, (lo, hi) in enumerate([(0, tert[0]), (tert[0], tert[1]), (tert[1], 9e9)]):
        m = (mag >= lo) & (mag < hi)
        by_mag[f"tercile{i}"] = float(pe[m].mean()) if m.sum() else None
    closed = Ae[:, 6] < 0
    s = {"pos_mae": float(pe.mean()), "rot_mae": float(re_.mean()),
         "grip_err": float(ev["bad_grip"].mean()),
         "pos_mae_p50": float(np.median(ev["pos_err"])),
         "pos_mae_p90": float(np.quantile(ev["pos_err"], 0.9)),
         "per_dim_mae": [float(x) for x in np.abs(Ae - pred).mean(0)],
         "by_magnitude_pos_mae": by_mag,
         "grip_open_phase_pos_mae": float(pe[~closed].mean()) if (~closed).sum() else None,
         "grip_closed_phase_pos_mae": float(pe[closed].mean()) if closed.sum() else None,
         "bad_any_rate": float(ev["bad_any"].mean())}
    if lv is not None:
        var = np.exp(np.asarray(lv, dtype=float)) * (
            np.asarray(a_std[:6], dtype=float) ** 2)
        se = ((pred[:, :6] - Ae[:, :6]) ** 2 / np.maximum(var, 1e-12)).sum(1)
        s["gauss_nll_raw_proxy"] = float(0.5 * (np.log(np.maximum(var, 1e-12)).sum(1) + se).mean())
    return s


def _run_probes(Zg, Ag, Zstack, Ae, log):
    """Direction + gripper probes: gold-train, eval-test, trivial baselines."""
    from sklearn.linear_model import LogisticRegression
    out = {}
    for ax, name in enumerate(["dx", "dy", "dz"]):
        ytr = np.sign(Ag[:, ax]).astype(int)
        yte = np.sign(Ae[:, ax]).astype(int)
        maj = float((yte == np.bincount((yte + 1).astype(int), minlength=3).argmax() - 1).mean())
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Zg, ytr + 1)
        acc = float(clf.score(Zstack, yte + 1))
        out[f"dir_{name}"] = {"probe_acc": acc, "majority": maj}
    gtr = (np.abs(np.diff(Ag[:, 6], prepend=Ag[:1, 6])) > 0).astype(int)
    gte = (np.abs(np.diff(Ae[:, 6], prepend=Ae[:1, 6])) > 0).astype(int)
    base = float(1 - gte.mean())
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Zg, gtr)
    out["gripper_event"] = {"probe_acc": float(clf.score(Zstack, gte)),
                            "no_change_baseline": base}
    log(f"[e3ground] probes: {json.dumps(out)}")
    return out


# ---------------------------------------------------------------------------
# Phase e3cal: JV risk maps (reserved calib) + coverage (eval-holdout)
# ---------------------------------------------------------------------------

def run_e3cal(log):
    """Fit isotonic risk maps for JV-SEMANTIC / JV-HYBRID / JV-DIRECT on the
    8 reserved calib episodes (same taus as E2); coverage vs random on the
    12 eval-holdout episodes. Writes e3cal.json + risk maps + filter lists."""
    import time as _time
    import torch
    from .e2 import _setup_episodes
    from .train_annot import predict_dist, DEV
    from . import calibrate as calmod
    t_start = _time.time()
    seed = config.VARIANT.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = config.run_dir()
    split, pool, eps = _setup_episodes(load_cache=False, log=log)
    sets = vlm_target_sets()
    unlabeled = sets["unlabeled"]
    a_mean = np.asarray(eps.norm["a_mean"], dtype=np.float32)
    a_std = np.asarray(eps.norm["a_std"], dtype=np.float32)

    e2cal_dir = os.path.join(config.RUNS_DIR, f"{config.TASK}_e2cal_none_seed{seed}")
    e2cal = json.load(open(os.path.join(e2cal_dir, "e2cal.json")))
    taus = e2cal["thresholds"]
    log(f"[e3cal] reused E2 thresholds (identical bad-event definition): {taus}")

    nt_cal = transition_counts(pool["calib"])
    nt_eval = transition_counts(pool["eval_holdout"])
    nt_unl = transition_counts(unlabeled)
    Z_cal = z_matrix("semantic", pool["calib"], nt_cal)
    Z_eval = z_matrix("semantic", pool["eval_holdout"], nt_eval)
    Z_unl = z_matrix("semantic", unlabeled, nt_unl)

    def F_of(demos, nt):
        return {d: eps.norm_feat(eps.eps[d]["feat"])[:nt[d]] for d in demos}

    Fc, Fe, Fu = F_of(pool["calib"], nt_cal), F_of(pool["eval_holdout"], nt_eval), F_of(unlabeled, nt_unl)
    Xc = np.concatenate([np.concatenate([Z_cal[d], Fc[d]], 1) for d in pool["calib"]], 0)
    Xe_ = np.concatenate([np.concatenate([Z_eval[d], Fe[d]], 1) for d in pool["eval_holdout"]], 0)
    Xu = np.concatenate([np.concatenate([Z_unl[d], Fu[d]], 1) for d in unlabeled], 0)

    gpath = os.path.join(config.RUNS_DIR, f"{config.TASK}_e3ground_none_seed{seed}",
                         "ground_e3_state_z.pt")
    hpath = os.path.join(config.RUNS_DIR, f"{config.TASK}_e3ground_none_seed{seed}",
                         "ground_e3_hybrid.pt")
    G = make_grounding(Xc.shape[1], seed)
    G.load_state_dict(torch.load(gpath, map_location="cpu"))
    # HYBRID input dim = 6 + 1 + 1 + 24 + F; rebuild exactly:
    from .train_annot import predict_dist as _pd
    from . import models as _models
    ck = config.SHARED["annotator"]
    b2m = _models.B2Inverse(**(ck["kwargs"].get("b2") or {}))
    b2m.load_state_dict(torch.load(os.path.join(e2cal_dir, f"annot_e2_b2_seed{seed}.pt"),
                                   map_location="cpu"))
    b2m = b2m.to(DEV)
    b2m.eval()
    b2 = {"model": b2m, "type": "b2"}

    def hybrid_rows(demos, nt, Z, F):
        dd = _pd(b2, eps, demos)
        return np.concatenate([np.concatenate(
            [dd[d]["mu"][:nt[d]], dd[d]["logvar"][:nt[d]].mean(1, keepdims=True),
             dd[d]["p_grip"][:nt[d]].reshape(-1, 1), Z[d], F[d]], 1) for d in demos], 0)

    H = make_grounding(hybrid_rows(pool["calib"], nt_cal, Z_cal, Fc).shape[1], seed)
    H.load_state_dict(torch.load(hpath, map_location="cpu"))

    out = {"phase": "e3cal", "seed": seed, "thresholds": taus, "arms": {}}
    for tag, model, Xc_, Xe__, Xu_ in (
            ("jv-semantic", G, Xc, Xe_, Xu),
            ("jv-hybrid", H,
             hybrid_rows(pool["calib"], nt_cal, Z_cal, Fc),
             hybrid_rows(pool["eval_holdout"], nt_eval, Z_eval, Fe),
             hybrid_rows(unlabeled, nt_unl, Z_unl, Fu))):
        mu_c, lv_c, pg_c = predict_head(model, Xc_.astype(np.float32))
        A_cal = np.concatenate([eps.acts[d][:nt_cal[d]].astype(np.float32)
                                for d in pool["calib"]], 0)
        ev_c = calmod.error_events(A_cal, map_actions(denorm6(mu_c, a_mean, a_std), pg_c),
                                   taus["tau_pos"], taus["tau_rot"])
        csc = [{"score_cont": float(v), "score_grip": float(g)}
               for v, g in zip(lv_c.mean(1), calmod.bernoulli_entropy(pg_c))]
        risk = calmod.fit_risk_maps(
            csc, {"bad_pos": ev_c["bad_pos"], "bad_rot": ev_c["bad_rot"],
                  "bad_grip": ev_c["bad_grip"]})
        risk.update({"thresholds": taus, "n_calib_episodes": len(pool["calib"])})
        with open(os.path.join(out_dir, f"risk_maps_{tag}.json"), "w") as f:
            json.dump(risk, f, indent=1)

        mu_e, lv_e, pg_e = predict_head(model, Xe__.astype(np.float32))
        A_eval = np.concatenate([eps.acts[d][:nt_eval[d]].astype(np.float32)
                                 for d in pool["eval_holdout"]], 0)
        ev_e = calmod.error_events(A_eval, map_actions(denorm6(mu_e, a_mean, a_std), pg_e),
                                   taus["tau_pos"], taus["tau_rot"])
        flat_risk = calmod.predict_risk(
            {"score_cont": lv_e.mean(1),
             "score_grip": calmod.bernoulli_entropy(pg_e)}, risk)["risk_any"]
        flat_any = ev_e["bad_any"].astype(int)
        auc = calmod.rank_auc(flat_risk, flat_any)
        t_cfg = config.SHARED["e2"]["risk"]
        rng = np.random.default_rng(t_cfg["random_seed"])
        curves = {}
        for c in t_cfg["coverages"]:
            m = calmod.select_lowest_risk(flat_risk, c)
            rm = [calmod.random_mask(len(flat_risk), c, rng) for _ in range(t_cfg["random_baseline_resamples"])]
            curves[str(c)] = {
                "risk_any_bad": float(flat_any[m].mean()),
                "random_any_bad": float(np.mean([flat_any[x].mean() for x in rm]))}
        mu_u, lv_u, pg_u = predict_head(model, Xu_.astype(np.float32))
        ru = calmod.predict_risk(
            {"score_cont": lv_u.mean(1),
             "score_grip": calmod.bernoulli_entropy(pg_u)}, risk)["risk_any"]
        offs = np.cumsum([0] + [nt_unl[d] for d in unlabeled])
        ep_risk = {d: float(ru[offs[i]:offs[i + 1]].mean())
                   for i, d in enumerate(unlabeled)}
        ranked = sorted(unlabeled, key=lambda d: ep_risk[d])
        keep = ranked[:t_cfg["filter_keep_episodes"]]
        out["arms"][tag] = {"auc_any": auc,
                            "base_any_bad_rate": float(flat_any.mean()),
                            "curves": curves, "episode_mean_risk": ep_risk,
                            "episode_ranking": ranked, "filter_keep": keep}
        log(f"[e3cal] {tag}: AUC={auc:.3f} base={flat_any.mean():.3f} keep={keep}")

    # JV-DIRECT: single confidence map (documented simplification).
    D_cal = direct_matrix(pool["calib"], nt_cal)
    D_eval = direct_matrix(pool["eval_holdout"], nt_eval)
    DREC = load_z_records("direct")
    conf_c = np.array([DREC[(d, t)]["confidence"]
                       for d in pool["calib"] for t in range(nt_cal[d])], dtype=float)
    A_cal = np.concatenate([eps.acts[d][:nt_cal[d]].astype(np.float32) for d in pool["calib"]], 0)
    P_cal = np.concatenate([D_cal[d] for d in pool["calib"]], 0)
    ev_c = calmod.error_events(A_cal, P_cal, taus["tau_pos"], taus["tau_rot"])
    dmap = calmod.isotonic_fit(1.0 - conf_c, ev_c["bad_any"].astype(float))
    A_eval = np.concatenate([eps.acts[d][:nt_eval[d]].astype(np.float32) for d in pool["eval_holdout"]], 0)
    P_eval = np.concatenate([D_eval[d] for d in pool["eval_holdout"]], 0)
    ev_e = calmod.error_events(A_eval, P_eval, taus["tau_pos"], taus["tau_rot"])
    conf_e = np.array([DREC[(d, t)]["confidence"]
                       for d in pool["eval_holdout"] for t in range(nt_eval[d])], dtype=float)
    r_e = calmod.isotonic_predict(dmap, 1.0 - conf_e)
    auc_d = calmod.rank_auc(r_e, ev_e["bad_any"].astype(int))
    out["arms"]["jv-direct"] = {"auc_any": auc_d,
                                "base_any_bad_rate": float(ev_e["bad_any"].mean()),
                                "note": "single confidence map; per-component risks = bad_any value"}
    log(f"[e3cal] jv-direct: AUC={auc_d:.3f}")
    out["wall_seconds"] = _time.time() - t_start
    with open(os.path.join(out_dir, "e3cal.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    log(f"[e3cal] wrote e3cal.json to {out_dir}")
    return out


# ---------------------------------------------------------------------------
# Phase e3pol: 8-arm BC policy utility (PRIMARY metric)
# ---------------------------------------------------------------------------

def _pseudo_bundle(predict_fn, demos, nt, seed, log, tag):
    """Build {demo: (L-1,7) raw} pseudo bundle with sampled gripper (E2 conv)."""
    out = {}
    for d in demos:
        mu_raw6, _, pg = predict_fn(d)
        g = sample_grip(pg, seed + (abs(hash(d)) % 1000))
        out[d] = np.concatenate([np.asarray(mu_raw6, dtype=np.float32),
                                 g.reshape(-1, 1)], 1).astype(np.float32)
        assert out[d].shape == (nt[d], 7), (d, out[d].shape, nt[d])
    log(f"[e3pol] bundle {tag}: {len(out)} demos")
    return out


def run_e3pol(log):
    """Eight policy arms x 3 seeds, identical protocol to E2. Writes
    e3pol.json + e3pol_records.json. PRIMARY E3 metric."""
    import time as _time
    import torch
    from . import policy as pol
    from .e2 import _setup_episodes
    from .train_annot import predict_dist, DEV
    t_start = _time.time()
    t_cfg = config.SHARED["e2"]
    p_cfg = t_cfg["policy"]
    seed = config.VARIANT.get("seed", 0)
    out_dir = config.run_dir()
    split, pool, eps = _setup_episodes(load_cache=False, log=log)
    sets = vlm_target_sets()
    unlabeled = sets["unlabeled"]
    a_mean = np.asarray(eps.norm["a_mean"], dtype=np.float32)
    a_std = np.asarray(eps.norm["a_std"], dtype=np.float32)
    train_pool = pool["train_pool"]

    e2cal_dir = os.path.join(config.RUNS_DIR, f"{config.TASK}_e2cal_none_seed{seed}")
    e2cal = json.load(open(os.path.join(e2cal_dir, "e2cal.json")))
    e3ground_dir = os.path.join(config.RUNS_DIR, f"{config.TASK}_e3ground_none_seed{seed}")
    e3cal = json.load(open(os.path.join(
        config.RUNS_DIR, f"{config.TASK}_e3cal_none_seed{seed}", "e3cal.json")))

    def load_e2_pseudo(arm):
        arr_dir = os.path.join(e2cal_dir, f"bundle_e2_{arm}", "arrays")
        return {d: np.load(os.path.join(arr_dir, f"{d}_pseudo_actions.npy")).astype(np.float32)
                for d in e2cal["unlabeled"]}

    pseudo_b2 = load_e2_pseudo("b2")
    filt_b2 = [d for d in e2cal["filter_keep"] if d in pseudo_b2]
    log(f"[e3pol] E2 B2 bundle reused ({len(pseudo_b2)} demos, filter {filt_b2})")

    # JV predictors (posterior-mean cont + p_grip) on unlabeled demos.
    nt_unl = transition_counts(unlabeled)
    Z_unl = z_matrix("semantic", unlabeled, nt_unl)
    Fu = {d: eps.norm_feat(eps.eps[d]["feat"])[:nt_unl[d]] for d in unlabeled}
    G = make_grounding(24 + Fu[unlabeled[0]].shape[1], seed)
    G.load_state_dict(torch.load(os.path.join(e3ground_dir, "ground_e3_state_z.pt"),
                                 map_location="cpu"))
    G.eval()

    from . import models as _models
    ck = config.SHARED["annotator"]
    b2m = _models.B2Inverse(**(ck["kwargs"].get("b2") or {}))
    b2m.load_state_dict(torch.load(os.path.join(e2cal_dir, f"annot_e2_b2_seed{seed}.pt"),
                                   map_location="cpu"))
    b2m = b2m.to(DEV)
    b2m.eval()
    b2 = {"model": b2m, "type": "b2"}
    b2unl = predict_dist(b2, eps, unlabeled)
    H = make_grounding(6 + 1 + 1 + 24 + Fu[unlabeled[0]].shape[1], seed)
    H.load_state_dict(torch.load(os.path.join(e3ground_dir, "ground_e3_hybrid.pt"),
                                 map_location="cpu"))
    H.eval()

    def pred_G(d):
        X = np.concatenate([Z_unl[d], Fu[d]], 1).astype(np.float32)
        mu_n, _, pg = predict_head(G, X)
        return denorm6(mu_n, a_mean, a_std), None, pg

    def pred_H(d):
        X = np.concatenate(
            [b2unl[d]["mu"][:nt_unl[d]], b2unl[d]["logvar"][:nt_unl[d]].mean(1, keepdims=True),
             b2unl[d]["p_grip"][:nt_unl[d]].reshape(-1, 1), Z_unl[d], Fu[d]], 1).astype(np.float32)
        mu_n, _, pg = predict_head(H, X)
        return denorm6(mu_n, a_mean, a_std), None, pg

    pseudo_jv = _pseudo_bundle(pred_G, unlabeled, nt_unl, seed, log, "jv")
    pseudo_hyb = _pseudo_bundle(pred_H, unlabeled, nt_unl, seed, log, "hyb")

    filt_jv = [d for d in e3cal["arms"]["jv-semantic"]["filter_keep"] if d in pseudo_jv]
    filt_hyb = [d for d in e3cal["arms"]["jv-hybrid"]["filter_keep"] if d in pseudo_hyb]
    log(f"[e3pol] JV filters: {filt_jv} / {filt_hyb}")

    # persist JV bundles (same layout as E2 bundles for auditability)
    for tag, bundle in (("jv", pseudo_jv), ("hyb", pseudo_hyb)):
        adir = os.path.join(out_dir, f"bundle_e3_{tag}", "arrays")
        os.makedirs(adir, exist_ok=True)
        for d, a in bundle.items():
            np.save(os.path.join(adir, f"{d}_pseudo_actions.npy"), a)

    lowdim = data.load_lowdim_episodes(config.TASK, train_pool)
    stats = pol.obs_stats(train_pool, lowdim)
    arms = {
        "p0": (pool["labeled_train"], {}),
        "pb2": (pool["labeled_train"], pseudo_b2),
        "pb2_filt": (pool["labeled_train"], {d: pseudo_b2[d] for d in filt_b2}),
        "pjv": (pool["labeled_train"], pseudo_jv),
        "pjv_filt": (pool["labeled_train"], {d: pseudo_jv[d] for d in filt_jv}),
        "phybrid": (pool["labeled_train"], pseudo_hyb),
        "phybrid_filt": (pool["labeled_train"], {d: pseudo_hyb[d] for d in filt_hyb}),
        "pref": (train_pool, {}),
    }
    log("[e3pol] arms: " + ", ".join(
        f"{a}(gold={len(g)},pseudo={len(p)})" for a, (g, p) in arms.items()))
    datasets = {}
    for a, (g, p) in arms.items():
        X, A = pol.build_policy_dataset(g, p, lowdim, train_pool, split)
        datasets[a] = (X, A)
        log(f"[e3pol] {a}: {X.shape[0]} transitions")

    starts = [p_cfg["start_seed_base"] + i for i in range(p_cfg["n_starts"])]
    results = {}
    for a, (X, A) in datasets.items():
        for ps in p_cfg["seeds"]:
            key = f"{a}_pseed{ps}"
            log(f"[e3pol] training {key} ...")
            model, curve = pol.train_policy(
                X, A, stats, p_cfg["steps"], p_cfg["batch"], p_cfg["lr"],
                p_cfg["hidden"], p_cfg["layers"], log, ps, DEV)
            ckpt = os.path.join(out_dir, f"{key}.pt")
            torch.save(model.state_dict(), ckpt)
            model = model.to("cpu")
            log(f"[e3pol] evaluating {key} on {len(starts)} paired starts ...")
            recs = pol.evaluate_policy(model, stats, starts, p_cfg["horizon"], "cpu", log)
            summ = pol.summarize_policy(recs)
            log(f"[e3pol] {key}: success {summ['success_rate']['mean']:.3f} "
                f"[{summ['success_rate']['lo']}, {summ['success_rate']['hi']}] "
                f"modes {summ['failure_modes']}")
            results[key] = {"summary": summ,
                            "records": {str(k): {kk: v for kk, v in r.items()} for k, r in recs.items()},
                            "curve": curve, "ckpt": ckpt}

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
    # pooled paired Δ vs p0 AND vs pb2 (primary E3 comparisons)
    pooled = {}
    for a in arms:
        if a == "p0":
            continue
        for base in ("p0", "pb2"):
            if a == base:
                continue
            dall = []
            for ps in p_cfg["seeds"]:
                ra = results[f"{a}_pseed{ps}"]["records"]
                rb = results[f"{base}_pseed{ps}"]["records"]
                dall += [float(ra[str(i)]["success"]) - float(rb[str(i)]["success"])
                         for i in range(len(starts))]
            pooled[f"{a}_vs_{base}"] = {"mean": float(np.mean(dall)),
                                        "ci": bootstrap_ci(dall),
                                        "n": len(dall)}
    arm_means = {}
    for a in arms:
        vals = [results[f"{a}_pseed{ps}"]["summary"]["success_rate"]["mean"] for ps in p_cfg["seeds"]]
        arm_means[a] = {"mean": float(np.mean(vals)),
                        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "per_seed": vals}
    d_ref = arm_means["pref"]["mean"] - arm_means["p0"]["mean"]
    rlv = {}
    for a in ("pb2", "pb2_filt", "pjv", "pjv_filt", "phybrid", "phybrid_filt"):
        d = arm_means[a]["mean"] - arm_means["p0"]["mean"]
        rlv[a] = float(d / d_ref) if d_ref > 0 else None
    e3pol = {"phase": "e3pol", "arms": list(arms), "policy_seeds": p_cfg["seeds"],
             "n_starts": len(starts), "arm_means": arm_means,
             "per_seed_delta_vs_p0": per_seed_delta, "pooled_deltas": pooled,
             "recovered_label_value": rlv,
             "denominator_pref_minus_p0": float(d_ref),
             "results": {k: {"summary": v["summary"], "curve": v["curve"], "ckpt": v["ckpt"]}
                         for k, v in results.items()},
             "wall_seconds": _time.time() - t_start}
    with open(os.path.join(out_dir, "e3pol.json"), "w") as f:
        json.dump(e3pol, f, indent=1, default=float)
    with open(os.path.join(out_dir, "e3pol_records.json"), "w") as f:
        json.dump({k: v["records"] for k, v in results.items()}, f, default=float)
    log(f"[e3pol] arm means: {json.dumps(arm_means, indent=1)}")
    log(f"[e3pol] pooled deltas: {json.dumps(pooled, indent=1)}")
    log(f"[e3pol] RLV: {json.dumps(rlv)}")
    return e3pol
