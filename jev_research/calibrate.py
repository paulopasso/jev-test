"""B2 uncertainty calibration on the RESERVED calibration episodes.

Jev v0 (MVP) = direct probabilistic inverse dynamics (B2) + calibrated
confidence/risk + selective pseudo-labeling + downstream policy utility.
J0 remains in the repo as an experimental latent-action arm (see models.py).

Role discipline (fail-closed):
  - The action predictor is NEVER trained on calibration episodes.
  - Calibration actions are used ONLY to fit/evaluate the risk maps here.
  - Pseudo-label targets (unlabeled train pool) are never scored against
    ground truth: their actions are never loaded.

Risk definition: for a transition with predicted action, predict the
probability that the prediction is "bad". Bad-events are defined from
ground-truth calibration actions with predeclared threshold rules:

    bad_position = mean|pos_err over dims 0..2| > tau_pos
    bad_rotation = mean|rot_err over dims 3..5| > tau_rot
    bad_gripper  = sign mismatch on dim 6

Threshold rule (predeclared, documented in E2_PLAN.md): each tau is the
median B2 error on the dev_labeled SELECTION episodes. Rationale: the ledger
reserves dev_labeled exactly for selection choices, so calib stays a clean
fitting set and eval-holdout a clean evaluation. Exact values are recorded
in the risk artifact.

Raw uncertainty signals (from predict_dist):
    score_cont = mean normalized-unit log-variance over the 6 cont dims
    score_grip = Bernoulli entropy of P(gripper=+1)

Mapping: isotonic regression (PAVA, dependency-free) of each binary event
on its score, fit on calibration transitions only. Outputs:

    risk = jev.predict_risk(transition, predicted_action)
         -> {risk_pos, risk_rot, risk_grip, risk_any} in [0, 1]
         0 = highly reliable, 1 = highly unreliable.

    risk_any = 1 - (1-risk_pos)(1-risk_rot)(1-risk_grip), documented as an
    independence approximation over the three failure modes.
"""
import numpy as np

COMPONENTS = ("pos", "rot", "grip")


def bernoulli_entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def transition_scores(dist):
    """Raw uncertainty scores from a predict_dist entry."""
    score_cont = float(np.asarray(dist["logvar"], dtype=float).mean())
    score_grip = float(bernoulli_entropy(np.asarray(dist["p_grip"], dtype=float)).mean())
    return {"score_cont": score_cont, "score_grip": score_grip}


def error_events(a_true, a_pred, tau_pos, tau_rot):
    """Per-transition errors and binary bad-events (raw action units)."""
    a_true = np.asarray(a_true, dtype=float)
    a_pred = np.asarray(a_pred, dtype=float)
    pos_err = np.abs(a_true[:, :3] - a_pred[:, :3]).mean(1)
    rot_err = np.abs(a_true[:, 3:6] - a_pred[:, 3:6]).mean(1)
    grip_bad = ((a_true[:, 6] > 0) != (a_pred[:, 6] > 0))
    return {
        "pos_err": pos_err, "rot_err": rot_err,
        "bad_pos": pos_err > tau_pos,
        "bad_rot": rot_err > tau_rot,
        "bad_grip": grip_bad,
        "bad_any": (pos_err > tau_pos) | (rot_err > tau_rot) | grip_bad,
    }


def median_thresholds(pos_err_train, rot_err_train):
    """Predeclared rule: tau = median B2 error on the selection set."""
    return {"tau_pos": float(np.median(pos_err_train)),
            "tau_rot": float(np.median(rot_err_train))}


def isotonic_fit(scores, labels):
    """PAVA isotonic regression: nondecreasing P(label=1 | score).

    Returns {"xs": sorted unique score knots, "ps": fitted probabilities}.
    Ties are pooled with weights; output is clipped to [0, 1].
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(scores) == 0:
        raise ValueError("Empty calibration set")
    if not np.isfinite(scores).all() or not np.isfinite(labels).all():
        raise ValueError("Nonfinite calibration inputs")
    if set(np.unique(labels)) - {0.0, 1.0}:
        raise ValueError("Isotonic labels must be binary")
    order = np.argsort(scores, kind="stable")
    s, y = scores[order], labels[order]
    # pool adjacent violators over (block sum, block weight)
    sums, weights, bounds = [], [], []
    for si, yi in zip(s, y):
        sums.append(yi); weights.append(1.0); bounds.append(si)
        while len(sums) >= 2 and sums[-2] / weights[-2] > sums[-1] / weights[-1]:
            sums[-2] += sums[-1]; weights[-2] += weights[-1]
            bounds[-2] = bounds[-1]
            del sums[-1]; del weights[-1]; del bounds[-1]
    xs = np.array(bounds)
    ps = np.clip(np.array(sums) / np.array(weights), 0.0, 1.0)
    return {"xs": xs.tolist(), "ps": ps.tolist()}


def isotonic_predict(mapping, scores):
    """Right-continuous step evaluation; out-of-range clamps to edge values."""
    xs = np.asarray(mapping["xs"], dtype=float)
    ps = np.asarray(mapping["ps"], dtype=float)
    q = np.asarray(scores, dtype=float)
    idx = np.searchsorted(xs, q, side="left")
    idx = np.clip(idx, 0, len(ps) - 1)
    # scores below the first knot map to the first block value (already),
    # scores above the last knot map to the last block value.
    return ps[idx]


def fit_risk_maps(calib_scores, calib_events):
    """Fit component risk maps on calibration transitions.

    calib_scores: list of {"score_cont", "score_grip"}; calib_events: dict
    of binary arrays (bad_pos/bad_rot/bad_grip). Returns a JSON-serializable
    artifact including fit-set sizes for provenance.
    """
    sc = np.array([s["score_cont"] for s in calib_scores], dtype=float)
    sg = np.array([s["score_grip"] for s in calib_scores], dtype=float)
    artifact = {
        "n_transitions": int(len(sc)),
        "maps": {
            "pos": isotonic_fit(sc, calib_events["bad_pos"]),
            "rot": isotonic_fit(sc, calib_events["bad_rot"]),
            "grip": isotonic_fit(sg, calib_events["bad_grip"]),
        },
    }
    return artifact


def predict_risk(scores, artifact):
    """Map raw scores to calibrated risks in [0, 1]. Single transition or batch.

    scores: {"score_cont": float|array, "score_grip": float|array}.
    Returns dict of floats (scalar input) or arrays (array input).
    """
    scalar = np.ndim(scores["score_cont"]) == 0
    rp = isotonic_predict(artifact["maps"]["pos"], scores["score_cont"])
    rr = isotonic_predict(artifact["maps"]["rot"], scores["score_cont"])
    rg = isotonic_predict(artifact["maps"]["grip"], scores["score_grip"])
    ra = 1.0 - (1.0 - rp) * (1.0 - rr) * (1.0 - rg)
    out = {"risk_pos": rp, "risk_rot": rr, "risk_grip": rg, "risk_any": ra}
    if scalar:
        return {k: float(v) for k, v in out.items()}
    return out


def reliability_table(risks, labels, bins=10):
    """Mean predicted risk vs empirical event rate per bin (diagnostic)."""
    risks = np.asarray(risks, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for b in range(bins):
        m = (risks >= edges[b]) & (risks < edges[b + 1] if b < bins - 1 else risks <= edges[b + 1])
        if m.sum():
            rows.append({"bin": [float(edges[b]), float(edges[b + 1])],
                         "n": int(m.sum()),
                         "mean_risk": float(risks[m].mean()),
                         "empirical_rate": float(labels[m].mean())})
    return rows


def rank_auc(risks, labels):
    """Ranking quality of the risk score (no threshold needed)."""
    from sklearn.metrics import roc_auc_score
    risks = np.asarray(risks, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, risks))


def select_lowest_risk(risks, coverage):
    """Boolean mask keeping the lowest-risk `coverage` fraction (ties: stable order)."""
    risks = np.asarray(risks, dtype=float)
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    k = max(1, int(round(len(risks) * coverage)))
    order = np.argsort(risks, kind="stable")
    mask = np.zeros(len(risks), dtype=bool)
    mask[order[:k]] = True
    return mask


def random_mask(n, coverage, rng):
    """Uniform random retention mask at identical coverage (baseline)."""
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    k = max(1, int(round(n * coverage)))
    idx = rng.choice(n, k, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask
