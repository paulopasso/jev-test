"""Consumer W training: identical architecture/optimizer across all arms.

Data per arm (pilot, frozen):
  b0   : gold-labeled train episodes only (24)
  b2/b3/j0 : gold (24) + pseudo-labeled remaining pool episodes (16)
  ref  : all train-pool episodes gold-labeled (40)
The consumer never sees latent z; it consumes native actions only.
Model selection: best dev-8 (labeled dev subset) rollout error.
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import config, models
from .train_annot import DEV
from .evaluate import rollout_error


def _build_supervised(eps, gold_ids, pseudo):
    """Returns X (B, 4, 2304) past frames, A (B, 7), Y (B, 2304) = f_{t+1}."""
    if set(gold_ids) & set(pseudo or {}):
        raise ValueError("Gold and pseudo episodes overlap")
    if not set(gold_ids) <= eps.gold or set(pseudo or {}) & (eps.select | eps.eval_only):
        raise ValueError("Training data role violation")
    k = config.SHARED["consumer"]["kwargs"]["ctx"]
    X, A, Y = [], [], []
    for d in gold_ids:
        f = eps.norm_feat(eps.eps[d]["feat"])
        a = eps.norm_act(eps.acts[d])
        for t in range(len(f) - 1):
            ctx = np.stack([f[max(0, i)] for i in range(t-k+1, t+1)])
            X.append(ctx); A.append(a[t]); Y.append(f[t + 1])
    for d, pa in (pseudo or {}).items():
        f = eps.norm_feat(eps.eps[d]["feat"])
        if len(pa) != len(f) - 1 or not np.isfinite(pa).all():
            raise ValueError("Invalid pseudo-label length or values")
        a = eps.norm_act(pa)
        for t in range(len(f) - 1):
            ctx = np.stack([f[max(0, i)] for i in range(t-k+1, t+1)])
            X.append(ctx); A.append(a[t]); Y.append(f[t + 1])
    return (torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(A)),
            torch.from_numpy(np.stack(Y)))


def train_consumer(eps, gold_ids, pseudo, dev_ids, log, seed, steps):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ck = config.SHARED["consumer"]
    model = models.ConsumerW(**ck["kwargs"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=ck["lr"], weight_decay=1e-4)
    X, A, Y = [t.to(DEV) for t in _build_supervised(eps, gold_ids, pseudo)]
    n = X.shape[0]
    log(f"  [consumer] training transitions: {n} (gold eps {len(gold_ids)}, pseudo eps {len(pseudo or {})})")
    ctx = ck["kwargs"]["ctx"]
    best_dev, best_state = float("inf"), None
    for it in range(steps):
        idx = torch.randint(0, n, (ck["batch"],), device=DEV)
        pred = model(X[idx], A[idx])
        loss = F.mse_loss(pred, Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 250 == 0 or it == steps - 1:
            model.eval()
            dev_err, _ = rollout_error(model, eps, dev_ids, [1, 5, 10], ctx)
            m10 = dev_err["mean_1_10"]
            log(f"  [consumer] step {it} train {loss.item():.5f} dev_roll {dev_err}")
            if m10 < best_dev:
                best_dev = m10
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.train()
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_dev
