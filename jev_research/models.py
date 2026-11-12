"""All trainable heads. Frozen DINOv2 tokens are the only visual inputs.

Annotators (produce native-action pseudo-labels for unlabeled episodes):
  B2  direct probabilistic inverse model: learned temporal head over
      (f_{t-1}, f_t, f_{t+1}) -> Gaussian(6D) + Bernoulli(gripper).
  B3  frozen-feature pair/difference baseline: explicit (f_{t+1} - f_t, f_t)
      -> MLP -> same distribution head. No learned temporal mixing.
  J0  continuous Jev: q_phi over (f_{t-1}, f_t, f_{t+1}) -> 16D Gaussian
      latent with KL bottleneck; forward predictor F(f_t, z) -> f_{t+1};
      grounding decoder D(z, f_t) -> action distribution (labeled subset only).

Consumer W (identical across arms, never sees z): transformer over past K
projected frame tokens + candidate native action embedding -> next frame tokens.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TOKEN_DIM = 2304  # 6 tokens x 384
ACT_DIM = 7       # 6 OSC deltas + 1 gripper


class ActionHead(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.cont = nn.Linear(hidden, 12)   # 6 means + 6 log-vars
        self.grip = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x)
        c = self.cont(h)
        return c[:, :6], c[:, 6:].clamp(-8, 8), self.grip(h).squeeze(-1)


class FrameProjector(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.proj = nn.Linear(TOKEN_DIM, out_dim)

    def forward(self, tokens):  # (B, T, 2304)
        return self.proj(tokens)


class B2Inverse(nn.Module):
    """Direct probabilistic IDM: small transformer over the context window."""

    def __init__(self, d=256, nlayers=2, heads=4, ctx=3):
        super().__init__()
        self.ctx = ctx
        self.proj = FrameProjector(d)
        self.pos = nn.Parameter(torch.zeros(1, ctx, d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, nlayers)
        self.head = ActionHead(d)

    def forward(self, frames):  # (B, ctx, 2304) ordered [t-1, t, t+1]
        h = self.tr(self.proj(frames) + self.pos)
        return self.head(h[:, -1])


class B3FeatureDiff(nn.Module):
    """Frozen-feature pair/difference baseline: explicit difference, MLP."""

    def __init__(self, d=256):
        super().__init__()
        self.head = ActionHead(2 * TOKEN_DIM, hidden=d)

    def forward(self, frames):  # (B, ctx, 2304); uses only f_t and f_{t+1}-f_t
        f_t = frames[:, 1]
        diff = frames[:, 2] - frames[:, 1]
        return self.head(torch.cat([f_t, diff], -1))


class JevEncoder(nn.Module):
    """q_phi(z | context): 16D diagonal Gaussian with KL bottleneck."""

    def __init__(self, zdim=16, d=256, nlayers=2, heads=4, ctx=3):
        super().__init__()
        self.ctx = ctx
        self.proj = FrameProjector(d)
        self.pos = nn.Parameter(torch.zeros(1, ctx, d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, nlayers)
        self.mu = nn.Linear(d, zdim)
        self.logvar = nn.Linear(d, zdim)

    def forward(self, frames):
        h = self.tr(self.proj(frames) + self.pos)[:, -1]
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)


class JevPredictor(nn.Module):
    """F(f_t, z) -> predicted normalized next-frame tokens."""

    def __init__(self, zdim=16, d=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TOKEN_DIM + zdim, d), nn.GELU(),
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, TOKEN_DIM),
        )

    def forward(self, f_t, z):
        return self.net(torch.cat([f_t, z], -1))


class JevDecoder(nn.Module):
    """D(z, f_t) -> native action distribution; trained on labeled subset only."""

    def __init__(self, zdim=16, d=256):
        super().__init__()
        self.head = ActionHead(zdim + TOKEN_DIM, hidden=d)

    def forward(self, z, f_t):
        return self.head(torch.cat([z, f_t], -1))


class ConsumerW(nn.Module):
    """Independent action-conditioned world model over pooled tokens.

    Input: past K frame tokens + one candidate native action.
    Output: predicted next-frame tokens (2304). No future access, no z.
    """

    def __init__(self, d=256, nlayers=2, heads=4, ctx=4):
        super().__init__()
        self.ctx = ctx
        self.proj = nn.Linear(TOKEN_DIM, d)
        self.pos = nn.Parameter(torch.zeros(1, ctx + 1, d))
        self.act = nn.Linear(ACT_DIM, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, nlayers)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, TOKEN_DIM))

    def forward(self, frames, actions):
        """frames (B, K, 2304), actions (B, 7) -> (B, 2304)"""
        B = frames.shape[0]
        tokens = torch.cat([self.proj(frames),
                            self.act(actions).unsqueeze(1)], 1) + self.pos
        h = self.tr(tokens)
        return self.out(h[:, -1])


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
