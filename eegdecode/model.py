"""Causal EEG hybrid with checkpoint-compatible A0 module names."""
import math

import torch
from torch import nn
from torch.nn import functional as F

def detach_state(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {k: detach_state(v) for k, v in value.items()}
    return value


class CausalStem(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.kernel = 250
        self.temporal = nn.Conv2d(1, 8, (1, self.kernel), bias=False)
        self.spatial = nn.Conv2d(8, 16, (28, 1), groups=8, bias=False)
        self.norm = nn.LayerNorm(16)
        self.project = nn.Linear(16, dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x, past=None):
        if past is None:
            past = x.new_zeros(x.shape[0], 28, self.kernel - 1)
        raw = torch.cat([past, x], dim=-1)
        features = self.spatial(self.temporal(raw.unsqueeze(1))).squeeze(2)
        features = F.elu(self.norm(features.transpose(1, 2))).transpose(1, 2)
        pooled = F.avg_pool1d(features, 25, 25).transpose(1, 2)
        return self.out_norm(self.project(pooled)), raw[:, :, -(self.kernel-1):]


class RetentionBlock(nn.Module):
    def __init__(self, dim=64, heads=4, ffn=128, dropout=.25):
        super().__init__()
        if dim % heads or (dim // heads) % 2 or heads != 4:
            raise ValueError("Retention requires four heads and an even head dimension")
        self.heads, self.head_dim = heads, dim // heads
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, dim))
        self.drop = nn.Dropout(dropout)
        self.register_buffer("gamma", torch.exp(-math.log(2) * .05 / torch.tensor([1., 2., 4., 8.])))
        self.register_buffer("rotary_freq", 10000. ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim))

    def _heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def _rotate(self, x, start):
        pos = torch.arange(start, start+x.shape[2], device=x.device, dtype=x.dtype)
        phase = pos[:, None] * self.rotary_freq[None, :]
        cos, sin = phase.cos()[None, None], phase.sin()[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack([even*cos-odd*sin, even*sin+odd*cos], dim=-1).flatten(-2)

    def forward(self, x, state=None, start=0):
        z = self.norm(x)
        q = self._rotate(self._heads(self.q(z)), start) / math.sqrt(self.head_dim)
        k = self._rotate(self._heads(self.k(z)), start)
        v = self._heads(self.v(z))
        length = x.shape[1]
        if state is None:
            state = x.new_zeros(x.shape[0], self.heads, self.head_dim, self.head_dim)
        idx = torch.arange(length, device=x.device)
        distance = idx[:, None] - idx[None, :]
        decay = self.gamma[:, None, None] ** distance.clamp_min(0)[None]
        decay = decay * (distance >= 0)[None]
        local = ((q @ k.transpose(-1, -2)) * decay[None]) @ v
        cross = (q @ state) * (self.gamma[:, None] ** (idx + 1)[None])[None, :, :, None]
        values = F.layer_norm(local + cross, (self.head_dim,))
        values = values.transpose(1, 2).reshape_as(x)
        y = x + self.drop(self.out(values * F.silu(self.gate(z))))
        y = y + self.drop(self.ff(self.ff_norm(y)))
        tail_weight = self.gamma[:, None] ** (length-1-idx)[None]
        new_state = state * self.gamma[None, :, None, None] ** length
        new_state = new_state + k.transpose(-1, -2) @ (v * tail_weight[None, :, :, None])
        return y, new_state


class LocalTransformer(nn.Module):
    def __init__(self, dim, heads, ffn, dropout, token_stride=1):
        super().__init__()
        if type(token_stride) is not int or token_stride not in (1, 2, 4):
            raise ValueError("token_stride must be integer 1, 2 or 4")
        self.token_stride = token_stride
        self.block = nn.TransformerEncoderLayer(dim, heads, ffn, dropout=dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
        pos = torch.arange(40).float()[:, None]
        freq = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.)/dim))
        pe = torch.zeros(40, dim)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos*freq), torch.cos(pos*freq)
        self.register_buffer("positions", pe)
        self.norm = nn.LayerNorm(dim)

    def forward(self, windows):
        # Each window ends at its own decision time. Full attention here is endpoint-causal.
        b, n, w, d = windows.shape
        if w != 40:
            raise ValueError("Expected the original trailing 40-token window")
        # Select inside EACH decision window, anchored to its newest token.
        # Retention has already consumed every 50 ms token; its state is untouched.
        indices = torch.arange(self.token_stride - 1, 40, self.token_stride, device=windows.device)
        y = windows.reshape(b*n, w, d).index_select(1, indices) + self.positions.index_select(0, indices)[None]
        return self.norm(self.block(y)[:, -1]).reshape(b, n, d)


class Hybrid(nn.Module):
    """A0_full_L10_safe student and causal K0 teacher backbone."""

    def __init__(self):
        super().__init__()
        self.stem = CausalStem(64)
        self.retention = RetentionBlock(64, 4, 128, 0.25)
        self.transformer = LocalTransformer(64, 4, 128, 0.25, token_stride=4)
        self.classifier = nn.Linear(64, 4)

    def forward_chunk(self, x, state=None):
        """Return logits, end-exclusive sample counts, and state for the next chunk."""
        if x.ndim != 3 or x.shape[1] != 28 or not x.shape[-1] or x.shape[-1] % 125:
            raise ValueError("Expected B x 28 x samples with a positive multiple of 125 samples")
        old = {} if state is None else state
        count = old.get("sample_count", 0)
        steps = torch.arange(125, x.shape[-1] + 1, 125, device=x.device)
        steps = steps[count + steps >= 1000]
        ends = count + steps
        new = {"sample_count": count + x.shape[-1]}
        features, new["cnn_raw"] = self.stem(x, old.get("cnn_raw"))
        features, new["retention"] = self.retention(features, old.get("retention"), count // 25)
        history = old.get("tokens", features[:, :0])
        combined = torch.cat([history, features], dim=1)
        new["tokens"] = combined[:, -39:]
        if not len(steps):
            return x.new_empty(x.shape[0], 0, 4), ends, new
        starts = history.shape[1] + steps//25 - 40
        windows = combined.unfold(1, 40, 1)[:, starts].permute(0, 1, 3, 2)
        selected = self.transformer(windows)
        return self.classifier(selected), ends, new
