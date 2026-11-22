"""K0-safe teacher targets, deterministic second view and weighted student loss."""
import hashlib

import numpy as np
import torch
from scipy.signal import lfilter
from scipy.special import softmax, xlogy
from torch.nn import functional as F


def mild_view(x, trial):
    """Fixed channel dropout plus causal AR(1) noise in source-normalized units."""
    key = f"{trial}|1729|0|mild".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
    view = x.copy()
    view[rng.choice(28, 1, replace=False)] = 0
    initial = rng.normal(0, 0.05, 28)
    innovations = rng.normal(0, 0.05 * np.sqrt(1 - 0.8**2), (x.shape[1], 28))
    noise, _ = lfilter([1.], [1., -0.8], innovations, axis=0, zi=(0.8 * initial)[None])
    view += noise.T.astype(np.float32)
    return view


def teacher_targets(clean_logits, view_logits):
    """Arrays are [3 teachers, aligned endpoints, 4 classes]."""
    clean = np.asarray(clean_logits, dtype=np.float64)
    view = np.asarray(view_logits, dtype=np.float64)
    if clean.ndim != 3 or clean.shape[::2] != (3, 4) or clean.shape != view.shape:
        raise ValueError("Expected matching [3, endpoints, 4] teacher logits")
    if not np.isfinite(clean).all() or not np.isfinite(view).all():
        raise ValueError("Non-finite teacher logits")

    def entropy(p):
        return -xlogy(p, p).sum(axis=-1)

    members = softmax(clean, axis=-1)
    mean = members.mean(axis=0)
    perturbed = softmax(view, axis=-1).mean(axis=0)
    h = entropy(mean)
    member_js = np.maximum(0, h - entropy(members).mean(axis=0))
    view_js = np.maximum(0, entropy((mean + perturbed) / 2) - (h + entropy(perturbed)) / 2)
    r = np.clip((1 - h / np.log(4)) * np.exp(-member_js / 0.1) * np.exp(-view_js / 0.1), 0, 1)
    q = softmax(clean / 3, axis=-1).mean(axis=0)
    return q.astype(np.float32), r.astype(np.float32)


def student_loss(logits, labels, weights, q, reliability):
    """Class-balanced interpolation of CE and T² KL(q || student at T)."""
    if q.shape != logits.shape or reliability.shape != labels.shape:
        raise ValueError("Misaligned teacher targets")
    ce = F.cross_entropy(logits, labels, reduction="none")
    kl = F.kl_div(F.log_softmax(logits / 3, dim=-1), q.detach(), reduction="none").sum(-1)
    strength = 0.5 * reliability.detach()
    loss = (1 - strength) * ce + strength * 9 * kl
    w = weights[labels]
    return (w * loss).sum() / w.sum()
