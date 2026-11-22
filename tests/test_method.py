import unittest

import numpy as np
import torch
from torch.nn import functional as F

from eegdecode import Hybrid
from eegdecode.distillation import mild_view, student_loss, teacher_targets


class MethodTest(unittest.TestCase):
    @torch.no_grad()
    def test_streaming_and_prefix_causality(self):
        torch.manual_seed(42)
        model = Hybrid().eval()
        x = torch.randn(1, 28, 1625)
        full, ends, _ = model.forward_chunk(x)
        outputs, emitted, state = [], [], None
        for chunk in x.split([375, 250, 500, 500], dim=-1):
            logits, indices, state = model.forward_chunk(chunk, state)
            outputs.append(logits)
            emitted.append(indices)
        torch.testing.assert_close(torch.cat(emitted), ends)
        torch.testing.assert_close(torch.cat(outputs, 1), full, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ends, torch.arange(1000, 1626, 125))
        x[..., 1125:] += 100
        changed = model.forward_chunk(x)[0]
        torch.testing.assert_close(changed[:, :2], full[:, :2])

    def test_reliability_and_temperature(self):
        clean = np.zeros((3, 2, 4))
        q, r = teacher_targets(clean, clean)
        np.testing.assert_allclose(q, .25)
        np.testing.assert_allclose(r, 0, atol=1e-7)
        clean[..., 0] = 9.
        q, confident = teacher_targets(clean, clean)
        np.testing.assert_allclose(q[:, 0], np.exp(3) / (np.exp(3) + 3))
        _, disagreement = teacher_targets(clean, np.roll(clean, 1, axis=-1))
        split = clean.copy()
        split[0] = np.roll(split[0], 1, axis=-1)
        _, mixed = teacher_targets(split, split)
        self.assertTrue(np.all((disagreement < confident) & (mixed < confident)))

    def test_class_weighted_kd(self):
        torch.manual_seed(17)
        logits = torch.randn(3, 4, requires_grad=True)
        labels = torch.tensor([0, 2, 2])
        weights = torch.tensor([1., 2., 3., 4.])
        q = torch.full_like(logits, .25, requires_grad=True)
        r = torch.tensor([0., .5, 1.], requires_grad=True)
        ce = F.cross_entropy(logits, labels, reduction="none")
        kl = (q * (q.log() - (logits / 3).log_softmax(-1))).sum(-1)
        expected = ((ce * (1 - .5*r) + 4.5*r*kl) * weights[labels]).sum() / weights[labels].sum()
        loss = student_loss(logits, labels, weights, q, r)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertIsNone(q.grad)
        self.assertIsNone(r.grad)

    def test_fixed_view_preserves_prefix(self):
        x = np.ones((28, 2000), dtype=np.float32)
        view = mild_view(x, "trial-1")
        np.testing.assert_array_equal(view[:, :1000], mild_view(x[:, :1000], "trial-1"))
        self.assertFalse(np.array_equal(x, view))
