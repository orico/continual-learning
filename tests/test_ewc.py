"""EWC penalty is zero on the snapshot and positive after an important weight moves."""

import unittest

import torch
from torch import nn

from cl_toy.methods import estimate_fisher, ewc_penalty, snapshot_params, stabilize_fisher
from cl_toy.model import MLP


class EwcTest(unittest.TestCase):
    def test_penalty_is_zero_at_the_snapshot(self) -> None:
        layer = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            layer.weight.fill_(0.5)
        star = snapshot_params(layer)
        fisher = {name: torch.ones_like(parameter) for name, parameter in layer.named_parameters()}
        penalty = ewc_penalty(layer, [(star, fisher, 10.0)])
        self.assertAlmostEqual(float(penalty.detach()), 0.0, places=6)

        with torch.no_grad():
            layer.weight.fill_(2.5)
        penalty = ewc_penalty(layer, [(star, fisher, 10.0)])
        self.assertAlmostEqual(float(penalty.detach()), 20.0, places=4)

    def test_fisher_from_examples_is_positive(self) -> None:
        model = MLP(n_outputs=2, n_heads=1, hidden=8, in_dim=4)
        images = torch.randn(10, 4)
        labels = torch.randint(0, 2, (10,))
        task_ids = torch.zeros(10, dtype=torch.long)
        fisher = estimate_fisher(model, images, labels, task_ids, "domain", "ewc", max_samples=5, seed=0)
        total = sum(float(value.sum()) for value in fisher.values())
        self.assertGreater(total, 0.0)

    def test_stable_fisher_keeps_one_sgd_step_from_amplifying_drift(self) -> None:
        fisher = {"weight": torch.tensor([1000.0, 0.001])}
        stable = stabilize_fisher(fisher, lam=1_000_000.0, lr=0.01, max_anchors=4)
        self.assertLessEqual(float(stable["weight"].max()) * 1_000_000.0 * 0.01 * 4, 1.0 + 1e-5)


if __name__ == "__main__":
    unittest.main()
