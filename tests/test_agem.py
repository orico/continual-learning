"""A-GEM removes the part of a step that points against stored past data."""

import unittest

import torch
from torch import nn

from cl_toy.methods import apply_agem, project_gradients


class AgemTest(unittest.TestCase):
    def test_opposing_gradient_is_projected_to_a_nonnegative_dot_product(self) -> None:
        current = torch.tensor([1.0, 1.0])
        reference = torch.tensor([-1.0, 0.0])
        projected = project_gradients(current, reference)
        self.assertTrue(torch.allclose(projected, torch.tensor([0.0, 1.0])))
        self.assertGreaterEqual(float(torch.dot(projected, reference)), -1e-6)

    def test_aligned_gradient_is_unchanged(self) -> None:
        current = torch.tensor([1.0, 0.0])
        reference = torch.tensor([0.5, 0.0])
        projected = project_gradients(current, reference)
        self.assertTrue(torch.allclose(projected, current))

    def test_momentum_cannot_keep_pushing_up_the_memory_loss(self) -> None:
        layer = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            layer.weight.zero_()
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1, momentum=0.9)
        optimizer.state[layer.weight]["momentum_buffer"] = torch.full_like(layer.weight, -5.0)
        apply_agem(layer, optimizer, -layer.weight.sum(), layer.weight.sum())
        velocity = optimizer.state[layer.weight]["momentum_buffer"]
        reference = torch.ones_like(velocity)
        self.assertGreaterEqual(float(torch.dot(velocity.reshape(-1), reference.reshape(-1))), -1e-5)
        self.assertAlmostEqual(float(layer.weight.detach()), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
