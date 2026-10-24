"""A tiny two-task problem where naive training forgets more than replay."""

import unittest

import torch

from cl_toy.data import PAIRS, split_by_pairs
from cl_toy.train import run_sequence


def _draw(n_per_class: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = []
    digits = []
    for digit in range(4):
        mean = torch.zeros(16)
        mean[digit] = 5.0
        images.append(mean + 0.4 * torch.randn(n_per_class, 16, generator=generator))
        digits.append(torch.full((n_per_class,), digit, dtype=torch.long))
    return torch.cat(images), torch.cat(digits)


class SyntheticSmokeTest(unittest.TestCase):
    def test_naive_forgets_more_than_replay(self) -> None:
        train_images, train_digits = _draw(120, seed=0)
        test_images, test_digits = _draw(60, seed=1)
        train_tasks = split_by_pairs(train_images, train_digits, "class", PAIRS[:2])
        test_tasks = split_by_pairs(test_images, test_digits, "class", PAIRS[:2])
        shared = dict(
            scenario="class",
            epochs=8,
            lr=0.1,
            momentum=0.9,
            batch_size=64,
            hidden=32,
            in_dim=16,
            class_outputs=4,
            seed=0,
        )
        naive = run_sequence(train_tasks, test_tasks, method="naive", **shared)
        replay = run_sequence(
            train_tasks,
            test_tasks,
            method="replay",
            buffer_capacity=80,
            **shared,
        )
        self.assertGreaterEqual(naive["matrix"][0][0], 0.9)
        naive_kept = naive["matrix"][1][0]
        replay_kept = replay["matrix"][1][0]
        self.assertLessEqual(naive_kept, 0.7)
        self.assertGreaterEqual(replay_kept, naive_kept + 0.15)


if __name__ == "__main__":
    unittest.main()
