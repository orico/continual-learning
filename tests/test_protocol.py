"""Scenario labels and the ban on leaking the task id."""

import unittest

import torch

from cl_toy.data import PAIRS, ProtocolError, map_label, split_by_pairs, uses_true_task_id
from cl_toy.model import MLP, predict_logits
from cl_toy.train import evaluate_task


class Spy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_heads = 1
        self.n_outputs = 2
        self.seen: list = []
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, images, task_ids=None):
        self.seen.append(task_ids)
        return self.linear(images.view(images.size(0), -1))


class ProtocolTest(unittest.TestCase):
    def test_label_mapping(self) -> None:
        self.assertEqual(map_label(2, "task", 1), 0)
        self.assertEqual(map_label(3, "task", 1), 1)
        self.assertEqual(map_label(2, "domain", 1), 0)
        self.assertEqual(map_label(3, "domain", 1), 1)
        self.assertEqual(map_label(8, "class", 4), 8)
        self.assertEqual(map_label(9, "domain", 4), 1)

    def test_task_splits_do_not_contain_future_digits(self) -> None:
        digits = torch.tensor([0, 1, 2, 3, 4, 5])
        images = torch.zeros(6, 1, 28, 28)
        tasks = split_by_pairs(images, digits, "class", PAIRS[:3])
        self.assertEqual(set(tasks[0]["labels"].tolist()), {0, 1})
        self.assertEqual(set(tasks[1]["digits"].tolist()), {2, 3})
        self.assertEqual(set(tasks[2]["labels"].tolist()), {4, 5})

    def test_domain_and_class_methods_do_not_receive_the_task_id(self) -> None:
        blocked = ("naive", "joint", "replay", "ewc", "lwf", "agem", "merged", "experts_routed")
        for method in blocked:
            self.assertFalse(uses_true_task_id("class", method))
            self.assertFalse(uses_true_task_id("domain", method))
        self.assertTrue(uses_true_task_id("task", "naive"))
        self.assertTrue(uses_true_task_id("class", "experts"))

    def test_multi_head_class_incremental_forward_is_rejected(self) -> None:
        model = MLP(n_outputs=2, n_heads=3, hidden=4, in_dim=4)
        images = torch.randn(4, 4)
        task_ids = torch.tensor([0, 1, 0, 1])
        with self.assertRaises(ProtocolError):
            predict_logits(model, images, task_ids, "class", "replay")
        logits = predict_logits(model, images, task_ids, "task", "naive")
        self.assertEqual(tuple(logits.shape), (4, 2))

    def test_class_incremental_evaluation_calls_the_model_without_a_task_id(self) -> None:
        spy = Spy()
        task = {
            "images": torch.randn(4, 4),
            "labels": torch.zeros(4, dtype=torch.long),
            "task_ids": torch.arange(4),
            "digits": torch.zeros(4, dtype=torch.long),
        }
        evaluate_task(spy, task, "class", "naive", batch_size=4)
        self.assertIsNone(spy.seen[-1])


if __name__ == "__main__":
    unittest.main()
