"""Capacity and digit balance of the replay buffer."""

import unittest

import torch

from cl_toy.buffer import ClassBalancedBuffer


class BufferTest(unittest.TestCase):
    def test_length_never_exceeds_capacity(self) -> None:
        buffer = ClassBalancedBuffer(10, seed=0)
        image = torch.zeros(3)
        for _ in range(20):
            for digit in range(8):
                buffer.add(image, label=digit, task_id=0, digit=digit)
                self.assertLessEqual(len(buffer), 10)
        self.assertLessEqual(len(buffer), 10)
        self.assertGreaterEqual(len(buffer.digit_counts()), 8)

    def test_digits_that_share_a_scenario_label_are_both_kept(self) -> None:
        buffer = ClassBalancedBuffer(2, seed=0)
        for _ in range(30):
            buffer.add(torch.zeros(2), label=0, task_id=0, digit=0)
            buffer.add(torch.ones(2), label=0, task_id=1, digit=2)
        counts = buffer.digit_counts()
        self.assertEqual(counts[0], 1)
        self.assertEqual(counts[2], 1)
        self.assertEqual(len(buffer), 2)

    def test_sample_is_capped_by_what_is_stored(self) -> None:
        buffer = ClassBalancedBuffer(5, seed=1)
        self.assertIsNone(buffer.sample(2))
        for digit in range(3):
            buffer.add(torch.zeros(2), label=digit, task_id=0, digit=digit)
        images, labels, task_ids = buffer.sample(10)
        self.assertEqual(images.shape[0], 3)
        self.assertEqual(labels.shape[0], 3)
        self.assertEqual(task_ids.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
