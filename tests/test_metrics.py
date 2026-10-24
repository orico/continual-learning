"""Forgetting and average accuracy from a known matrix."""

import unittest

from cl_toy.metrics import average_forgetting, final_average_accuracy


class MetricsTest(unittest.TestCase):
    def test_average_accuracy_and_forgetting(self) -> None:
        matrix = [
            [1.0],
            [0.4, 1.0],
            [0.2, 0.5, 1.0],
        ]
        self.assertAlmostEqual(final_average_accuracy(matrix), (0.2 + 0.5 + 1.0) / 3)
        self.assertAlmostEqual(average_forgetting(matrix), (0.8 + 0.5) / 2)

    def test_one_task_has_no_forgetting(self) -> None:
        self.assertEqual(average_forgetting([[0.91]]), 0.0)


if __name__ == "__main__":
    unittest.main()
