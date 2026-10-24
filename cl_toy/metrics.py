"""Accuracy and forgetting from a sequential evaluation matrix.

matrix[i][j] is accuracy on task j after training through task i.
Row i has length i + 1, covering tasks seen so far.
"""

from __future__ import annotations


def average_accuracy(row: list[float]) -> float:
    if not row:
        raise ValueError("accuracy row is empty")
    return sum(row) / len(row)


def final_average_accuracy(matrix: list[list[float]]) -> float:
    if not matrix:
        raise ValueError("matrix is empty")
    return average_accuracy(matrix[-1])


def average_forgetting(matrix: list[list[float]]) -> float:
    """Mean drop from the accuracy just after a task was learned to the end.

    The last task has no later measurement, so it is left out. A one-task
    matrix therefore has forgetting 0.
    """
    if len(matrix) < 2:
        return 0.0
    final = matrix[-1]
    drops = [matrix[task][task] - final[task] for task in range(len(matrix) - 1)]
    return sum(drops) / len(drops)


def diagonal(matrix: list[list[float]]) -> list[float]:
    return [row[index] for index, row in enumerate(matrix)]
