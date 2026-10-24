"""Split MNIST and the three continual-learning labelings."""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision import datasets

# Five tasks, each a disjoint pair of digits. This is the Split MNIST sequence
# used by van de Ven and Tolias for task-, domain-, and class-incremental learning.
PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
)

SCENARIOS = ("task", "domain", "class")

# Oracle experts are allowed to condition on the true task id in every setting.
# Every other method may see it only in the task-incremental protocol, where the
# setting itself provides the task id at test time.
ORACLE_METHODS = frozenset({"experts"})


class ProtocolError(RuntimeError):
    """Raised when a method is handed a task id the protocol does not allow."""


def map_label(digit: int, scenario: str, task_id: int) -> int:
    """Map a raw digit to the label the model is trained to predict.

    Task-incremental: which member of the pair (0 or 1), given the task.
    Domain-incremental: even (0) versus odd (1), shared across pairs.
    Class-incremental: the digit itself, among all ten.
    """
    if scenario == "task":
        return PAIRS[task_id].index(digit)
    if scenario == "domain":
        return digit % 2
    if scenario == "class":
        return digit
    raise ValueError(f"Unknown scenario: {scenario}")


def uses_true_task_id(scenario: str, method: str) -> bool:
    """Whether this run may condition on the ground-truth task id."""
    if method in ORACLE_METHODS:
        return True
    return scenario == "task"


def build_task(
    images: torch.Tensor,
    digits: torch.Tensor,
    scenario: str,
    task_id: int,
    pair: tuple[int, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Pack one task's tensors. Digits must belong to that task's pair."""
    expected = PAIRS[task_id] if pair is None else pair
    digit_list = [int(d) for d in digits.tolist()]
    if any(d not in expected for d in digit_list):
        raise ValueError(f"Task {task_id} received digits outside {expected}")
    labels = torch.tensor(
        [map_label(d, scenario, task_id) for d in digit_list],
        dtype=torch.long,
    )
    task_ids = torch.full((len(digit_list),), task_id, dtype=torch.long)
    return {
        "images": images.float(),
        "digits": digits.long(),
        "labels": labels,
        "task_ids": task_ids,
    }


def split_by_pairs(
    images: torch.Tensor,
    digits: torch.Tensor,
    scenario: str,
    pairs: tuple[tuple[int, int], ...] = PAIRS,
) -> list[dict[str, torch.Tensor]]:
    """Cut a digit tensor into sequential tasks."""
    tasks = []
    for task_id, pair in enumerate(pairs):
        mask = (digits == pair[0]) | (digits == pair[1])
        tasks.append(build_task(images[mask], digits[mask], scenario, task_id, pair))
    return tasks


def load_mnist(root: str | Path = "data") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load MNIST as float images in [N, 1, 28, 28] and integer digits.

    Pixels are normalized with the standard MNIST mean and standard deviation
    so the same learning rate can fit a single task in a couple of epochs.
    """
    root = Path(root)
    train = datasets.MNIST(root=str(root), train=True, download=True)
    test = datasets.MNIST(root=str(root), train=False, download=True)

    def convert(ds: datasets.MNIST) -> tuple[torch.Tensor, torch.Tensor]:
        images = ds.data.unsqueeze(1).float().div(255.0)
        images = images.sub(0.1307).div(0.3081)
        return images, ds.targets.long()

    train_x, train_y = convert(train)
    test_x, test_y = convert(test)
    return train_x, train_y, test_x, test_y
