"""Small multilayer perceptron used for every method."""

from __future__ import annotations

import torch
from torch import nn

from cl_toy.data import ProtocolError, uses_true_task_id


class MLP(nn.Module):
    """784-256-256-outputs network with ReLU, matching the toy in the plan.

    Task-incremental training uses one 2-way head per task on a shared trunk.
    Domain- and class-incremental training use a single head. There is no
    dropout and no batch norm, so nothing besides the methods regularizes.
    """

    def __init__(
        self,
        n_outputs: int,
        n_heads: int = 1,
        hidden: int = 256,
        in_dim: int = 784,
    ) -> None:
        super().__init__()
        if n_heads < 1:
            raise ValueError("n_heads must be positive")
        self.n_outputs = n_outputs
        self.n_heads = n_heads
        self.in_dim = in_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        if n_heads == 1:
            self.head = nn.Linear(hidden, n_outputs)
        else:
            self.heads = nn.ModuleList(
                [nn.Linear(hidden, n_outputs) for _ in range(n_heads)]
            )

    def features(self, images: torch.Tensor) -> torch.Tensor:
        flat = images.view(images.size(0), -1)
        hidden = torch.relu(self.fc1(flat))
        return torch.relu(self.fc2(hidden))

    def forward(self, images: torch.Tensor, task_ids: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.features(images)
        if self.n_heads == 1:
            return self.head(hidden)
        if task_ids is None:
            raise ProtocolError("This multi-head model requires a task id.")
        if not torch.is_tensor(task_ids):
            return self.heads[int(task_ids)](hidden)
        flat_ids = task_ids.reshape(-1)
        if torch.unique(flat_ids).numel() == 1:
            return self.heads[int(flat_ids[0])](hidden)
        logits = hidden.new_zeros(hidden.size(0), self.n_outputs)
        for task_id in flat_ids.unique().tolist():
            mask = flat_ids == task_id
            logits[mask] = self.heads[int(task_id)](hidden[mask])
        return logits


def build_model(
    scenario: str,
    n_tasks: int,
    class_outputs: int = 10,
    hidden: int = 256,
    in_dim: int = 784,
    n_heads: int | None = None,
) -> MLP:
    """Build the shared model for a scenario.

    Experts pass n_heads=1 and are constructed one task at a time. The shared
    task-incremental model gets one head per task.
    """
    if scenario == "task":
        heads = n_tasks if n_heads is None else n_heads
        return MLP(n_outputs=2, n_heads=heads, hidden=hidden, in_dim=in_dim)
    if scenario == "domain":
        return MLP(n_outputs=2, n_heads=1, hidden=hidden, in_dim=in_dim)
    if scenario == "class":
        return MLP(n_outputs=class_outputs, n_heads=1, hidden=hidden, in_dim=in_dim)
    raise ValueError(f"Unknown scenario: {scenario}")


def count_parameters(model: nn.Module | list[nn.Module]) -> int:
    if isinstance(model, list):
        return sum(count_parameters(item) for item in model)
    return sum(parameter.numel() for parameter in model.parameters())


def predict_logits(
    model: MLP,
    images: torch.Tensor,
    task_ids: torch.Tensor,
    scenario: str,
    method: str,
) -> torch.Tensor:
    """Forward pass that refuses a task id the scenario does not provide.

    A single-head model never receives a task id. A multi-head model receives
    one only when the protocol allows it (task-incremental, or an oracle expert
    selector that has already chosen a module before this call).
    """
    if model.n_heads == 1:
        return model(images, None)
    if not uses_true_task_id(scenario, method):
        raise ProtocolError(
            f"{method} on the {scenario} setting cannot see the true task id."
        )
    return model(images, task_ids)
