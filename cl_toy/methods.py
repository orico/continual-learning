"""Loss terms and update rules for the four continual-learning families."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from cl_toy.model import MLP, predict_logits


@dataclass(frozen=True)
class MethodPieces:
    """Which of the three shared-network protections a method turns on.

    The weight penalty, the frozen copy, and the step check are independent.
    Replay still mixes stored photos into the batch, and that mix is not one of
    these switches. Experts freeze a whole network per lesson and are not these
    switches either.
    """

    ewc: bool = False
    lwf: bool = False
    agem: bool = False


_NO_PIECES = MethodPieces()

METHOD_PIECES: dict[str, MethodPieces] = {
    "naive": _NO_PIECES,
    "joint": _NO_PIECES,
    "replay": _NO_PIECES,
    "ewc": MethodPieces(ewc=True),
    "lwf": MethodPieces(lwf=True),
    "agem": MethodPieces(agem=True),
    "merged": MethodPieces(ewc=True, lwf=True, agem=True),
    "experts": _NO_PIECES,
    "experts_routed": _NO_PIECES,
}


def pieces_for(method: str) -> MethodPieces:
    """Return the switches for a known method. An unknown name raises."""
    try:
        return METHOD_PIECES[method]
    except KeyError:
        raise ValueError(f"Unknown method: {method}") from None


def snapshot_params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def project_gradients(current: torch.Tensor, reference: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """A-GEM projection: a step that would increase loss on stored past data is bent back.

    When the current gradient has a negative dot product with the memory gradient,
    remove that component so the update does not point against the memory gradient.
    """
    dot = torch.dot(current, reference)
    if float(dot) >= 0.0:
        return current
    denom = torch.dot(reference, reference).clamp_min(eps)
    return current - (dot / denom) * reference


def stabilize_fisher(
    fisher: dict[str, torch.Tensor],
    lam: float,
    lr: float,
    max_anchors: int = 4,
) -> dict[str, torch.Tensor]:
    """Rescale the Fisher so SGD can apply the quadratic penalty without diverging.

    The diagonal is divided by its mean, so lambda is a relative strength.
    Each entry is capped so that one SGD step, summed across every past-task
    anchor, cannot amplify a weight's distance from those snapshots. A large
    lambda therefore pins important weights instead of sending them to infinity.
    """
    total = None
    count = 0
    for value in fisher.values():
        total = value.sum() if total is None else total + value.sum()
        count += value.numel()
    mean = (total / count).clamp_min(1e-12)
    cap = 1.0 / max(lr * float(lam) * max(max_anchors, 1), 1e-12)
    return {name: (value / mean).clamp(max=cap) for name, value in fisher.items()}


def ewc_penalty(
    model: nn.Module,
    anchors: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], float]],
) -> torch.Tensor:
    """Sum of quadratic pulls toward each past task's parameters.

    Each anchor is (snapshot, diagonal Fisher, lambda). The penalty is zero
    when the weights still sit on that snapshot.
    """
    total = None
    for star, fisher, lam in anchors:
        penalty = None
        for name, parameter in model.named_parameters():
            term = (fisher[name] * (parameter - star[name]).pow(2)).sum()
            penalty = term if penalty is None else penalty + term
        scaled = (lam / 2.0) * penalty
        total = scaled if total is None else total + scaled
    if total is None:
        parameter = next(model.parameters())
        return parameter.new_zeros(())
    return total


def estimate_fisher(
    model: MLP,
    images: torch.Tensor,
    labels: torch.Tensor,
    task_ids: torch.Tensor,
    scenario: str,
    method: str,
    max_samples: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Diagonal empirical Fisher from per-example gradients.

    Squaring the mini-batch gradient underestimates curvature once a task is
    fit, because example gradients cancel inside the batch. Averaging the
    squared gradient of each example is the estimate EWC uses.
    """
    fisher = {name: torch.zeros_like(parameter) for name, parameter in model.named_parameters()}
    was_training = model.training
    model.eval()
    count = images.shape[0]
    take = min(max_samples, count)
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(count, generator=generator)[:take]
    for index in order.tolist():
        model.zero_grad(set_to_none=True)
        sample = images[index : index + 1]
        label = labels[index : index + 1]
        task_id = task_ids[index : index + 1]
        logits = predict_logits(model, sample, task_id, scenario, method)
        loss = F.cross_entropy(logits, label)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                fisher[name] += parameter.grad.detach().pow(2)
    for name in fisher:
        fisher[name] /= take
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return fisher


def softened_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    target = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(log_probs, target, reduction="batchmean") * (temperature ** 2)


def distillation_loss(
    student: MLP,
    teacher: MLP,
    images: torch.Tensor,
    current_task: int,
    temperature: float,
) -> torch.Tensor:
    """Learning without Forgetting: keep the previous model's outputs stable.

    A single head is distilled on the current batch. A multi-head model distills
    every head from before this task. No stored images from past tasks are used.
    """
    if current_task <= 0:
        return student.fc1.weight.new_zeros(())
    if student.n_heads == 1:
        with torch.no_grad():
            old_logits = teacher(images, None)
        new_logits = student(images, None)
        return softened_kl(new_logits, old_logits, temperature)
    total = student.fc1.weight.new_zeros(())
    for task_id in range(current_task):
        ids = torch.full((images.size(0),), task_id, dtype=torch.long)
        with torch.no_grad():
            old_logits = teacher(images, ids)
        new_logits = student(images, ids)
        total = total + softened_kl(new_logits, old_logits, temperature)
    return total / current_task


def clone_frozen(model: nn.Module) -> nn.Module:
    frozen = copy.deepcopy(model).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen


def apply_agem(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_current: torch.Tensor,
    loss_reference: torch.Tensor,
) -> None:
    """SGD step whose actual update, including momentum, does not point against the memory gradient.

    Projecting only the new gradient is not enough. With momentum the step follows
    the velocity, and an old velocity can keep increasing the loss on stored
    examples after the new gradient has been made safe.
    """
    optimizer.zero_grad(set_to_none=True)
    loss_current.backward()
    current = [
        parameter.grad.detach().clone() if parameter.grad is not None else torch.zeros_like(parameter)
        for parameter in model.parameters()
    ]
    optimizer.zero_grad(set_to_none=True)
    loss_reference.backward()
    reference = [
        parameter.grad.detach().clone() if parameter.grad is not None else torch.zeros_like(parameter)
        for parameter in model.parameters()
    ]
    group = optimizer.param_groups[0]
    learning_rate = group["lr"]
    momentum = group["momentum"]
    with torch.no_grad():
        buffers = []
        pieces = []
        for parameter, grad in zip(model.parameters(), current):
            state = optimizer.state[parameter]
            buffer = state.get("momentum_buffer")
            if buffer is None:
                buffer = state["momentum_buffer"] = torch.zeros_like(parameter)
            buffer.mul_(momentum).add_(grad)
            buffers.append(buffer)
            pieces.append(buffer.reshape(-1))
        velocity = torch.cat(pieces).clone()
        flat_reference = torch.cat([grad.reshape(-1) for grad in reference])
        projected = project_gradients(velocity, flat_reference)
        offset = 0
        for parameter, buffer in zip(model.parameters(), buffers):
            width = parameter.numel()
            update = projected[offset : offset + width].view_as(parameter)
            buffer.copy_(update)
            parameter.add_(update, alpha=-learning_rate)
            offset += width
