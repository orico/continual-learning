"""Sequential training loop shared by every method."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import nn

from cl_toy.buffer import ClassBalancedBuffer
from cl_toy.metrics import average_forgetting, diagonal, final_average_accuracy
from cl_toy.methods import (
    apply_agem,
    clone_frozen,
    distillation_loss,
    estimate_fisher,
    ewc_penalty,
    pieces_for,
    snapshot_params,
    stabilize_fisher,
)
from cl_toy.model import MLP, build_model, count_parameters, predict_logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def iterate_batches(
    images: torch.Tensor,
    labels: torch.Tensor,
    task_ids: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
):
    count = images.shape[0]
    order = torch.randperm(count, generator=generator) if shuffle else torch.arange(count)
    for start in range(0, count, batch_size):
        index = order[start : start + batch_size]
        yield images[index], labels[index], task_ids[index]


def concat_tasks(tasks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "images": torch.cat([task["images"] for task in tasks]),
        "labels": torch.cat([task["labels"] for task in tasks]),
        "task_ids": torch.cat([task["task_ids"] for task in tasks]),
        "digits": torch.cat([task["digits"] for task in tasks]),
    }


@torch.no_grad()
def accuracy_of_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    predictions = logits.argmax(dim=-1)
    correct = int((predictions == labels).sum().item())
    return correct, int(labels.numel())


@torch.no_grad()
def evaluate_task(
    model: MLP | None,
    task: dict[str, torch.Tensor],
    scenario: str,
    method: str,
    experts: list[MLP] | None = None,
    router: MLP | None = None,
    task_index: int = 0,
    batch_size: int = 512,
) -> float:
    """Accuracy on one task's test set under the protocol for this method."""
    correct = 0
    total = 0
    generator = make_generator(0)
    batches = iterate_batches(
        task["images"],
        task["labels"],
        task["task_ids"],
        batch_size,
        shuffle=False,
        generator=generator,
    )
    for images, labels, task_ids in batches:
        if method == "experts":
            module = experts[task_index]
            module.eval()
            logits = module(images, None)
        elif method == "experts_routed":
            assert experts is not None and router is not None
            router.eval()
            available = router(images, None)[:, : len(experts)]
            choice = available.argmax(dim=-1)
            logits = images.new_zeros(images.size(0), experts[0].n_outputs)
            for chosen in choice.unique().tolist():
                mask = choice == chosen
                expert = experts[int(chosen)]
                expert.eval()
                logits[mask] = expert(images[mask], None)
        else:
            assert model is not None
            model.eval()
            logits = predict_logits(model, images, task_ids, scenario, method)
        got, size = accuracy_of_logits(logits, labels)
        correct += got
        total += size
    if total == 0:
        raise RuntimeError("evaluation batch was empty")
    return correct / total


def evaluate_seen(
    model: MLP | None,
    test_tasks: list[dict[str, torch.Tensor]],
    up_to: int,
    scenario: str,
    method: str,
    experts: list[MLP] | None = None,
    router: MLP | None = None,
    batch_size: int = 512,
) -> list[float]:
    return [
        evaluate_task(
            model,
            test_tasks[task_index],
            scenario,
            method,
            experts=experts,
            router=router,
            task_index=task_index,
            batch_size=batch_size,
        )
        for task_index in range(up_to + 1)
    ]


def _new_optimizer(model: nn.Module, lr: float, momentum: float) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0)


def _train_joint(
    model: MLP,
    train_tasks: list[dict[str, torch.Tensor]],
    scenario: str,
    method: str,
    epochs: int,
    lr: float,
    momentum: float,
    batch_size: int,
    seed: int,
) -> None:
    merged = concat_tasks(train_tasks)
    optimizer = _new_optimizer(model, lr, momentum)
    generator = make_generator(seed + 17)
    model.train()
    for _ in range(epochs):
        for images, labels, task_ids in iterate_batches(
            merged["images"],
            merged["labels"],
            merged["task_ids"],
            batch_size,
            shuffle=True,
            generator=generator,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = predict_logits(model, images, task_ids, scenario, method)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()


def run_sequence(
    train_tasks: list[dict[str, torch.Tensor]],
    test_tasks: list[dict[str, torch.Tensor]],
    scenario: str,
    method: str,
    seed: int = 0,
    epochs: int = 2,
    lr: float = 0.01,
    momentum: float = 0.9,
    batch_size: int = 128,
    buffer_capacity: int = 500,
    ewc_lambda: float = 1000.0,
    lwf_alpha: float = 1.0,
    lwf_temperature: float = 2.0,
    hidden: int = 256,
    in_dim: int = 784,
    class_outputs: int = 10,
    fisher_samples: int = 256,
    watch_task0_on_task1: bool = False,
) -> dict:
    """Train one method on a sequence and return the forgetting matrix.

    `method` is one of naive, joint, replay, ewc, lwf, agem, merged, experts,
    experts_routed. `merged` turns on the weight penalty, the frozen copy, and
    the step check together. Experts keep a fresh frozen network per task. The
    routed variant picks that network with a task classifier trained
    sequentially, which is the setting where parameter isolation loses the task id.
    """
    pieces = pieces_for(method)
    set_seed(seed)
    n_tasks = len(train_tasks)
    if len(test_tasks) != n_tasks:
        raise ValueError("train and test sequences differ in length")

    model: MLP | None = None
    experts: list[MLP] = []
    router: MLP | None = None
    router_optimizer: torch.optim.Optimizer | None = None
    optimizer: torch.optim.Optimizer | None = None
    buffer: ClassBalancedBuffer | None = None
    anchors: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], float]] = []
    teacher: MLP | None = None
    speed: dict[str, list] = {"steps": [], "task0_accuracy": []}

    if method == "replay" or pieces.agem:
        buffer = ClassBalancedBuffer(buffer_capacity, seed=seed)

    if method not in ("experts", "experts_routed"):
        model = build_model(
            scenario,
            n_tasks=n_tasks,
            class_outputs=class_outputs,
            hidden=hidden,
            in_dim=in_dim,
        )
        optimizer = _new_optimizer(model, lr, momentum)

    if method == "experts_routed":
        set_seed(seed + 500)
        router = build_model(
            "class",
            n_tasks=n_tasks,
            class_outputs=n_tasks,
            hidden=hidden,
            in_dim=in_dim,
        )
        router_optimizer = _new_optimizer(router, lr, momentum)

    if method == "joint":
        assert model is not None and optimizer is not None
        _train_joint(
            model,
            train_tasks,
            scenario,
            method,
            epochs,
            lr,
            momentum,
            batch_size,
            seed,
        )
        row = evaluate_seen(model, test_tasks, n_tasks - 1, scenario, method, batch_size=batch_size)
        return _pack_result(
            scenario,
            method,
            seed,
            [row],
            count_parameters(model),
            forgetting=None,
            speed=None,
        )

    matrix: list[list[float]] = []
    for task_index, task in enumerate(train_tasks):
        if method in ("experts", "experts_routed"):
            set_seed(seed + 100 * task_index)
            model = build_model(
                scenario,
                n_tasks=1,
                class_outputs=class_outputs,
                hidden=hidden,
                in_dim=in_dim,
                n_heads=1,
            )
            optimizer = _new_optimizer(model, lr, momentum)

        assert model is not None and optimizer is not None
        model.train()
        generator = make_generator(seed + 1009 * task_index + 3)

        def watch(step: int, current_index: int = task_index) -> None:
            if not watch_task0_on_task1 or current_index != 1:
                return
            accuracy = evaluate_task(
                model if method not in ("experts", "experts_routed") else None,
                test_tasks[0],
                scenario,
                method,
                experts=experts if method == "experts" else ([*experts, model] if method == "experts_routed" else None),
                router=router,
                task_index=0,
                batch_size=batch_size,
            )
            speed["steps"].append(step)
            speed["task0_accuracy"].append(accuracy)

        if watch_task0_on_task1 and task_index == 1:
            watch(0)

        for _ in range(epochs):
            for images, labels, task_ids in iterate_batches(
                task["images"],
                task["labels"],
                task["task_ids"],
                batch_size,
                shuffle=True,
                generator=generator,
            ):
                _update(
                    model,
                    optimizer,
                    images,
                    labels,
                    task_ids,
                    scenario,
                    method,
                    buffer,
                    anchors,
                    teacher,
                    task_index,
                    lwf_alpha,
                    lwf_temperature,
                )
                if watch_task0_on_task1 and task_index == 1:
                    watch(speed["steps"][-1] + 1 if speed["steps"] else 1)

        if method in ("experts", "experts_routed"):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.eval()
            experts.append(model)

        if method == "experts_routed":
            assert router is not None and router_optimizer is not None
            _train_router(router, router_optimizer, task, task_index, epochs, batch_size, seed)

        active_experts = experts if method in ("experts", "experts_routed") else None
        row = evaluate_seen(
            None if method in ("experts", "experts_routed") else model,
            test_tasks,
            task_index,
            scenario,
            method,
            experts=active_experts,
            router=router,
            batch_size=batch_size,
        )
        matrix.append(row)

        if method == "replay" or pieces.agem:
            assert buffer is not None
            buffer.add_many(task["images"], task["labels"], task["task_ids"], task["digits"])

        if pieces.ewc:
            fisher = estimate_fisher(
                model,
                task["images"],
                task["labels"],
                task["task_ids"],
                scenario,
                method,
                fisher_samples,
                seed + task_index,
            )
            if seed == 0 and task_index == 0:
                fisher_mean = torch.stack([value.mean() for value in fisher.values()]).mean()
                print(f"  EWC Fisher mean after task 1: {float(fisher_mean):.6e}", flush=True)
            anchors.append(
                (
                    snapshot_params(model),
                    stabilize_fisher(fisher, ewc_lambda, lr, max_anchors=max(1, n_tasks - 1)),
                    ewc_lambda,
                )
            )

        if pieces.lwf:
            teacher = clone_frozen(model)

    if method == "experts":
        params = count_parameters(experts)
    elif method == "experts_routed":
        params = count_parameters(experts) + count_parameters(router)
    else:
        params = count_parameters(model)

    return _pack_result(
        scenario,
        method,
        seed,
        matrix,
        params,
        forgetting=average_forgetting(matrix),
        speed=speed if watch_task0_on_task1 else None,
    )


def current_loss(
    model: MLP,
    images: torch.Tensor,
    labels: torch.Tensor,
    task_ids: torch.Tensor,
    scenario: str,
    method: str,
    anchors: list,
    teacher: MLP | None,
    task_index: int,
    lwf_alpha: float,
    lwf_temperature: float,
) -> torch.Tensor:
    """Cross-entropy on this batch, plus the protections this method turns on.

    The weight penalty is added when that switch is on and a snapshot exists.
    The frozen-copy pull is added when that switch is on and a teacher exists.
    """
    pieces = pieces_for(method)
    logits = predict_logits(model, images, task_ids, scenario, method)
    loss = F.cross_entropy(logits, labels)
    if pieces.ewc and anchors:
        loss = loss + ewc_penalty(model, anchors)
    if pieces.lwf and teacher is not None:
        loss = loss + lwf_alpha * distillation_loss(
            model, teacher, images, task_index, lwf_temperature
        )
    return loss


def memory_loss(
    model: MLP,
    images: torch.Tensor,
    labels: torch.Tensor,
    task_ids: torch.Tensor,
    scenario: str,
    method: str,
    anchors: list,
    teacher: MLP | None,
    task_index: int,
    lwf_alpha: float,
    lwf_temperature: float,
) -> torch.Tensor:
    """Cross-entropy on stored photos. The penalties are not part of this loss."""
    del anchors, teacher, task_index, lwf_alpha, lwf_temperature
    logits = predict_logits(model, images, task_ids, scenario, method)
    return F.cross_entropy(logits, labels)


def _reject_nonfinite(loss: torch.Tensor, method: str, task_index: int) -> None:
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss in {method} on task {task_index}")


def _update(
    model: MLP,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    task_ids: torch.Tensor,
    scenario: str,
    method: str,
    buffer: ClassBalancedBuffer | None,
    anchors: list,
    teacher: MLP | None,
    task_index: int,
    lwf_alpha: float,
    lwf_temperature: float,
) -> None:
    model.train()
    pieces = pieces_for(method)
    if pieces.agem and buffer is not None and len(buffer) > 0:
        loss_current = current_loss(
            model,
            images,
            labels,
            task_ids,
            scenario,
            method,
            anchors,
            teacher,
            task_index,
            lwf_alpha,
            lwf_temperature,
        )
        drawn = buffer.sample(images.size(0))
        assert drawn is not None
        ref_images, ref_labels, ref_task_ids = drawn
        loss_reference = memory_loss(
            model,
            ref_images,
            ref_labels,
            ref_task_ids,
            scenario,
            method,
            anchors,
            teacher,
            task_index,
            lwf_alpha,
            lwf_temperature,
        )
        _reject_nonfinite(loss_current, method, task_index)
        apply_agem(model, optimizer, loss_current, loss_reference)
        return

    if method == "replay" and buffer is not None and len(buffer) > 0:
        drawn = buffer.sample(images.size(0))
        if drawn is not None:
            extra_images, extra_labels, extra_task_ids = drawn
            images = torch.cat([images, extra_images], dim=0)
            labels = torch.cat([labels, extra_labels], dim=0)
            task_ids = torch.cat([task_ids, extra_task_ids], dim=0)

    optimizer.zero_grad(set_to_none=True)
    loss = current_loss(
        model,
        images,
        labels,
        task_ids,
        scenario,
        method,
        anchors,
        teacher,
        task_index,
        lwf_alpha,
        lwf_temperature,
    )
    _reject_nonfinite(loss, method, task_index)
    loss.backward()
    optimizer.step()


def _train_router(
    router: MLP,
    optimizer: torch.optim.Optimizer,
    task: dict[str, torch.Tensor],
    task_index: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> None:
    """Naive sequential training of a task-id classifier. It forgets old tasks."""
    router.train()
    targets = torch.full((task["images"].shape[0],), task_index, dtype=torch.long)
    generator = make_generator(seed + 4000 + task_index)
    for _ in range(epochs):
        for images, labels, _task_ids in iterate_batches(
            task["images"],
            targets,
            task["task_ids"],
            batch_size,
            shuffle=True,
            generator=generator,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = router(images, None)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()


def _pack_result(
    scenario: str,
    method: str,
    seed: int,
    matrix: list[list[float]],
    params: int,
    forgetting: float | None,
    speed: dict | None,
) -> dict:
    if len(matrix) == 1 and len(matrix[0]) > 1:
        diag = list(matrix[0])
    else:
        diag = diagonal(matrix)
    result = {
        "scenario": scenario,
        "method": method,
        "seed": seed,
        "matrix": matrix,
        "avg_accuracy": final_average_accuracy(matrix),
        "avg_forgetting": forgetting,
        "diagonal": diag,
        "params": params,
    }
    if speed is not None:
        result["speed"] = speed
    return result
