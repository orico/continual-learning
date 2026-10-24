"""Figures for the Split MNIST continual-learning runs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHOD_LABELS = {
    "naive": "Naive",
    "joint": "Joint",
    "replay": "Replay",
    "ewc": "EWC",
    "lwf": "LwF",
    "agem": "A-GEM",
    "experts": "Experts (oracle id)",
    "experts_routed": "Experts (predicted id)",
}

SCENARIO_LABELS = {
    "task": "Task-incremental",
    "domain": "Domain-incremental",
    "class": "Class-incremental",
}

METHOD_ORDER = [
    "naive",
    "replay",
    "ewc",
    "lwf",
    "agem",
    "experts",
    "experts_routed",
    "joint",
]


def _grouped(runs: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["scenario"], run["method"])].append(run)
    return grouped


def _mean_matrix(runs: list[dict]) -> np.ndarray:
    length = len(runs[0]["matrix"])
    total = np.zeros((length, length), dtype=float)
    counts = np.zeros((length, length), dtype=float)
    for run in runs:
        for row_index, row in enumerate(run["matrix"]):
            for col_index, value in enumerate(row):
                total[row_index, col_index] += value
                counts[row_index, col_index] += 1
    counts[counts == 0] = np.nan
    return total / counts


def plot_forgetting_matrices(runs: list[dict], path: Path) -> None:
    """Heatmaps of class-incremental accuracy after each task, averaged over seeds."""
    methods = [method for method in METHOD_ORDER if method != "joint"]
    present = []
    for method in methods:
        matched = [run for run in runs if run["scenario"] == "class" and run["method"] == method]
        if matched:
            present.append((method, matched))
    if not present:
        return
    columns = 4
    rows = int(np.ceil(len(present) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.6 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for index, (method, matched) in enumerate(present):
        axis = axes.ravel()[index]
        axis.axis("on")
        matrix = _mean_matrix(matched)
        masked = np.ma.masked_invalid(matrix)
        image = axis.imshow(masked, vmin=0.0, vmax=1.0, cmap="viridis")
        length = matrix.shape[0]
        axis.set_xticks(range(length))
        axis.set_yticks(range(length))
        axis.set_xlabel("Task")
        axis.set_ylabel("After task")
        axis.set_title(METHOD_LABELS.get(method, method))
        for row in range(length):
            for col in range(row + 1):
                value = matrix[row, col]
                color = "white" if value < 0.55 else "black"
                axis.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Class-incremental forgetting matrices (mean over seeds)", fontsize=14)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_accuracy_curves(runs: list[dict], path: Path) -> None:
    """Average accuracy on tasks seen so far, after each task."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    grouped = _grouped(runs)
    for axis, scenario in zip(axes, ("task", "domain", "class")):
        for method in METHOD_ORDER:
            matched = grouped.get((scenario, method), [])
            if not matched:
                continue
            if method == "joint":
                height = float(np.mean([run["avg_accuracy"] for run in matched]))
                width = max(len(run["matrix"][-1]) for run in matched)
                axis.plot(
                    np.arange(1, width + 1),
                    np.full(width, height * 100),
                    linestyle="--",
                    label=METHOD_LABELS[method],
                )
                continue
            curves = []
            for run in matched:
                curves.append([100 * sum(row) / len(row) for row in run["matrix"]])
            stack = np.vstack(curves)
            mean = stack.mean(axis=0)
            std = stack.std(axis=0)
            xs = np.arange(1, len(mean) + 1)
            axis.plot(xs, mean, label=METHOD_LABELS.get(method, method))
            axis.fill_between(xs, mean - std, mean + std, alpha=0.15)
        axis.set_title(SCENARIO_LABELS[scenario])
        axis.set_xlabel("Tasks learned")
        axis.set_ylabel("Average accuracy so far (%)")
        axis.set_xticks(range(1, 6))
        axis.set_ylim(0, 100)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8, loc="lower left")
    figure.suptitle("Accuracy while tasks arrive (band is the seed spread)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_final_bars(runs: list[dict], path: Path) -> None:
    """Final average accuracy and average forgetting by method and setting."""
    grouped = _grouped(runs)
    scenarios = ["task", "domain", "class"]
    methods = [
        method
        for method in METHOD_ORDER
        if any(grouped.get((scenario, method)) for scenario in scenarios)
    ]
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(len(methods))
    width = 0.25
    for offset, scenario in enumerate(scenarios):
        acc_mean, acc_std, forg_mean, forg_std = [], [], [], []
        for method in methods:
            matched = grouped.get((scenario, method), [])
            if not matched:
                acc_mean.append(np.nan)
                acc_std.append(0)
                forg_mean.append(np.nan)
                forg_std.append(0)
                continue
            accuracies = np.array([run["avg_accuracy"] for run in matched], dtype=float)
            acc_mean.append(accuracies.mean() * 100)
            acc_std.append(accuracies.std() * 100)
            forgetting = [run["avg_forgetting"] for run in matched if run["avg_forgetting"] is not None]
            if forgetting:
                values = np.array(forgetting, dtype=float)
                forg_mean.append(values.mean() * 100)
                forg_std.append(values.std() * 100)
            else:
                forg_mean.append(np.nan)
                forg_std.append(0)
        shift = (offset - 1) * width
        axes[0].bar(x + shift, acc_mean, width, yerr=acc_std, capsize=3, label=SCENARIO_LABELS[scenario])
        axes[1].bar(x + shift, forg_mean, width, yerr=forg_std, capsize=3, label=SCENARIO_LABELS[scenario])
    axes[0].set_ylabel("Final average accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].set_ylabel("Average forgetting (points)")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([METHOD_LABELS.get(method, method) for method in methods], rotation=15)
    figure.suptitle("End of the sequence, with the seed spread as error bars")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_forgetting_speed(speed_runs: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    for run in speed_runs:
        speed = run["speed"]
        axis.plot(
            speed["steps"],
            [100 * value for value in speed["task0_accuracy"]],
            label=METHOD_LABELS.get(run["method"], run["method"]),
        )
    axis.set_xlabel("Training steps on task 2")
    axis.set_ylabel("Accuracy on task 1 (%)")
    axis.set_ylim(0, 100)
    axis.set_title("How fast task 1 is forgotten while learning task 2 (class-incremental)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_lambda_sweep(runs: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, scenario in zip(axes, ("task", "class")):
        matched = [run for run in runs if run["scenario"] == scenario]
        matched.sort(key=lambda run: run["ewc_lambda"])
        if not matched:
            continue
        lambdas = [run["ewc_lambda"] for run in matched]
        first = [100 * run["matrix"][-1][0] for run in matched]
        last = [100 * run["matrix"][-1][-1] for run in matched]
        early = [100 * float(np.mean(run["matrix"][-1][:-1])) for run in matched]
        axis.plot(lambdas, first, marker="o", label="Task 1 at the end")
        axis.plot(lambdas, early, marker="o", label="Mean of earlier tasks")
        axis.plot(lambdas, last, marker="o", label="Last task at the end")
        axis.set_xscale("log")
        axis.set_xlabel("EWC lambda")
        axis.set_ylabel("Accuracy (%)")
        axis.set_ylim(0, 100)
        axis.set_title(SCENARIO_LABELS[scenario])
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Stability against plasticity as the EWC penalty grows")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_buffer_sweep(runs: list[dict], joint_accuracy: float, path: Path) -> None:
    matched = sorted(runs, key=lambda run: run["buffer_capacity"])
    figure, axis = plt.subplots(figsize=(7, 4))
    sizes = [run["buffer_capacity"] for run in matched]
    accuracy = [100 * run["avg_accuracy"] for run in matched]
    forgetting = [
        100 * run["avg_forgetting"] if run["avg_forgetting"] is not None else np.nan for run in matched
    ]
    axis.plot(sizes, accuracy, marker="o", label="Final average accuracy")
    axis.plot(sizes, forgetting, marker="o", label="Average forgetting")
    axis.axhline(100 * joint_accuracy, linestyle="--", color="gray", label="Joint accuracy")
    axis.set_xlabel("Replay buffer size (examples)")
    axis.set_ylabel("Percent")
    axis.set_ylim(0, 100)
    axis.set_title("Class-incremental replay as the memory grows")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
