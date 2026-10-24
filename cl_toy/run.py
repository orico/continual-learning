"""Run the Split MNIST continual-learning comparison and write the figures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cl_toy.config import (
    BATCH_SIZE,
    BUFFER_CAPACITY,
    BUFFER_SWEEP,
    CLASS_OUTPUTS,
    EPOCHS,
    EWC_LAMBDA,
    EWC_LAMBDAS,
    FISHER_SAMPLES,
    HIDDEN,
    LR,
    LWF_ALPHA,
    LWF_TEMPERATURE,
    MOMENTUM,
    SEEDS,
)
from cl_toy.data import PAIRS, SCENARIOS, load_mnist, split_by_pairs
from cl_toy.methods import pieces_for
from cl_toy.plots import (
    plot_accuracy_curves,
    plot_buffer_sweep,
    plot_final_bars,
    plot_forgetting_matrices,
    plot_forgetting_speed,
    plot_lambda_sweep,
)
from cl_toy.report import write_summary
from cl_toy.train import run_sequence

FORMAT = 2
BASE_METHODS = ("naive", "joint", "replay", "ewc", "lwf", "agem", "merged", "experts")
SPEED_METHODS = ("naive", "replay", "ewc", "agem", "merged")

COMMON = dict(
    lr=LR,
    momentum=MOMENTUM,
    batch_size=BATCH_SIZE,
    hidden=HIDDEN,
    class_outputs=CLASS_OUTPUTS,
    fisher_samples=FISHER_SAMPLES,
    lwf_alpha=LWF_ALPHA,
    lwf_temperature=LWF_TEMPERATURE,
    buffer_capacity=BUFFER_CAPACITY,
    ewc_lambda=EWC_LAMBDA,
)


def _load_partial(path: Path) -> dict:
    if not path.exists():
        return {"format": FORMAT, "runs": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT:
        print("Ignoring partial results from an older run format.", flush=True)
        return {"format": FORMAT, "runs": []}
    return payload


def _save_partial(path: Path, runs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": FORMAT, "runs": runs}, indent=2), encoding="utf-8")


def _run_sequence_with_overrides(train_tasks, test_tasks, **kwargs):
    """Apply caller overrides on top of the shared hyperparameters."""
    options = dict(COMMON)
    options.update(kwargs)
    return run_sequence(train_tasks, test_tasks, **options)


def _note_divergence(runs: list[dict], partial_path: Path, tag: str, kwargs: dict, exc: RuntimeError) -> None:
    """Remember a run whose loss left the finite range, and keep the other runs going."""
    if "non-finite loss" not in str(exc):
        raise exc
    method = kwargs["method"]
    pieces = pieces_for(method)
    epochs = kwargs.get("epochs", EPOCHS)
    ewc_lambda = kwargs["ewc_lambda"] if "ewc_lambda" in kwargs else (EWC_LAMBDA if pieces.ewc else None)
    buffer_capacity = (
        kwargs["buffer_capacity"]
        if "buffer_capacity" in kwargs
        else (BUFFER_CAPACITY if pieces.agem or method == "replay" else None)
    )
    note = {
        "tag": tag,
        "scenario": kwargs["scenario"],
        "method": method,
        "seed": kwargs.get("seed", 0),
        "epochs": epochs,
        "ewc_lambda": ewc_lambda,
        "buffer_capacity": buffer_capacity,
        "diverged": True,
    }
    print(
        f"diverged {tag} {note['scenario']} {method} seed {note['seed']}: {exc}",
        flush=True,
    )
    runs.append(note)
    _save_partial(partial_path, runs)


def execute(runs: list[dict], partial_path: Path, train_tasks, test_tasks, tag: str, **kwargs) -> dict:
    epochs = kwargs.get("epochs", EPOCHS)
    method = kwargs["method"]
    pieces = pieces_for(method)
    ewc_lambda = kwargs["ewc_lambda"] if "ewc_lambda" in kwargs else (EWC_LAMBDA if pieces.ewc else None)
    buffer_capacity = (
        kwargs["buffer_capacity"]
        if "buffer_capacity" in kwargs
        else (BUFFER_CAPACITY if pieces.agem or method == "replay" else None)
    )
    probe = {
        "tag": tag,
        "scenario": kwargs["scenario"],
        "method": method,
        "seed": kwargs.get("seed", 0),
        "epochs": epochs,
        "ewc_lambda": ewc_lambda,
        "buffer_capacity": buffer_capacity,
    }
    existing = None
    for run in runs:
        if (
            run.get("tag") == probe["tag"]
            and run["scenario"] == probe["scenario"]
            and run["method"] == probe["method"]
            and run["seed"] == probe["seed"]
            and run.get("epochs") == probe["epochs"]
            and run.get("ewc_lambda") == probe["ewc_lambda"]
            and run.get("buffer_capacity") == probe["buffer_capacity"]
        ):
            existing = run
            break
    if existing is not None:
        print(f"skip {tag} {probe['scenario']} {method} seed {probe['seed']}", flush=True)
        return existing

    started = time.time()
    result = _run_sequence_with_overrides(train_tasks, test_tasks, **kwargs)
    result["tag"] = tag
    result["epochs"] = epochs
    result["ewc_lambda"] = ewc_lambda
    result["buffer_capacity"] = buffer_capacity
    forgetting = result["avg_forgetting"]
    forgetting_text = "n/a" if forgetting is None else f"{100 * forgetting:.1f}"
    print(
        f"{tag} {result['scenario']} {result['method']} seed {result['seed']}: "
        f"acc {100 * result['avg_accuracy']:.1f}%  forgetting {forgetting_text}  "
        f"({time.time() - started:.0f}s)",
        flush=True,
    )
    runs.append(result)
    _save_partial(partial_path, runs)
    return result


def smoke(train_x, train_y, test_x, test_y, epochs: int) -> None:
    """Two-task check that a task is learned, then naive training forgets it, and replay forgets less."""
    print("Smoke check on the first two digit pairs.", flush=True)
    pairs = PAIRS[:2]
    naive_task = _run_sequence_with_overrides(
        split_by_pairs(train_x, train_y, "task", pairs),
        split_by_pairs(test_x, test_y, "task", pairs),
        scenario="task",
        method="naive",
        seed=0,
        epochs=epochs,
    )
    naive_class = _run_sequence_with_overrides(
        split_by_pairs(train_x, train_y, "class", pairs),
        split_by_pairs(test_x, test_y, "class", pairs),
        scenario="class",
        method="naive",
        seed=0,
        epochs=epochs,
    )
    replay_class = _run_sequence_with_overrides(
        split_by_pairs(train_x, train_y, "class", pairs),
        split_by_pairs(test_x, test_y, "class", pairs),
        scenario="class",
        method="replay",
        seed=0,
        epochs=epochs,
    )
    learned = min(naive_task["diagonal"])
    class_learned = naive_class["matrix"][0][0]
    drop = naive_class["matrix"][0][0] - naive_class["matrix"][1][0]
    gap = replay_class["matrix"][1][0] - naive_class["matrix"][1][0]
    print(
        f"smoke task-IL just-learned {100 * learned:.1f}%  "
        f"class-IL task 1 {100 * naive_class['matrix'][0][0]:.1f}% -> {100 * naive_class['matrix'][1][0]:.1f}%  "
        f"replay keeps {100 * replay_class['matrix'][1][0]:.1f}%",
        flush=True,
    )
    if learned < 0.90:
        raise SystemExit(f"Smoke failed: task-incremental accuracy just after training is {learned:.3f}, below 0.90.")
    if class_learned < 0.85:
        raise SystemExit(f"Smoke failed: class-incremental task 1 was not learned ({class_learned:.3f}).")
    if drop < 0.25:
        raise SystemExit(f"Smoke failed: class-incremental accuracy on task 1 dropped only {drop:.3f}.")
    if gap < 0.10:
        raise SystemExit(f"Smoke failed: replay kept only {gap:.3f} more task-1 accuracy than naive training.")
    print("Smoke check passed.", flush=True)


def _methods(scenario: str) -> tuple[str, ...]:
    if scenario == "task":
        return BASE_METHODS
    return BASE_METHODS + ("experts_routed",)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split MNIST continual-learning toy")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    print("Loading MNIST.", flush=True)
    train_x, train_y, test_x, test_y = load_mnist(args.data)
    print(f"Train images {tuple(train_x.shape)}, test images {tuple(test_x.shape)}.", flush=True)

    if not args.skip_smoke:
        smoke(train_x, train_y, test_x, test_y, args.epochs)

    partial_path = args.results / "partial.json"
    runs = _load_partial(partial_path)["runs"]
    prepared = {
        scenario: (
            split_by_pairs(train_x, train_y, scenario),
            split_by_pairs(test_x, test_y, scenario),
        )
        for scenario in SCENARIOS
    }

    for seed in args.seeds:
        for scenario in SCENARIOS:
            train_tasks, test_tasks = prepared[scenario]
            for method in _methods(scenario):
                try:
                    execute(
                        runs,
                        partial_path,
                        train_tasks,
                        test_tasks,
                        "main",
                        scenario=scenario,
                        method=method,
                        seed=seed,
                        epochs=args.epochs,
                    )
                except RuntimeError as exc:
                    _note_divergence(
                        runs,
                        partial_path,
                        "main",
                        {
                            "scenario": scenario,
                            "method": method,
                            "seed": seed,
                            "epochs": args.epochs,
                        },
                        exc,
                    )

    speed_pairs = PAIRS[:2]
    speed_train = split_by_pairs(train_x, train_y, "class", speed_pairs)
    speed_test = split_by_pairs(test_x, test_y, "class", speed_pairs)
    for method in SPEED_METHODS:
        try:
            execute(
                runs,
                partial_path,
                speed_train,
                speed_test,
                "speed",
                scenario="class",
                method=method,
                seed=0,
                epochs=args.epochs,
                watch_task0_on_task1=True,
            )
        except RuntimeError as exc:
            _note_divergence(
                runs,
                partial_path,
                "speed",
                {
                    "scenario": "class",
                    "method": method,
                    "seed": 0,
                    "epochs": args.epochs,
                },
                exc,
            )

    train_tasks, test_tasks = prepared["task"]
    class_train, class_test = prepared["class"]
    for scenario, sequence in (("task", (train_tasks, test_tasks)), ("class", (class_train, class_test))):
        for lam in EWC_LAMBDAS:
            execute(
                runs,
                partial_path,
                sequence[0],
                sequence[1],
                "lambda",
                scenario=scenario,
                method="ewc",
                seed=0,
                epochs=args.epochs,
                ewc_lambda=lam,
            )

    for capacity in BUFFER_SWEEP:
        execute(
            runs,
            partial_path,
            class_train,
            class_test,
            "buffer",
            scenario="class",
            method="replay",
            seed=0,
            epochs=args.epochs,
            buffer_capacity=capacity,
        )

    for method in ("naive", "replay"):
        execute(
            runs,
            partial_path,
            class_train,
            class_test,
            "stream",
            scenario="class",
            method=method,
            seed=0,
            epochs=1,
        )

    payload = {
        "config": {
            "lr": LR,
            "momentum": MOMENTUM,
            "epochs": args.epochs,
            "seeds": list(args.seeds),
            "buffer_capacity": BUFFER_CAPACITY,
            "ewc_lambda": EWC_LAMBDA,
            "hidden": HIDDEN,
            "fisher_samples": FISHER_SAMPLES,
        },
        "main": [
            run
            for run in runs
            if run["tag"] == "main"
            and not run.get("diverged")
            and run["seed"] in args.seeds
            and run["epochs"] == args.epochs
        ],
        "diverged": [
            run
            for run in runs
            if run.get("diverged") and run["seed"] in args.seeds and run.get("epochs") == args.epochs
        ],
        "speed": [
            run
            for run in runs
            if run["tag"] == "speed" and not run.get("diverged") and run["epochs"] == args.epochs
        ],
        "lambda": [run for run in runs if run["tag"] == "lambda" and run["epochs"] == args.epochs],
        "buffer": [run for run in runs if run["tag"] == "buffer" and run["epochs"] == args.epochs],
        "stream": [run for run in runs if run["tag"] == "stream"],
    }
    metrics_path = args.results / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    figures = args.results / "figures"
    plot_forgetting_matrices(payload["main"], figures / "forgetting_matrices_class.png")
    plot_accuracy_curves(payload["main"], figures / "accuracy_curves.png")
    plot_final_bars(payload["main"], figures / "final_bars.png")
    plot_forgetting_speed(payload["speed"], figures / "forgetting_speed.png")
    plot_lambda_sweep(payload["lambda"], figures / "ewc_lambda.png")
    joint = [run["avg_accuracy"] for run in payload["main"] if run["scenario"] == "class" and run["method"] == "joint"]
    joint_accuracy = sum(joint) / len(joint)
    plot_buffer_sweep(payload["buffer"], joint_accuracy, figures / "replay_buffer.png")

    checks = write_summary(payload, args.results / "summary.md")
    failed = [check for check in checks if not check["held"]]
    print(f"Wrote {args.results / 'summary.md'}", flush=True)
    for check in checks:
        mark = "held" if check["held"] else "NOT held"
        print(f"  {mark}: {check['name']}", flush=True)
    if failed:
        print(f"{len(failed)} expected patterns did not hold. See the summary for the measured values.", flush=True)


if __name__ == "__main__":
    main()
