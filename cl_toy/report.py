"""Turn measured runs into a short written summary."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from cl_toy.plots import METHOD_LABELS, SCENARIO_LABELS

SCENARIO_ORDER = ("task", "domain", "class")


def _group(runs: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["scenario"], run["method"])].append(run)
    return grouped


def _values(grouped: dict, scenario: str, method: str, field: str) -> np.ndarray:
    runs = grouped[(scenario, method)]
    return np.array([run[field] for run in runs if run[field] is not None], dtype=float)


def _fmt_mean_std(values: np.ndarray, scale: float = 100.0) -> str:
    if values.size == 0:
        return "n/a"
    mean = float(values.mean() * scale)
    if values.size == 1:
        return f"{mean:.1f}%"
    return f"{mean:.1f}% ± {float(values.std() * scale):.1f}"


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def _beats(better: np.ndarray, worse: np.ndarray) -> tuple[bool, float, float]:
    """True when the mean gap exceeds the larger seed spread."""
    gap = _mean(better) - _mean(worse)
    spread = max(float(better.std()) if better.size else 0.0, float(worse.std()) if worse.size else 0.0)
    return gap > spread, gap, spread


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _points(value: float) -> str:
    return f"{100 * value:.1f} points"


def write_summary(payload: dict, path: Path) -> list[dict]:
    """Write results/summary.md and return the confirmation checks."""
    main = payload["main"]
    grouped = _group(main)
    checks = _checks(payload, grouped)
    lines = _prose(payload, grouped, checks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checks


def _checks(payload: dict, grouped: dict) -> list[dict]:
    checks: list[dict] = []

    naive_task_diag = []
    for run in grouped[("task", "naive")]:
        naive_task_diag.extend(run["diagonal"])
    diag = np.array(naive_task_diag, dtype=float)
    checks.append(
        {
            "name": "Tasks are actually learned",
            "held": bool(diag.size and diag.min() >= 0.90),
            "detail": (
                f"Naive task-incremental accuracy just after each task ranges from "
                f"{_pct(float(diag.min()))} to {_pct(float(diag.max()))}."
            ),
        }
    )

    class_naive = grouped[("class", "naive")]
    diag_means = [float(np.mean(run["diagonal"])) for run in class_naive]
    final_means = [run["avg_accuracy"] for run in class_naive]
    drop = float(np.mean(diag_means) - np.mean(final_means))
    checks.append(
        {
            "name": "Naive class-incremental training forgets",
            "held": drop >= 0.20,
            "detail": (
                f"Mean just-learned accuracy is {_pct(float(np.mean(diag_means)))}; "
                f"final average accuracy is {_fmt_mean_std(_values(grouped, 'class', 'naive', 'avg_accuracy'))}. "
                f"The drop is {_points(drop)}."
            ),
        }
    )

    task_acc = _values(grouped, "task", "naive", "avg_accuracy")
    class_acc = _values(grouped, "class", "naive", "avg_accuracy")
    held, gap, spread = _beats(task_acc, class_acc)
    checks.append(
        {
            "name": "Task-incremental naive ends higher than class-incremental naive",
            "held": held,
            "detail": (
                f"Task-incremental {_fmt_mean_std(task_acc)} versus class-incremental "
                f"{_fmt_mean_std(class_acc)} (gap {_points(gap)}, seed spread {_points(spread)})."
            ),
        }
    )

    domain_acc = _values(grouped, "domain", "naive", "avg_accuracy")
    checks.append(
        {
            "name": "Domain-incremental naive is reported as measured",
            "held": True,
            "detail": f"Final average accuracy {_fmt_mean_std(domain_acc)}.",
        }
    )

    replay_acc = _values(grouped, "class", "replay", "avg_accuracy")
    replay_forgetting = _values(grouped, "class", "replay", "avg_forgetting")
    naive_forgetting = _values(grouped, "class", "naive", "avg_forgetting")
    acc_held, acc_gap, acc_spread = _beats(replay_acc, class_acc)
    forg_held, forg_gap, forg_spread = _beats(naive_forgetting, replay_forgetting)
    checks.append(
        {
            "name": "Replay beats naive on class-incremental accuracy and forgetting",
            "held": acc_held and forg_held,
            "detail": (
                f"Accuracy {_fmt_mean_std(replay_acc)} versus {_fmt_mean_std(class_acc)} "
                f"(gap {_points(acc_gap)}, spread {_points(acc_spread)}). "
                f"Forgetting {_fmt_mean_std(replay_forgetting)} versus {_fmt_mean_std(naive_forgetting)} "
                f"(naive is higher by {_points(forg_gap)}, spread {_points(forg_spread)})."
            ),
        }
    )

    joint = _values(grouped, "class", "joint", "avg_accuracy")
    joint_held, joint_gap, joint_spread = _beats(joint, class_acc)
    checks.append(
        {
            "name": "Joint training ends above naive class-incremental training",
            "held": joint_held,
            "detail": (
                f"Joint {_fmt_mean_std(joint)} versus naive {_fmt_mean_std(class_acc)} "
                f"(gap {_points(joint_gap)}, spread {_points(joint_spread)})."
            ),
        }
    )

    speed = payload.get("speed", [])
    if speed:
        naive_speed = next(run for run in speed if run["method"] == "naive")
        curve = naive_speed["speed"]["task0_accuracy"]
        start = curve[0]
        early_index = max(1, len(curve) // 10)
        early = curve[early_index]
        early_drop = start - early
        checks.append(
            {
                "name": "Task 1 falls sharply early in task 2",
                "held": early_drop >= 0.15,
                "detail": (
                    f"Class-incremental naive accuracy on task 1 starts at {_pct(start)} and is "
                    f"{_pct(early)} after {early_index} steps of task 2 "
                    f"(drop {_points(early_drop)}). At the end of task 2 it is {_pct(curve[-1])}."
                ),
            }
        )

    lambdas = [run for run in payload.get("lambda", []) if run["scenario"] == "task"]
    if lambdas:
        naive_seed0 = next(run["avg_accuracy"] for run in grouped[("task", "naive")] if run["seed"] == lambdas[0]["seed"])
        best = max(lambdas, key=lambda run: run["avg_accuracy"])
        low = min(lambdas, key=lambda run: run["ewc_lambda"])
        high = max(lambdas, key=lambda run: run["ewc_lambda"])
        checks.append(
            {
                "name": "Some EWC lambda beats naive on task-incremental accuracy",
                "held": best["avg_accuracy"] > naive_seed0,
                "detail": (
                    f"Best lambda {best['ewc_lambda']:g} reaches {_pct(best['avg_accuracy'])}; "
                    f"naive on the same seed is {_pct(naive_seed0)}."
                ),
            }
        )
        # The first task can already be perfect at a small lambda. "Held up" means it
        # stays within a point of that, while the last task actually gets harder.
        high_protects = high["matrix"][-1][0] >= low["matrix"][-1][0] - 0.01
        high_hurts_new = high["matrix"][-1][-1] < low["matrix"][-1][-1] - 0.02
        checks.append(
            {
                "name": "A large EWC lambda holds the first task up and the last task down",
                "held": high_protects and high_hurts_new,
                "detail": (
                    f"Lambda {low['ewc_lambda']:g}: task 1 {_pct(low['matrix'][-1][0])}, "
                    f"last task {_pct(low['matrix'][-1][-1])}. "
                    f"Lambda {high['ewc_lambda']:g}: task 1 {_pct(high['matrix'][-1][0])}, "
                    f"last task {_pct(high['matrix'][-1][-1])}."
                ),
            }
        )
        tiny_gap = abs(low["avg_accuracy"] - naive_seed0)
        checks.append(
            {
                "name": "A tiny EWC lambda behaves like naive training",
                "held": tiny_gap <= 0.05,
                "detail": (
                    f"Lambda {low['ewc_lambda']:g} average accuracy {_pct(low['avg_accuracy'])} "
                    f"versus naive {_pct(naive_seed0)} (absolute gap {_points(tiny_gap)})."
                ),
            }
        )

    class_ewc = _values(grouped, "class", "ewc", "avg_accuracy")
    if class_ewc.size:
        checks.append(
            {
                "name": "Class-incremental EWC is reported at its real effect",
                "held": True,
                "detail": (
                    f"EWC {_fmt_mean_std(class_ewc)} versus naive {_fmt_mean_std(class_acc)} "
                    f"at lambda {payload['config']['ewc_lambda']:g}."
                ),
            }
        )

    agem = _values(grouped, "class", "agem", "avg_accuracy")
    agem_held, agem_gap, agem_spread = _beats(agem, class_acc)
    checks.append(
        {
            "name": "A-GEM beats naive on class-incremental accuracy",
            "held": agem_held,
            "detail": (
                f"A-GEM {_fmt_mean_std(agem)} versus naive {_fmt_mean_std(class_acc)} "
                f"(gap {_points(agem_gap)}, spread {_points(agem_spread)})."
            ),
        }
    )

    lwf = _values(grouped, "class", "lwf", "avg_accuracy")
    checks.append(
        {
            "name": "Learning without Forgetting is reported at its real effect",
            "held": True,
            "detail": (
                f"Class-incremental LwF {_fmt_mean_std(lwf)} versus naive {_fmt_mean_std(class_acc)}. "
                f"Task-incremental LwF {_fmt_mean_std(_values(grouped, 'task', 'lwf', 'avg_accuracy'))} "
                f"versus naive {_fmt_mean_std(task_acc)}."
            ),
        }
    )

    expert_forgetting = _values(grouped, "task", "experts", "avg_forgetting")
    checks.append(
        {
            "name": "Oracle experts on task-incremental learning almost do not forget",
            "held": bool(expert_forgetting.size and _mean(expert_forgetting) <= 0.05),
            "detail": f"Average forgetting {_fmt_mean_std(expert_forgetting)}.",
        }
    )

    if ("class", "experts_routed") in grouped:
        routed = _values(grouped, "class", "experts_routed", "avg_accuracy")
        oracle = _values(grouped, "class", "experts", "avg_accuracy")
        route_held, route_gap, route_spread = _beats(oracle, routed)
        checks.append(
            {
                "name": "Predicted routing on class-incremental learning lands below the oracle",
                "held": route_held,
                "detail": (
                    f"Oracle experts {_fmt_mean_std(oracle)} versus predicted routing "
                    f"{_fmt_mean_std(routed)} (gap {_points(route_gap)}, spread {_points(route_spread)})."
                ),
            }
        )

    buffer_runs = sorted(payload.get("buffer", []), key=lambda run: run["buffer_capacity"])
    if len(buffer_runs) >= 2 and joint.size:
        accuracies = [run["avg_accuracy"] for run in buffer_runs]
        monotonic = all(accuracies[index] <= accuracies[index + 1] + 0.02 for index in range(len(accuracies) - 1))
        closer = abs(accuracies[-1] - _mean(joint)) < abs(accuracies[0] - _mean(joint))
        detail_bits = [
            f"size {run['buffer_capacity']}: {_pct(run['avg_accuracy'])}" for run in buffer_runs
        ]
        checks.append(
            {
                "name": "A larger replay buffer is closer to joint training",
                "held": monotonic and closer,
                "detail": (
                    f"{'; '.join(detail_bits)}. Joint is {_fmt_mean_std(joint)}. "
                    f"Monotonic within 2 points: {monotonic}. Largest buffer closer to joint: {closer}."
                ),
            }
        )

    stream = payload.get("stream", [])
    if stream:
        stream_naive = next(run for run in stream if run["method"] == "naive")
        stream_replay = next(run for run in stream if run["method"] == "replay")
        stream_drop = float(np.mean(stream_naive["diagonal"]) - stream_naive["avg_accuracy"])
        checks.append(
            {
                "name": "A single pass still forgets, and replay still helps",
                "held": stream_drop >= 0.15 and stream_replay["avg_accuracy"] > stream_naive["avg_accuracy"],
                "detail": (
                    f"One-epoch class-incremental naive final accuracy {_pct(stream_naive['avg_accuracy'])} "
                    f"(drop from the diagonal {_points(stream_drop)}). "
                    f"Replay finishes at {_pct(stream_replay['avg_accuracy'])}."
                ),
            }
        )

    return checks


def _diverged_count(diverged: list, scenario: str, method: str) -> int:
    return sum(
        1
        for run in diverged
        if run.get("tag", "main") == "main" and run["scenario"] == scenario and run["method"] == method
    )


def _merged_sentence(grouped: dict, diverged: list) -> str:
    """Report the merged scores, including repeats whose loss left the finite range."""
    if ("task", "merged") not in grouped and ("domain", "merged") not in grouped and ("class", "merged") not in grouped:
        return ""
    task = _fmt_mean_std(_values(grouped, "task", "merged", "avg_accuracy")) if ("task", "merged") in grouped else None
    domain = _fmt_mean_std(_values(grouped, "domain", "merged", "avg_accuracy")) if ("domain", "merged") in grouped else None
    failed = _diverged_count(diverged, "class", "merged")
    sentence = (
        "Merged, which uses the weight penalty, the frozen copy, and the step check together, "
    )
    if task and domain:
        sentence += f"reaches {task} when the lesson is known and {domain} on even versus odd. "
    if failed and ("class", "merged") in grouped:
        finished = len(grouped[("class", "merged")])
        total = finished + failed
        score = _fmt_mean_std(_values(grouped, "class", "merged", "avg_accuracy"))
        sentence += (
            f"On naming the digit, {failed} of {total} repeats left the finite range on the last lesson; "
            f"the repeat that finished reached {score}. "
        )
    elif ("class", "merged") in grouped:
        score = _fmt_mean_std(_values(grouped, "class", "merged", "avg_accuracy"))
        sentence += f"On naming the digit it reaches {score}. "
    return sentence


def _picture(grouped: dict, diverged: list | None = None) -> str:
    """One paragraph of the measured pattern, computed from the same runs as the checks."""
    class_naive = _fmt_mean_std(_values(grouped, "class", "naive", "avg_accuracy"))
    class_replay = _fmt_mean_std(_values(grouped, "class", "replay", "avg_accuracy"))
    class_joint = _fmt_mean_std(_values(grouped, "class", "joint", "avg_accuracy"))
    class_ewc = _fmt_mean_std(_values(grouped, "class", "ewc", "avg_accuracy"))
    class_lwf = _fmt_mean_std(_values(grouped, "class", "lwf", "avg_accuracy"))
    class_agem = _fmt_mean_std(_values(grouped, "class", "agem", "avg_accuracy"))
    class_oracle = _fmt_mean_std(_values(grouped, "class", "experts", "avg_accuracy"))
    class_routed = _fmt_mean_std(_values(grouped, "class", "experts_routed", "avg_accuracy"))
    task_naive = _fmt_mean_std(_values(grouped, "task", "naive", "avg_accuracy"))
    domain_naive = _fmt_mean_std(_values(grouped, "domain", "naive", "avg_accuracy"))
    domain_replay = _fmt_mean_std(_values(grouped, "domain", "replay", "avg_accuracy"))
    return (
        f"On this network, task-incremental naive training barely forgets ({task_naive}), "
        f"because each pair keeps its own output head. "
        f"Domain-incremental naive training, even versus odd with a shared head, ends at {domain_naive}; "
        f"replay brings that to {domain_replay}. "
        f"Class-incremental naive training is the collapse: final accuracy {class_naive}, "
        f"against joint training at {class_joint}. "
        f"Replay reaches {class_replay}. "
        f"A-GEM, which only constrains the step with stored gradients, reaches {class_agem}. "
        f"EWC at the default penalty stays near the naive result ({class_ewc}); "
        f"learning without forgetting, which stores no old images, reaches {class_lwf}. "
        f"{_merged_sentence(grouped, diverged or [])}"
        f"A frozen expert per task scores {class_oracle} when it is handed the true task id, "
        f"and {class_routed} when a sequentially trained router has to guess that id."
    )


def _prose(payload: dict, grouped: dict, checks: list[dict]) -> list[str]:
    config = payload["config"]
    lines = [
        "# Split MNIST continual-learning results",
        "",
        "## Setup",
        "",
        "Each run trains a 784-256-256 multilayer perceptron with SGD "
        f"(learning rate {config['lr']}, momentum {config['momentum']}, no weight decay) "
        f"for {config['epochs']} epochs on each of five Split MNIST tasks: "
        "(0, 1), (2, 3), (4, 5), (6, 7), (8, 9). "
        "Task-incremental evaluation chooses between the two digits of a known pair. "
        "Domain-incremental evaluation asks even versus odd with no task id. "
        "Class-incremental evaluation names the digit among all ten with no task id. "
        f"The main comparison uses seeds {config['seeds']}. "
        f"Replay stores {config['buffer_capacity']} examples. "
        f"EWC uses lambda {config['ewc_lambda']:g}. "
        "Accuracy in the saved metrics is a fraction from 0 to 1; the figures and this note use percents. "
        "The EWC Fisher is divided by its mean and capped so a large penalty pins weights instead of making SGD diverge. "
        "A-GEM projects the momentum update, because projecting only the latest gradient leaves the old velocity free to increase loss on stored examples.",
        "",
        "## The picture",
        "",
        _picture(grouped, payload.get("diverged", [])),
        "",
        "## What the runs showed",
        "",
    ]
    for scenario in SCENARIO_ORDER:
        bits = []
        diverged = payload.get("diverged", [])
        for method in (
            "naive",
            "replay",
            "ewc",
            "lwf",
            "agem",
            "merged",
            "experts",
            "experts_routed",
            "joint",
        ):
            if (scenario, method) not in grouped:
                continue
            accuracy = _fmt_mean_std(_values(grouped, scenario, method, "avg_accuracy"))
            forgetting_values = _values(grouped, scenario, method, "avg_forgetting")
            failed = _diverged_count(diverged, scenario, method)
            if forgetting_values.size:
                forgetting = _fmt_mean_std(forgetting_values)
                text = f"{METHOD_LABELS[method]} accuracy {accuracy}, forgetting {forgetting}"
            else:
                text = f"{METHOD_LABELS[method]} accuracy {accuracy}"
            if failed:
                finished = len(grouped[(scenario, method)])
                text += f" ({failed} of {finished + failed} repeats left the finite range)"
            bits.append(text)
        lines.append(f"{SCENARIO_LABELS[scenario]}: " + "; ".join(bits) + ".")
        lines.append("")

    lines.append("Parameter counts, identical across seeds:")
    lines.append("")
    seen = set()
    for run in payload["main"]:
        key = (run["scenario"], run["method"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- {SCENARIO_LABELS[run['scenario']]}, {METHOD_LABELS[run['method']]}: {run['params']:,} parameters."
        )
    lines.append("")
    lines.append("## Checks against the expected pattern")
    lines.append("")
    for check in checks:
        mark = "held" if check["held"] else "did not hold"
        lines.append(f"- {check['name']}: {mark}. {check['detail']}")
    lines.append("")
    lines.append("## How to read a partial result")
    lines.append("")
    lines.append(
        "A method works to some degree on a metric when its mean beats naive training by more than the seed spread. "
        "Elastic weight consolidation and learning without forgetting store no old images, so a small class-incremental gain is still the measured effect. "
        "Oracle experts are shown the true task id in every setting, including class-incremental learning, where that id is not part of the problem. "
        "Predicted routing is the version that has to discover the task. "
        "Joint training sees every task in a shuffled mix for the same number of epochs and is the ceiling when keeping all of the data is allowed."
    )
    lines.append("")
    return lines
