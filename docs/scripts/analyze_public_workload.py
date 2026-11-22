#!/usr/bin/env python3
"""Verify the complete paired design and analyze the frozen comparison family."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
from pathlib import Path
from statistics import median

INCUMBENT = (
    "finite_loss", "nonzero_gradients", "weight_update", "early_loss_decrease",
    "tiny_batch_overfit", "training_mode", "preprocessing_contract",
)
POLICIES = {
    "finite": ("finite_loss",),
    "validation_loss": ("finite_loss", "early_loss_decrease"),
    "seven": INCUMBENT,
    "seven_plus_updates": (*INCUMBENT, "expected_updates"),
    "updates": ("expected_updates",),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision(row: dict, policy: str) -> bool | None:
    if row["status"] == "failed":
        return None
    return any(not row["detections"][key] for key in POLICIES[policy])


def wilson(successes: int, n: int) -> tuple[float | None, float | None]:
    if not n:
        return None, None
    z = 1.959963984540054
    rate = successes / n
    scale = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / scale
    width = z * math.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / scale
    return max(0.0, center - width), min(1.0, center + width)


def mcnemar(left: list[bool | None], right: list[bool | None]) -> dict:
    pairs = [(a, b) for a, b in zip(left, right, strict=True) if a is not None and b is not None]
    left_only = sum(a and not b for a, b in pairs)
    right_only = sum(b and not a for a, b in pairs)
    discordant = left_only + right_only
    tail = sum(math.comb(discordant, i) for i in range(min(left_only, right_only) + 1))
    return {
        "pairs": len(pairs), "left_only": left_only, "right_only": right_only,
        "left_minus_right": (left_only - right_only) / len(pairs) if pairs else None,
        "exact_p": min(1.0, 2 * tail / 2**discordant) if pairs else 1.0,
    }


def apply_holm(tests: list[dict]) -> None:
    adjusted = 0.0
    for rank, test in enumerate(sorted(tests, key=lambda entry: entry["exact_p"])):
        adjusted = max(adjusted, min(1.0, (len(tests) - rank) * test["exact_p"]))
        test["holm_p"] = adjusted


def validate(bundle: Path, design_path: Path, workload: str) -> tuple[dict, dict]:
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    design = json.loads(design_path.read_bytes())
    phase = manifest["phase"]
    if (
        manifest["design"] != design or manifest["design_sha256"] != digest(design_path)
        or manifest["workload"] != workload or phase not in ("development", "confirmatory")
        or (phase == "confirmatory" and (design["status"] != "frozen" or manifest["tracked_source_changes"]))
        or digest(bundle / "records.jsonl") != manifest["records_sha256"]
    ):
        raise ValueError(f"Manifest provenance does not match {bundle}")
    if phase == "confirmatory" and (
        manifest["data_sha256"] != design["data_sha256"]
        or (os.environ.get("GITHUB_SHA") and manifest["source_revision"] != os.environ["GITHUB_SHA"])
    ):
        raise ValueError(f"Confirmatory source or dataset mismatch in {bundle}")
    cells = list(itertools.product(design["faults"], design["budgets_seconds"]))
    start, stop = design[f"{phase}_seed_range"]
    expected = []
    for seed in range(start, stop + 1):
        order = cells.copy()
        random.Random(design["order_seed_offset"] + seed).shuffle(order)
        expected.extend((seed, fault, budget, i) for i, (fault, budget) in enumerate(order))
    rows = {}
    lines = (bundle / "records.jsonl").read_text().splitlines()
    if len(lines) != len(expected) or len(lines) != manifest["record_count"]:
        raise ValueError(f"Incomplete record grid in {bundle}")
    failures = 0
    for line, key in zip(lines, expected, strict=True):
        row = json.loads(line)
        if (
            (row["seed"], row["fault"], row["budget_seconds"], row["position"]) != key
            or row["workload"] != workload or row["phase"] != phase
            or row["source_revision"] != manifest["source_revision"]
        ):
            raise ValueError(f"Invalid record identity or order in {bundle}: {key}")
        if row["status"] == "ok":
            if any(type(row["detections"].get(k)) is not bool for k in (*INCUMBENT, "expected_updates")):
                raise ValueError(f"Missing diagnostic value in {bundle}: {key}")
            steps, calls = row["steps"], row["optimizer_step_calls"]
            if (
                type(steps) is not int or type(calls) is not int
                or not 0 <= calls <= steps <= design["max_steps"]
                or row["detections"]["expected_updates"] != (steps > 0 and calls == steps)
            ):
                raise ValueError(f"Inconsistent optimizer trace in {bundle}: {key}")
            expected_calls = {"skip_step": 0, "intermittent_skip_step": steps - (steps + 3) // 4}
            if calls != expected_calls.get(row["fault"], steps):
                raise ValueError(f"Mutation did not execute the declared schedule in {bundle}: {key}")
            for metric in ("heldout_accuracy", "heldout_clean_accuracy", "heldout_loss", "heldout_clean_loss",
                           "elapsed_seconds", "diagnose_wall_seconds", "attempt_seconds"):
                value = row.get(metric)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid {metric} in {bundle}: {key}")
            if (
                not 0 <= row["heldout_accuracy"] <= 1 or not 0 <= row["heldout_clean_accuracy"] <= 1
                or not row["elapsed_seconds"] <= row["diagnose_wall_seconds"] <= row["attempt_seconds"]
                or row["python_version"] != manifest["python_version"]
                or row["torch_version"] != manifest["torch_version"]
                or row["configuration"]["max_steps"] != design["max_steps"]
                or any(manifest["data_sha256"].get(name) != value
                       for name, value in row["configuration"]["dataset_sha256"].items())
            ):
                raise ValueError(f"Inconsistent measurement in {bundle}: {key}")
        elif row["status"] == "failed":
            failures += 1
        else:
            raise ValueError(f"Invalid record status in {bundle}: {key}")
        rows[key[:3]] = row
    if failures != manifest["failed_runs"]:
        raise ValueError(f"Incorrect failure count in {bundle}")
    return manifest, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_stats(rows: list[dict], key: str) -> tuple[float | None, float | None]:
    values = sorted(row[key] for row in rows if row.get(key) is not None)
    if not values:
        return None, None

    def percentile(fraction: float) -> float:
        position = (len(values) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        return values[lower] + (position - lower) * (values[upper] - values[lower])

    return median(values), percentile(0.75) - percentile(0.25)


def analyze(root: Path, design_path: Path, output: Path) -> None:
    design = json.loads(design_path.read_bytes())
    counts, metrics, tests, budget_pairs, confusion, semantic = [], [], [], [], [], []
    manifests = []
    for workload in design["workloads"]:
        manifest, rows = validate(root / f"public-workload-{workload}", design_path, workload)
        manifests.append(manifest)
        phase = manifest["phase"]
        start, end = design[f"{phase}_seed_range"]
        seeds = range(start, end + 1)
        faults = design["faults"]
        budgets = design["budgets_seconds"]

        def alarms(fault: str, budget: float, policy: str) -> list[bool | None]:
            return [decision(rows[(seed, fault, budget)], policy) for seed in seeds]

        for budget in budgets:
            for fault in faults:
                condition = [rows[(seed, fault, budget)] for seed in seeds]
                fields = {"workload": workload, "budget_seconds": budget, "fault": fault}
                for policy in POLICIES:
                    paired_quality_misses = 0
                    paired_quality_count = 0
                    for seed, row in zip(seeds, condition, strict=True):
                        baseline_quality = rows[(seed, "healthy", budget)].get("heldout_accuracy")
                        quality = row.get("heldout_accuracy")
                        alarm = decision(row, policy)
                        if baseline_quality is None or quality is None or alarm is None:
                            continue
                        paired_quality_count += 1
                        paired_quality_misses += not alarm and baseline_quality - quality >= 0.05
                    semantic.append({
                        **fields, "policy": policy, "paired_quality_count": paired_quality_count,
                        "misses_with_5pp_quality_drop": paired_quality_misses,
                    })
                    decisions = [item for item in alarms(fault, budget, policy) if item is not None]
                    n = len(decisions)
                    positive = sum(decisions)
                    lo, hi = wilson(positive, n)
                    attempted = len(condition)
                    conservative = (positive + attempted - n) / attempted if fault == "healthy" else positive / attempted
                    counts.append({
                        **fields, "policy": policy, "attempted": attempted, "evaluated": n,
                        "failed": attempted - n, "alarms": positive,
                        "alarm_rate": positive / n if n else None, "wilson_low": lo, "wilson_high": hi,
                        "conservative_alarm_rate": conservative,
                    })
                values = {}
                for key in (
                    "elapsed_seconds", "diagnose_wall_seconds", "elapsed_cpu_seconds", "diagnose_cpu_seconds", "attempt_seconds", "steps",
                    "optimizer_step_calls", "budget_overshoot_seconds", "heldout_accuracy",
                    "heldout_loss", "heldout_clean_accuracy", "heldout_clean_loss", "semantic_eval_wall_seconds",
                ):
                    middle, iqr = summary_stats(condition, key)
                    values[f"{key}_median"], values[f"{key}_iqr"] = middle, iqr
                accuracy_deltas, clean_deltas, transform_deltas = [], [], []
                for seed, entry in zip(seeds, condition, strict=True):
                    baseline = rows[(seed, "healthy", budget)]
                    healthy = baseline.get("heldout_accuracy")
                    quality = entry.get("heldout_accuracy")
                    if healthy is not None and quality is not None:
                        accuracy_deltas.append(quality - healthy)
                        clean_deltas.append(entry["heldout_clean_accuracy"] - baseline["heldout_clean_accuracy"])
                        transform_deltas.append(quality - entry["heldout_clean_accuracy"])
                metrics.append({
                    **fields, **values, "quality_pairs": len(accuracy_deltas),
                    "heldout_accuracy_minus_healthy_median": median(accuracy_deltas) if accuracy_deltas else None,
                    "heldout_clean_accuracy_minus_healthy_median": median(clean_deltas) if clean_deltas else None,
                    "serving_minus_clean_accuracy_median": median(transform_deltas) if transform_deltas else None,
                })
            for policy in POLICIES:
                healthy = [a for a in alarms("healthy", budget, policy) if a is not None]
                faulty = [a for f in faults if f != "healthy" for a in alarms(f, budget, policy) if a is not None]
                tp, fp = sum(faulty), sum(healthy)
                confusion.append({
                    "workload": workload, "budget_seconds": budget, "policy": policy,
                    "tp": tp, "fp": fp, "fn": len(faulty) - tp, "tn": len(healthy) - fp,
                    "precision": tp / (tp + fp) if tp + fp else None,
                    "recall": tp / len(faulty) if faulty else None,
                })
            partial = alarms("intermittent_skip_step", budget, "seven")
            full = alarms("skip_step", budget, "seven")
            tests.append({
                "hypothesis": "H1_partial_minus_full_alarm",
                "workload": workload, "budget_seconds": budget, **mcnemar(partial, full),
            })
            lower, higher = [], []
            for seed in seeds:
                healthy = rows[(seed, "healthy", budget)].get("heldout_clean_accuracy")
                faulty = rows[(seed, "intermittent_skip_step", budget)].get("heldout_clean_accuracy")
                valid = healthy is not None and faulty is not None
                lower.append(faulty < healthy if valid else None)
                higher.append(faulty > healthy if valid else None)
            tests.append({
                "hypothesis": "H2_partial_accuracy_lower_vs_higher",
                "workload": workload, "budget_seconds": budget, **mcnemar(lower, higher),
            })
        for fault, policy in itertools.product(faults, POLICIES):
            low, high = budgets
            comparison = mcnemar(alarms(fault, low, policy), alarms(fault, high, policy))
            comparison.pop("exact_p")
            budget_pairs.append({"workload": workload, "fault": fault, "policy": policy, **comparison})
    if len({(m["phase"], m["source_revision"], m["workflow_run_id"], m["workflow_attempt"]) for m in manifests}) != 1:
        raise ValueError("Workload bundles come from different study executions")
    apply_holm(tests)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "alarm-rates.csv", counts)
    write_csv(output / "timing-and-quality.csv", metrics)
    write_csv(output / "paired-hypotheses.csv", tests)
    write_csv(output / "paired-budget-decisions.csv", budget_pairs)
    write_csv(output / "confusion-and-precision.csv", confusion)
    write_csv(output / "quality-linked-misses.csv", semantic)
    (output / "summary.json").write_text(json.dumps({
        "manifests": manifests, "hypotheses": tests, "dataset_cells": len(metrics),
        "analysis_sha256": digest(Path(__file__)),
    }, indent=2) + "\n")
    if any(entry["evaluated"] == 0 for entry in counts):
        raise ValueError("A condition-budget cell has no evaluable attempts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    analyze(arguments.input, arguments.design, arguments.output)
