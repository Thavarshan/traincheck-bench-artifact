#!/usr/bin/env python3
"""Create descriptive check, runtime, and paired-seed summaries from run JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import statistics
from collections import Counter, defaultdict
from pathlib import Path

CHECK_VALIDITY = {
    "finite_loss": "training-step and matched-reference finiteness; candidate; not calibrated",
    "nonzero_gradients": "finite nonzero gradients at every executed step; candidate; not calibrated",
    "weight_update": "at least one parameter changed; direct update observable; candidate",
    "early_loss_decrease": "matched fixed-batch reference losses; cutoff exploratory; correct-label oracle assumed",
    "tiny_batch_overfit": "matched reference-batch loss threshold; not calibrated",
    "training_mode": "mode observed during optimization steps; directly observable contract",
    "preprocessing_contract": "train/serve tensor agreement on shared raw probe; direct contract, not production validation",
}


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z *
                         z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


ROOT = Path(__file__).resolve().parents[2]


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def candidate_files(inputs: list[Path], phase_filter: str | None, run_ids: set[str]) -> list[Path]:
    found: set[Path] = set()
    for path in inputs:
        if path.is_dir():
            found.update(item for item in path.rglob(
                "*.jsonl") if item.is_file())
        elif path.is_file() and path.suffix == ".jsonl":
            found.add(path)
    selected = sorted(found)
    if run_ids:
        selected = [path for path in selected if path.name.split(
            "-", 1)[0] in run_ids]
    if phase_filter:
        filtered: list[Path] = []
        for path in selected:
            try:
                rows = [json.loads(line) for line in path.read_text(
                    encoding="utf-8").splitlines() if line.strip()]
            except json.JSONDecodeError:
                filtered.append(path)
                continue
            if any(isinstance(row, dict) and row.get("phase") == phase_filter for row in rows):
                filtered.append(path)
        selected = filtered
    return selected


def load_file(path: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    malformed: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            malformed.append({"line": line_number, "error": error.msg})
            continue
        if not isinstance(row, dict):
            malformed.append(
                {"line": line_number, "error": "record must be a JSON object"})
            continue
        row["_source_file"] = path.name
        rows.append(row)
    return rows, malformed


def build_summaries(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    grouped: dict[tuple[str, str, str, float], list[dict]] = defaultdict(list)
    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        source_file = str(row.get("_source_file", "unknown"))
        by_file[source_file].append(row)
        budget = row.get("budget_seconds")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool):
            grouped[(source_file, str(row.get("phase", "unknown")), str(
                row.get("fault", "unknown")), float(budget))].append(row)

    check_summaries: list[dict] = []
    runtime_summaries: list[dict] = []
    paired_summaries: list[dict] = []
    confusion_summaries: list[dict] = []
    macro_summaries: list[dict] = []
    for (source_file, phase, fault, budget), condition_rows in sorted(grouped.items()):
        successful = [
            row for row in condition_rows if row.get("status") == "ok"]
        failed_runs = len(condition_rows) - len(successful)
        check_names = sorted(
            {name for row in successful for name in (row.get("detections") or {})})
        for check in check_names:
            observed = [
                row["detections"][check]
                for row in successful
                if isinstance((row.get("detections") or {}).get(check), bool)
            ]
            check_failures = sum(not value for value in observed)
            low, high = wilson(check_failures, len(observed))
            check_summaries.append({
                "phase": phase,
                "source_file": source_file,
                "fault": fault,
                "budget_seconds": budget,
                "check": check,
                "n_records": len(condition_rows),
                "n_successful": len(successful),
                "n_failed_runs": failed_runs,
                "n_observed": len(observed),
                "n_missing_check": len(successful) - len(observed),
                "check_failures": check_failures,
                "failure_rate": check_failures / len(observed) if observed else None,
                "wilson_95_low": low,
                "wilson_95_high": high,
                "interpretation": CHECK_VALIDITY.get(check, "unclassified candidate check"),
            })

        elapsed = [float(row["elapsed_seconds"]) for row in successful if isinstance(
            row.get("elapsed_seconds"), (int, float))]
        over_budget = [row.get("over_budget") for row in successful if isinstance(
            row.get("over_budget"), bool)]
        optimizer_calls = [
            row["optimizer_step_calls"] for row in successful
            if type(row.get("optimizer_step_calls")) is int
        ]
        overshoots = [
            float(row["budget_overshoot_seconds"]) for row in successful
            if type(row.get("budget_overshoot_seconds")) in (int, float)
            and math.isfinite(row["budget_overshoot_seconds"])
        ]
        runtime_summaries.append({
            "phase": phase,
            "source_file": source_file,
            "fault": fault,
            "budget_seconds": budget,
            "n_records": len(condition_rows),
            "n_successful": len(successful),
            "n_failed_runs": failed_runs,
            "n_elapsed": len(elapsed),
            "elapsed_median_seconds": statistics.median(elapsed) if elapsed else None,
            "elapsed_iqr_seconds": (quantile(elapsed, 0.75) - quantile(elapsed, 0.25)) if elapsed else None,
            "n_optimizer_call_counts": len(optimizer_calls),
            "optimizer_step_calls_median": statistics.median(optimizer_calls) if optimizer_calls else None,
            "n_overshoot_amounts": len(overshoots),
            "budget_overshoot_median_ms": 1000 * statistics.median(overshoots) if overshoots else None,
            "budget_overshoot_max_ms": 1000 * max(overshoots) if overshoots else None,
            "n_over_budget": sum(value is True for value in over_budget),
            "n_budget_observed": len(over_budget),
            "budget_overrun_rate": sum(value is True for value in over_budget) / len(over_budget) if over_budget else None,
            "termination_reasons": dict(Counter(str(row.get("termination_reason", "unknown")) for row in successful)),
        })

    # Keep each workflow run separate. Paired contrasts use only successful
    # healthy/fault rows with the same seed, phase, and requested budget.
    for source_file, file_rows in sorted(by_file.items()):
        requested_faults = next(
            (row.get("_requested_faults")
             for row in file_rows if isinstance(row.get("_requested_faults"), list)),
            None,
        )
        if requested_faults is None:
            requested_faults = sorted(
                {str(row.get("fault", "unknown")) for row in file_rows})
        faults = list(dict.fromkeys(str(fault)
                      for fault in requested_faults if fault != "healthy"))
        checks = sorted(
            {name for row in file_rows for name in (row.get("detections") or {})})
        phases = {str(row.get("phase", "unknown")) for row in file_rows}
        budgets = sorted({
            float(row["budget_seconds"])
            for row in file_rows
            if isinstance(row.get("budget_seconds"), (int, float)) and not isinstance(row.get("budget_seconds"), bool)
        })
        for phase in phases:
            for budget in budgets:
                confusion: dict[str, dict[str, int]] = defaultdict(
                    lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "observed": 0}
                )
                per_fault: dict[tuple[str, str], list[int]] = defaultdict(lambda: [
                                                                          0, 0])
                per_fault_successes: Counter[tuple[str, str]] = Counter()
                per_fault_failures: Counter[str] = Counter()
                for row in file_rows:
                    if row.get("phase") == phase and row.get("budget_seconds") == budget:
                        row_fault = str(row.get("fault", "unknown"))
                        if row_fault != "healthy" and row.get("status") != "ok":
                            per_fault_failures[row_fault] += 1
                for row in file_rows:
                    if row.get("phase") != phase or row.get("budget_seconds") != budget or row.get("status") != "ok":
                        continue
                    actual_positive = row.get("fault") != "healthy"
                    row_fault = str(row.get("fault", "unknown"))
                    if actual_positive:
                        for check in checks:
                            per_fault_successes[(check, row_fault)] += 1
                    for check, passed in (row.get("detections") or {}).items():
                        if not isinstance(passed, bool):
                            continue
                        predicted_positive = not passed
                        counts = confusion[check]
                        counts["observed"] += 1
                        if actual_positive and predicted_positive:
                            counts["tp"] += 1
                        elif actual_positive:
                            counts["fn"] += 1
                        elif predicted_positive:
                            counts["fp"] += 1
                        else:
                            counts["tn"] += 1
                        if actual_positive:
                            class_counts = per_fault[(
                                check, str(row.get("fault")))]
                            class_counts[0] += int(predicted_positive)
                            class_counts[1] += 1

                for check, counts in sorted(confusion.items()):
                    confusion_summaries.append({
                        "phase": phase,
                        "source_file": source_file,
                        "budget_seconds": budget,
                        "check": check,
                        **counts,
                        "interpretation": CHECK_VALIDITY.get(check, "unclassified candidate check"),
                    })
                    class_recalls = [
                        per_fault[(check, fault)][0] /
                        per_fault[(check, fault)][1]
                        for fault in faults
                        if per_fault.get((check, fault), [0, 0])[1] > 0
                    ]
                    complete_macro = bool(faults) and all(
                        per_fault_successes[(check, fault)] > 0
                        and per_fault[(check, fault)][1] == per_fault_successes[(check, fault)]
                        for fault in faults
                    )
                    macro_summaries.append({
                        "phase": phase,
                        "source_file": source_file,
                        "budget_seconds": budget,
                        "check": check,
                        "fault_classes_expected": len(faults),
                        "fault_classes_observed": len(class_recalls),
                        "successful_fault_runs": sum(per_fault_successes[(check, fault)] for fault in faults),
                        "observed_fault_runs": sum(per_fault[(check, fault)][1] for fault in faults),
                        "failed_fault_runs": sum(per_fault_failures[fault] for fault in faults),
                        "missing_check_outputs": sum(
                            per_fault_successes[(check, fault)] -
                            per_fault[(check, fault)][1]
                            for fault in faults
                        ),
                        "macro_recall": statistics.mean(class_recalls) if complete_macro else None,
                        "macro_complete_for_successful_runs": complete_macro,
                        "interpretation": CHECK_VALIDITY.get(check, "unclassified candidate check"),
                    })

                healthy_rows = [
                    row for row in file_rows
                    if row.get("phase") == phase and row.get("fault") == "healthy"
                    and row.get("budget_seconds") == budget and row.get("status") == "ok"
                ]
                healthy_by_seed = {row.get(
                    "seed"): row for row in healthy_rows if isinstance(row.get("seed"), int)}
                for fault in faults:
                    fault_rows = [
                        row for row in file_rows
                        if row.get("phase") == phase and row.get("fault") == fault
                        and row.get("budget_seconds") == budget and row.get("status") == "ok"
                    ]
                    fault_by_seed = {
                        row.get("seed"): row for row in fault_rows if isinstance(row.get("seed"), int)}
                    for check in checks:
                        differences: list[int] = []
                        for seed in sorted(set(healthy_by_seed) & set(fault_by_seed)):
                            healthy_value = (healthy_by_seed[seed].get(
                                "detections") or {}).get(check)
                            fault_value = (fault_by_seed[seed].get(
                                "detections") or {}).get(check)
                            if isinstance(healthy_value, bool) and isinstance(fault_value, bool):
                                differences.append(
                                    int(not fault_value) - int(not healthy_value))
                        paired_summaries.append({
                            "phase": phase,
                            "source_file": source_file,
                            "fault": fault,
                            "budget_seconds": budget,
                            "check": check,
                            "n_paired_seeds": len(differences),
                            "mean_paired_failure_difference": statistics.mean(differences) if differences else None,
                            "fault_only_failures": sum(value > 0 for value in differences),
                            "healthy_only_failures": sum(value < 0 for value in differences),
                            "same_outcome": sum(value == 0 for value in differences),
                            "interpretation": "descriptive paired differences; no significance test",
                        })
    return check_summaries, runtime_summaries, paired_summaries, confusion_summaries, macro_summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            for row in rows
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+",
                        required=True, help="JSONL files and/or directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("exploratory", "confirmatory"))
    parser.add_argument("--run-ids", default="",
                        help="Optional comma-separated GitHub workflow run IDs")
    args = parser.parse_args()
    run_ids = {value.strip()
               for value in args.run_ids.split(",") if value.strip()}
    paths = candidate_files(args.records, args.phase, run_ids)
    if not paths:
        parser.error("no matching JSONL run records were found")

    all_rows: list[dict] = []
    sources: list[dict] = []
    malformed_total = 0
    for path in paths:
        rows, malformed = load_file(path)
        if args.phase:
            rows = [row for row in rows if row.get("phase") == args.phase]
        manifest_path = path.with_name(path.stem + ".manifest.json")
        manifest_status = "missing"
        try:
            manifest = json.loads(manifest_path.read_text(
                encoding="utf-8")) if manifest_path.is_file() else None
            manifest_error = None
            if manifest_path.is_file():
                manifest_status = "present"
        except json.JSONDecodeError as error:
            manifest = None
            manifest_error = error.msg
            manifest_status = "malformed"
        if isinstance(manifest, dict) and isinstance(manifest.get("requested_faults"), list):
            for row in rows:
                row["_requested_faults"] = manifest["requested_faults"]
        expected_record_count = None
        expected_combinations: set[tuple[int, str, float]] = set()
        if isinstance(manifest, dict):
            try:
                expected_seeds = manifest["requested_seeds"]
                expected_faults = manifest["requested_faults"]
                expected_budgets = manifest["requested_budgets_seconds"]
                expected_record_count = len(
                    expected_seeds) * len(expected_faults) * len(expected_budgets)
                expected_combinations = {
                    (int(seed), str(fault), float(budget))
                    for seed in expected_seeds for fault in expected_faults for budget in expected_budgets
                }
            except (KeyError, TypeError):
                expected_record_count = None
                expected_combinations = set()
        observed_counts: Counter[tuple[int, str, float]] = Counter(
            (row.get("seed"), str(row.get("fault")),
             float(row.get("budget_seconds")))
            for row in rows
            if isinstance(row.get("seed"), int)
            and isinstance(row.get("budget_seconds"), (int, float))
            and not isinstance(row.get("budget_seconds"), bool)
        )
        observed_combinations = set(observed_counts)
        missing_combinations = sorted(
            expected_combinations - observed_combinations)
        unexpected_combinations = sorted(
            observed_combinations - expected_combinations) if expected_combinations else []
        duplicate_combinations = sorted(
            item for item, count in observed_counts.items() if count > 1)
        record_set_complete = (
            expected_record_count is not None
            and expected_record_count == len(rows)
            and not missing_combinations
            and not unexpected_combinations
            and not duplicate_combinations
            and not malformed
        )
        sources.append({
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "record_count": len(rows),
            "manifest_status": manifest_status,
            "expected_record_count": expected_record_count,
            "record_set_complete": record_set_complete,
            "missing_or_invalid_record_count": (
                max(0, expected_record_count - len(rows))
                if expected_record_count is not None
                else None
            ),
            "missing_combinations": [list(item) for item in missing_combinations],
            "unexpected_combinations": [list(item) for item in unexpected_combinations],
            "duplicate_combinations": [list(item) for item in duplicate_combinations],
            "malformed_line_count": len(malformed),
            "manifest": manifest,
            "manifest_error": manifest_error,
            "malformed_lines": malformed,
        })
        malformed_total += len(malformed)
        all_rows.extend(rows)
    check_rows, runtime_rows, paired_rows, confusion_rows, macro_rows = build_summaries(
        all_rows)
    phases = sorted({str(row.get("phase", "unknown")) for row in all_rows})
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "check-metrics.csv", check_rows)
    write_csv(output_dir / "runtime-metrics.csv", runtime_rows)
    write_csv(output_dir / "paired-seed-differences.csv", paired_rows)
    write_csv(output_dir / "confusion-matrices.csv", confusion_rows)
    write_csv(output_dir / "macro-recall.csv", macro_rows)
    summary = {
        "phases": phases,
        "analysis_revision": git_value("rev-parse", "HEAD"),
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analysis_python_version": platform.python_version(),
        "analysis_workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "source_files": sources,
        "record_count": len(all_rows),
        "malformed_line_count": malformed_total,
        "check_metrics": check_rows,
        "runtime_metrics": runtime_rows,
        "paired_seed_differences": paired_rows,
        "confusion_matrices": confusion_rows,
        "macro_recall": macro_rows,
        "interpretation": "Descriptive only. Candidate checks and thresholds remain unvalidated unless separately frozen and reviewed; this script does not update the manuscript.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary,
                                                        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Descriptive study summary",
        "",
        f"Phases: {', '.join(phases)}",
        f"Raw records: {len(all_rows)} across {len(paths)} JSONL file(s); malformed lines: {malformed_total}.",
        "",
        "These summaries do not validate the current detector or authorize confirmatory claims. Candidate check definitions and thresholds must be reviewed against the frozen analysis plan. The script does not insert results into the paper.",
        "",
        "## Candidate check outcomes",
        "",
        "`failure_rate` is the proportion of successful run records in which the named check returned false. For a fault condition it is a candidate recall estimate; for the healthy condition it is a candidate false-positive estimate. Wilson intervals are descriptive and do not account for paired seeds.",
        "",
        "See `check-metrics.csv`, `runtime-metrics.csv`, `confusion-matrices.csv`, `macro-recall.csv`, `paired-seed-differences.csv`, and `summary.json` for counts, missing combinations, failures, and source hashes.",
    ]
    (output_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        f"Wrote descriptive summaries for {len(all_rows)} records to {output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
