"""Validate optimizer-scheduling artifacts and apply the prespecified analysis."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
from pathlib import Path
from statistics import median

from analyze_public_workload import apply_holm, mcnemar, wilson, write_csv

HEALTHY = ("plain", "accumulate2", "declared_skip4")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration(groups, target):
    if not 0 < target < 1:
        raise ValueError("Calibration target must be between zero and one")
    cutoffs = []
    for values in groups:
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("Calibration requires finite development scores")
        rank = math.ceil((len(values) + 1) * (1 - target))
        if rank > len(values):
            raise ValueError("Too few development samples for the calibration target")
        cutoffs.append(sorted(values)[rank - 1])
    if not cutoffs:
        raise ValueError("Calibration requires at least one healthy schedule")
    return max(cutoffs)


def decisions(row, cutoff):
    if row["status"] == "failed":
        return {p: None for p in ("fixed_loss", "calibrated", "naive", "declared")}
    return dict(fixed_loss=row["loss_score"] >= 0.98, calibrated=row["loss_score"] > cutoff,
                naive=row["steps"] != row["actual_calls"], declared=row["mismatch_slots"] > 0)


def quality_summary(pairs, healthy):
    """Score actual omissions separately from legitimate schedule quality differences."""
    deltas = [delta for delta, _, _ in pairs]
    actual = [(delta, alarm) for delta, alarm, omitted in pairs if omitted]
    result = {
        "paired_quality": len(deltas),
        "median_accuracy_drop": median(delta for delta, _ in actual) if actual and not healthy else None,
        "median_legitimate_accuracy_drop": median(deltas) if deltas and healthy else None,
        "quality_pairs_with_omission": len(actual) if not healthy else None,
    }
    for points in (2, 5, 10):
        result[f"miss_loss_{points}pp"] = (
            sum(not alarm and delta >= points / 100 for delta, alarm in actual)
            if not healthy and pairs else None
        )
    return result


def load_records(records):
    rows = []
    for lineno, line in enumerate(records.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{records}:{lineno}: invalid JSON") from error
        if not isinstance(row, dict) or {"seed", "condition", "budget", "position", "status"} - row.keys():
            raise ValueError(f"{records}:{lineno}: invalid record structure")
        rows.append(row)
    return rows


def analyze(input_dir, output, config_path, phase):
    design = json.loads(config_path.read_bytes())
    evaluate = phase == "evaluation"
    study = {}
    thresholds = {}
    counts, pairs = [], []
    manifests = []
    lo, hi = design["evaluation_seeds" if evaluate else "development_seeds"]
    for workload in design["workloads"]:
        path = input_dir / f"schedule-{workload}"
        manifest = json.loads((path / "manifest.json").read_bytes())
        records = path / "records.jsonl"
        if (manifest["records_sha256"] != digest(records) or manifest["design"] != design
                or manifest["config_sha256"] != digest(config_path) or manifest["phase"] != phase
                or manifest["workload"] != workload
                or (os.getenv("GITHUB_SHA") and manifest["source_revision"] != os.environ["GITHUB_SHA"])):
            raise ValueError(f"{workload}: invalid manifest")
        rows = load_records(records)
        condition_names = list(HEALTHY)
        if evaluate:
            condition_names += ["full", *(f"{p}{d}" for p in design["patterns"] for d in design["denominators"])]
        grid = list(itertools.product(condition_names, design["budgets"]))
        expected = []
        for seed in range(lo, hi + 1):
            order = grid.copy()
            random.Random(seed ^ 0x5CA1).shuffle(order)
            expected.extend((seed, c, b, i) for i, (c, b) in enumerate(order))
        if ([(r["seed"],r["condition"],r["budget"],r["position"]) for r in rows] != expected
                or len(rows) != manifest["record_count"]
                or sum(r["status"] == "failed" for r in rows) != manifest["failures"]):
            raise ValueError(f"{workload}: unexpected run grid or failure count")
        for r in rows:
            if r["status"] == "failed":
                continue
            if (r["status"] != "ok" or not math.isfinite(r["loss_score"])
                    or not 0 < r["steps"]
                    or not 0 <= r["actual_calls"] <= r["planned_calls"] <= r["steps"]
                    or r["mismatch_slots"] != r["injected_omissions"]
                    or r["planned_calls"] - r["actual_calls"] != r["injected_omissions"]):
                raise ValueError(f"{workload}: invalid trace")
            if (r["heldout_accuracy"] is None) == evaluate:
                raise ValueError(f"{workload}: unexpected semantic test data")
            if evaluate and not 0 <= r["heldout_accuracy"] <= 1:
                raise ValueError(f"{workload}: invalid held-out accuracy")
            if not 0 < r["loop_wall_seconds"] <= r["diagnose_wall_seconds"]:
                raise ValueError(f"{workload}: invalid timing scope")
            steps = r["steps"]
            expected_calls = {
                "accumulate2": steps // 2,
                "declared_skip4": steps - (steps + 3) // 4,
            }.get(r["condition"], steps)
            if r["planned_calls"] != expected_calls:
                raise ValueError(f"{workload}: declared schedule calls do not match the trace")
            if r["condition"] in HEALTHY and r["injected_omissions"]:
                raise ValueError(f"{workload}: omission injected into a healthy run")
            if r["condition"].startswith("periodic") or r["condition"] == "full":
                denominator = 1 if r["condition"] == "full" else int(r["condition"][8:])
                if r["injected_omissions"] != (steps + denominator - 1) // denominator:
                    raise ValueError(f"{workload}: incorrect periodic omission exposure")
            if r["budget"] == "fixed64" and r["steps"] != design["budgets"]["fixed64"]["steps"]:
                raise ValueError(f"{workload}: incorrect fixed exposure")
        manifests.append(manifest)
        study[workload] = {
            (c, b): sorted([r for r in rows if (r["condition"], r["budget"]) == (c, b)], key=lambda r: r["seed"])
            for c, b in grid
        }
        thresholds[workload] = {}
        for budget in design["budgets"]:
            key = study[workload]
            if evaluate:
                cutoff = design["thresholds"][workload][budget]
            else:
                groups = [key[(schedule, budget)] for schedule in HEALTHY]
                if any(r["status"] != "ok" for group in groups for r in group):
                    raise ValueError("Calibration requires all planned development scores")
                cutoff = calibration([[r["loss_score"] for r in group] for group in groups], design["population_target"])
            thresholds[workload][budget] = cutoff
            healthy = key[("plain", budget)]
            for condition in condition_names:
                group = key[(condition, budget)]
                good = [r for r in group if r["status"] == "ok"]
                metric_keys = ("steps","actual_calls","diagnose_wall_seconds","diagnose_cpu_seconds","loop_wall_seconds")
                metrics = {f"median_{name}": median(r[name] for r in good) if good else None for name in metric_keys}
                for policy in decisions(group[0], cutoff):
                    alarm = [decisions(r, cutoff)[policy] for r in group]
                    observed = [a for a in alarm if a is not None]
                    total = sum(observed)
                    interval = wilson(total, len(observed))
                    had_omission = [(r, a) for r, a in zip(group, alarm, strict=True) if a is not None and r["injected_omissions"]]
                    quality = [(ref["heldout_accuracy"] - r["heldout_accuracy"], a, r["injected_omissions"] > 0)
                               for ref,r,a in zip(healthy,group,alarm,strict=True)
                               if evaluate and a is not None and ref["status"] == "ok"]
                    rates = [r["injected_omissions"]/r["planned_calls"] for r in good if r["planned_calls"]]
                    counts.append(dict(workload=workload,budget=budget,condition=condition,policy=policy,
                                       attempts=len(group), failures=len(group)-len(good), alarms=total,
                                       low=interval[0],high=interval[1],
                                       conservative=(total + (len(group)-len(good) if condition in HEALTHY else 0))/len(group),
                                       actual_omissions=len(had_omission), alarm_with_omission=sum(a for _,a in had_omission),
                                       median_omission_rate=median(rates) if rates else None,
                                       **quality_summary(quality, condition in HEALTHY), **metrics))
                if evaluate and condition in [f"{p}{design['primary_denominator']}" for p in design["patterns"]]:
                    pairs.append(dict(workload=workload,budget=budget,condition=condition,**mcnemar(
                        [decisions(r,cutoff)["calibrated"] for r in group],
                        [decisions(r,cutoff)["calibrated"] for r in healthy])))
    if len({(m["source_revision"], m["workflow_run_id"], m["workflow_attempt"]) for m in manifests}) != 1:
        raise ValueError("Different executions were mixed")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "outcomes.csv", counts)
    if pairs:
        apply_holm(pairs)
        write_csv(output / "paired-primary.csv", pairs)
    (output / "summary.json").write_text(json.dumps(dict(
        phase=phase,thresholds=thresholds,manifests=manifests,analysis_sha256=digest(Path(__file__)),
        paired=pairs),indent=2)+"\n")
    if any(row["attempts"] == row["failures"] for row in counts):
        raise ValueError("A condition has no evaluable attempts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--phase", choices=("development","evaluation"),required=True)
    args=parser.parse_args()
    analyze(args.input,args.output,args.design,args.phase)
