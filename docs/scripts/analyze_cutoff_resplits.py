"""Retrospectively refit the Digits loss cutoffs across seed-blocked splits."""

import argparse
import json
import os
from pathlib import Path
import random
from statistics import mean, median, quantiles

from analyze_public_workload import write_csv
from analyze_schedule_study import HEALTHY, digest
from analyze_score_separation import BUDGETS, MATCHED, grouped, loss_values, schedule_cutoffs, verified_bundle

RESPLITS = 2000
RANDOM_SEED = 24583
CALIBRATION_SIZE = 40


def refit(healthy, fault, calibration_seeds, target):
    """Use the same healthy seed block for all schedules, then score its complement."""
    fitted = schedule_cutoffs([[healthy[name][seed] for seed in sorted(calibration_seeds)]
                              for name in HEALTHY], target)
    holdout = healthy["plain"].keys() - calibration_seeds
    eligible_fault = fault.keys() & holdout
    result = {}
    for name in HEALTHY:
        alarms = sum(healthy[name][seed] > fitted[name] for seed in holdout)
        result[name] = dict(cutoff=fitted[name], alarms=alarms, n=len(holdout))
    alarms = sum(fault[seed] > fitted["plain"] for seed in eligible_fault)
    result["periodic4"] = dict(cutoff=fitted["plain"], alarms=alarms, n=len(eligible_fault))
    return result


def distribution(values):
    endpoints = quantiles(values, n=40, method="inclusive")
    return dict(mean=mean(values), median=median(values), low=endpoints[0], high=endpoints[-1])


def analyze(output):
    inputs = {}
    dev_manifest, dev_data = verified_bundle(MATCHED / "development/schedule-digits", inputs)
    ev_manifest, ev_data = verified_bundle(MATCHED / "evaluation/schedule-digits", inputs)
    for key in ("source_revision", "runner_boot_sha256", "cpu_model", "cpu_affinity", "dataset_sha256"):
        if dev_manifest[key] != ev_manifest[key]:
            raise ValueError(f"Matched archive differs in {key}")
    dev, ev = grouped(dev_data), grouped(ev_data)
    target = ev_manifest["design"]["population_target"]
    dev_seeds = dev[("wall25", "plain")].keys()
    ev_seeds = ev[("wall25", "plain")].keys()
    if len(dev_seeds) != CALIBRATION_SIZE or len(ev_seeds) != 80 or dev_seeds & ev_seeds:
        raise ValueError("Expected 40 development and 80 disjoint evaluation seed blocks")
    pool = sorted(dev_seeds | ev_seeds)
    rng = random.Random(RANDOM_SEED)
    splits = [set(rng.sample(pool, CALIBRATION_SIZE)) for _ in range(RESPLITS)]
    rows = []
    for budget in BUDGETS:
        healthy = {}
        for schedule in HEALTHY:
            train_scores = loss_values(dev[(budget, schedule)])
            eval_scores = loss_values(ev[(budget, schedule)])
            if train_scores.keys() != dev_seeds or eval_scores.keys() != ev_seeds:
                raise ValueError("Healthy arms must share each phase's seed block")
            healthy[schedule] = train_scores | eval_scores
        fault = loss_values(ev[(budget, "periodic4")])
        if fault.keys() != ev_seeds:
            raise ValueError("Fault scores must share evaluation seeds")
        for iteration, split in enumerate(splits, 1):
            for condition, counts in refit(healthy, fault, split, target).items():
                rows.append(dict(budget=budget, split=iteration, condition=condition, **counts))

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "splits.csv", rows)
    summary = {}
    for budget in BUDGETS:
        summary[budget] = {}
        for condition in (*HEALTHY, "periodic4"):
            selected = [row for row in rows if row["budget"] == budget and row["condition"] == condition]
            summary[budget][condition] = dict(
                cutoff=distribution([row["cutoff"] for row in selected]),
                rate=distribution([row["alarms"] / row["n"] for row in selected]),
                denominators=[min(row["n"] for row in selected), max(row["n"] for row in selected)],
            )
    metadata = dict(status="retrospective", resplits=RESPLITS, random_seed=RANDOM_SEED,
                    source_revision=os.environ.get("GITHUB_SHA"), run_id=os.environ.get("GITHUB_RUN_ID"),
                    calibration_seeds_per_split=CALIBRATION_SIZE,
                    nominal_alarm_target=target, exchangeable_continuous_rank_expectation=2 / 41,
                    input_sha256=inputs, script_sha256=digest(Path(__file__)), results=summary)
    (output / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args().output)
