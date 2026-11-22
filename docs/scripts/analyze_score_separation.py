"""Retrospective score separation and operating points from frozen archives."""

import argparse
from collections import Counter
from itertools import groupby
import json
from math import isfinite
import os
from pathlib import Path
import random
from statistics import median, quantiles

from analyze_public_workload import summary_stats, wilson, write_csv
from analyze_schedule_study import HEALTHY, calibration, digest

ROOT = Path(__file__).resolve().parents[2]
SCHEDULE = ROOT / "results/schedule-study"
MATCHED = ROOT / "results/matched-calibration/45057254348"
ORIGINAL = ROOT / "results/public-study/confirmatory/45018868091"
BUDGETS = ("wall25", "fixed64")
WORKLOADS = ("digits", "wine", "fashion")
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 821453


def rank_auc(pairs):
    """Estimate P(fault score > healthy score), with half credit for ties."""
    items = sorted((score, label) for pair in pairs for label, score in enumerate(pair))
    healthy_below = 0
    wins = 0.0
    for _, block in groupby(items, key=lambda item: item[0]):
        counts = Counter(label for _, label in block)
        wins += counts[1] * (healthy_below + 0.5 * counts[0])
        healthy_below += counts[0]
    return wins / len(pairs)**2


def paired_auc(pairs):
    """Resample complete seed rows, preserving the healthy/fault dependence."""
    rng = random.Random(BOOTSTRAP_SEED)
    samples = (
        rank_auc(rng.choices(pairs, k=len(pairs)))
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    percentiles = quantiles(samples, n=40, method="inclusive")
    return rank_auc(pairs), percentiles[0], percentiles[-1]


def verified_bundle(directory, inputs):
    manifest_path = directory / "manifest.json"
    records = directory / "records.jsonl"
    manifest = json.loads(manifest_path.read_bytes())
    for path in (manifest_path, records):
        inputs[str(path.relative_to(ROOT))] = digest(path)
    if inputs[str(records.relative_to(ROOT))] != manifest["records_sha256"]:
        raise ValueError(f"Archived records differ from manifest: {records}")
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    if len(rows) != manifest["record_count"]:
        raise ValueError(f"Archive record count differs from manifest: {records}")
    return manifest, rows


def grouped(rows, condition_key="condition", budget_key="budget"):
    result = {}
    for row in rows:
        key = (row[budget_key], row[condition_key])
        cell = result.setdefault(key, {})
        if row["seed"] in cell or row["status"] != "ok":
            raise ValueError(f"Expected complete, successful, distinct archived seeds: {key}")
        cell[row["seed"]] = row
    return result


def loss_values(cell):
    scores = {seed: row["loss_score"] for seed, row in cell.items()}
    if not scores or not all(map(isfinite, scores.values())):
        raise ValueError("All loss scores must be finite")
    return scores


def timing_row(study, phase, schedule, manifest, rows, calls_key):
    return dict(study=study, phase=phase, schedule=schedule, cpu=manifest["cpu_model"],
                study_revision=manifest["source_revision"], count=len(rows),
                median_minibatches=median(row["steps"] for row in rows.values()),
                median_optimizer_calls=median(row[calls_key] for row in rows.values()))


def schedule_cutoffs(development, target):
    return {schedule: calibration([scores], target)
            for schedule, scores in zip(HEALTHY, development, strict=True)}


def schedule_operating_point(cell, cutoff):
    scores = loss_values(cell)
    alarms = sum(score > cutoff for score in scores.values())
    low, high = wilson(alarms, len(scores))
    realized = [row for row in cell.values() if row["injected_omissions"]]
    return dict(n=len(scores), alarms=alarms, low=low, high=high, realized=len(realized),
                realized_alarms=sum(row["loss_score"] > cutoff for row in realized))


def analyze(output):
    inputs = {}
    auc_rows, distributions, operating, severity, timing = [], [], [], [], []
    plotted_scores = []
    schedule_operating = []
    original_manifest, original_rows = verified_bundle(ORIGINAL / "public-workload-digits", inputs)
    timing.append(timing_row("original", "evaluation", "plain", original_manifest,
                             grouped(original_rows, "fault", "budget_seconds")[(0.025, "healthy")],
                             "optimizer_step_calls"))
    for study, workloads, development, evaluation in (
        ("cross_host", WORKLOADS, SCHEDULE / "development/45045512022",
         SCHEDULE / "evaluation/45045778246"),
        ("matched_runner", ("digits",), MATCHED / "development", MATCHED / "evaluation"),
    ):
        for workload in workloads:
            dev_manifest, dev_rows = verified_bundle(development / f"schedule-{workload}", inputs)
            ev_manifest, ev_rows = verified_bundle(evaluation / f"schedule-{workload}", inputs)
            dev, ev = grouped(dev_rows), grouped(ev_rows)
            if study == "cross_host" and dev_manifest["design"]["status"] != "development":
                raise ValueError("Expected a development-only calibration archive")
            for budget in BUDGETS:
                groups = [list(loss_values(dev[(budget, name)]).values()) for name in HEALTHY]
                target = ev_manifest["design"]["population_target"]
                frozen = ev_manifest["design"]["thresholds"][workload][budget]
                if calibration(groups, target) != frozen or any(len(group) != 40 for group in groups):
                    raise ValueError("Archive does not reproduce the frozen three-schedule cutoff")
                restricted = calibration(groups[:2], target)
                conditional = schedule_cutoffs(groups, target)
                context = dict(study=study, workload=workload, budget=budget)
                fault = loss_values(ev[(budget, "periodic4")])
                if len(fault) != 80 or not all(row["injected_omissions"] for row in ev[(budget, "periodic4")].values()):
                    raise ValueError("Expected 80 realized periodic-omission runs")
                for condition, spec in ev_manifest["conditions"].items():
                    schedule = spec["schedule"]
                    cell = ev[(budget, condition)]
                    if cell.keys() != fault.keys():
                        raise ValueError("Expected the same evaluation seeds in every condition")
                    for policy, cutoff in (("shared", frozen), ("per_schedule", conditional[schedule])):
                        schedule_operating.append(dict(**context, condition=condition, declared_schedule=schedule,
                                                       policy=policy, cutoff=cutoff,
                                                       **schedule_operating_point(cell, cutoff)))
                for condition in (*HEALTHY, "periodic4"):
                    scores = loss_values(ev[(budget, condition)])
                    if scores.keys() != fault.keys() or scores.keys() & dev[(budget, "plain")].keys():
                        raise ValueError("Expected paired evaluation seeds disjoint from development")
                    summary, iqr = summary_stats(ev[(budget, condition)].values(), "loss_score")
                    distributions.append(dict(**context, condition=condition, n=len(scores),
                                              median=summary, iqr=iqr, minimum=min(scores.values()),
                                              maximum=max(scores.values())))
                    if condition in HEALTHY:
                        paired = [(scores[seed], fault[seed]) for seed in sorted(fault)]
                        auc, low, high = paired_auc(paired)
                        ties = sum(negative == positive for negative, positive in paired)
                        if condition == "declared_skip4" and budget == "fixed64" and ties != len(paired):
                            raise ValueError("Fixed-exposure declared skips differ from periodic omissions")
                        auc_rows.append(dict(**context, healthy=condition, fault="periodic4", pairs=len(paired),
                                             auc=auc, low=low, high=high, identical_seed_scores=ties))
                    for policy, cutoff in (("frozen", frozen), ("plain_accumulation", restricted)):
                        alarms = sum(score > cutoff for score in scores.values())
                        low, high = wilson(alarms, len(scores))
                        operating.append(dict(**context, condition=condition, policy=policy, cutoff=cutoff,
                                              n=len(scores), alarms=alarms, low=low, high=high))
                    if study == "matched_runner":
                        plotted_scores.extend(dict(**context, condition=condition, seed=seed, loss_score=score,
                                                  cross_host_cutoff=ev_manifest["design"]["previous_thresholds"][budget],
                                                  matched_cutoff=frozen, plain_cutoff=conditional["plain"])
                                              for seed, score in sorted(scores.items()))
                if study == "cross_host":
                    plain = ev[(budget, "plain")]
                    for (exposure, condition), group in ev.items():
                        if exposure != budget or condition in HEALTHY:
                            continue
                        actual = [row for row in group.values() if row["injected_omissions"]]
                        severity.append(dict(**context, condition=condition, assigned=len(group),
                                             realized=len(actual),
                                             median_omission=median(row["injected_omissions"] / row["planned_calls"]
                                                                    for row in actual),
                                             alarms=sum(row["loss_score"] > frozen for row in actual),
                                             median_accuracy_drop=median(plain[row["seed"]]["heldout_accuracy"]
                                                                         - row["heldout_accuracy"] for row in actual)))
            if workload == "digits":
                for phase, manifest, groups in (("development", dev_manifest, dev), ("evaluation", ev_manifest, ev)):
                    for schedule in HEALTHY[:2]:
                        timing.append(timing_row(study, phase, schedule, manifest, groups[("wall25", schedule)],
                                                 "actual_calls"))
    output.mkdir(parents=True, exist_ok=True)
    tables = {"auc": auc_rows, "loss-distributions": distributions, "operating-points": operating,
              "severity": severity, "exposure": timing, "matched-scores": plotted_scores,
              "per-schedule-operating-points": schedule_operating}
    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", rows)
    metadata = dict(status="retrospective", bootstrap_samples=BOOTSTRAP_SAMPLES, bootstrap_seed=BOOTSTRAP_SEED,
                    source_revision=os.environ.get("GITHUB_SHA"),
                    workflow_run_id=os.environ.get("GITHUB_RUN_ID"),
                    input_sha256=inputs, analysis_sha256=digest(Path(__file__)))
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args().output)
