"""Run frozen count-preserving faults and an exploratory Wine endpoint replay."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from statistics import median, quantiles

import torch

from analyze_public_workload import wilson
from analyze_schedule_study import calibration
from run_update_control import seeds
from traincheck.schedule_study import load_schedule_workload, measure_schedule

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "docs/research/GRADIENT_FAULT_STUDY.json"
SOURCES = (
    DESIGN,
    ROOT / "docs/research/UPDATE_CONTROL.json",
    ROOT / "docs/scripts/analyze_schedule_study.py",
    ROOT / "docs/scripts/analyze_public_workload.py",
    ROOT / "docs/scripts/run_update_control.py",
    ROOT / "traincheck/schedule_study.py",
    ROOT / "traincheck/heldout_workload.py",
    ROOT / "traincheck/heldout_datasets.py",
    ROOT / "requirements/study-cpu.txt",
    ROOT / ".github/workflows/gradient-faults.yml",
    Path(__file__).resolve(),
)
HEALTHY = ("plain", "accumulate2")
METRICS = ("loss_score", "heldout_loss", "heldout_accuracy")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def collect(path, workload, identities, conditions, steps):
    rows = []
    with path.open("w") as stream:
        for seed in identities:
            order = list(conditions)
            random.Random(seed ^ 0xE042).shuffle(order)
            for position, name in enumerate(order):
                spec = conditions[name]
                row = dict(seed=seed, position=position, condition=name, schedule=spec["schedule"])
                start = time.perf_counter()
                try:
                    budget = {"steps": spec.get("steps", steps)}
                    result = measure_schedule(workload, seed, spec, budget, include_heldout_loss=True)
                    calls = spec.get("calls", budget["steps"] // (2 if spec["schedule"] == "accumulate2" else 1))
                    if (result["steps"], result["actual_calls"]) != (budget["steps"], calls):
                        raise ValueError("Unexpected completed minibatch or optimizer count")
                    nonfinite = [key for key in METRICS if result[key] is not None and not math.isfinite(result[key])]
                    for key in nonfinite:
                        result[key] = None
                    if "heldout_loss" in nonfinite:
                        result["heldout_accuracy"] = None
                    row.update(result, status="ok", nonfinite=nonfinite)
                except Exception as error:
                    row.update(status="failed", error=f"{type(error).__name__}: {error}")
                row["elapsed_seconds"] = time.perf_counter() - start
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
    if any(row["status"] != "ok" for row in rows):
        raise ValueError(f"Failed attempts retained in {path}")
    return rows


def host(data):
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    return dict(source_revision=os.environ.get("GITHUB_SHA"), workflow_run_id=os.environ.get("GITHUB_RUN_ID"),
                workflow_attempt=os.environ.get("GITHUB_RUN_ATTEMPT"), cpu_model=cpu,
                affinity=sorted(os.sched_getaffinity(0)), boot=digest(Path("/proc/sys/kernel/random/boot_id")),
                platform=platform.platform(), python=platform.python_version(), torch=str(torch.__version__),
                dataset_sha256={file.name: digest(file) for file in sorted(data.iterdir()) if file.is_file()})


def paired_differences(left, right, metric, design):
    if left.keys() != right.keys():
        raise ValueError("Paired seed identities differ")
    deltas = [left[seed][metric] - right[seed][metric] for seed in sorted(left)
              if left[seed][metric] is not None and right[seed][metric] is not None]
    if not deltas:
        return dict(n=0, median=None, low=None, high=None)
    rng = random.Random(design["bootstrap_seed"])
    resamples = (median(rng.choices(deltas, k=len(deltas))) for _ in range(design["bootstrap_samples"]))
    endpoints = quantiles(resamples, n=40, method="inclusive")
    return dict(n=len(deltas), median=median(deltas), low=endpoints[0], high=endpoints[-1])


def grouped(rows, conditions):
    return {name: {row["seed"]: row for row in rows if row["condition"] == name} for name in conditions}


def evaluate(args, design, conditions):
    freeze = json.loads((args.out / "freeze.json").read_bytes())
    development = args.out / "development.jsonl"
    current = host(args.data)
    # Fashion development deliberately avoids test files, which arrive at evaluation.
    input_data = {name: current["dataset_sha256"][name] for name in freeze["host"]["dataset_sha256"]}
    current["dataset_sha256"] = input_data
    if current != freeze["host"] or digest(development) != freeze["development_sha256"]:
        raise ValueError("Evaluation host or development archive differs from freeze")
    if any(digest(ROOT / path) != expected for path, expected in freeze["source_sha256"].items()):
        raise ValueError("Study source differs from freeze")

    records = args.out / "evaluation.jsonl"
    workload = load_schedule_workload(args.workload, args.data, evaluate=True)
    rows = collect(records, workload, seeds(design["evaluation_seeds"]), conditions, design["steps"])
    groups = grouped(rows, conditions)
    outcomes = {}
    for name, group in groups.items():
        values = list(group.values())
        cutoff = freeze["cutoffs"][conditions[name]["schedule"]]
        nonfinite = sum(row["loss_score"] is None for row in values)
        alarms = nonfinite + sum(row["loss_score"] is not None and row["loss_score"] > cutoff for row in values)
        outcomes[name] = dict(n=len(values), loss_alarms=alarms, loss_wilson=wilson(alarms, len(values)),
                              nonfinite_scores=nonfinite, call_alarms=sum(row["mismatch_slots"] > 0 for row in values))
    contrasts = {
        name: {metric: paired_differences(groups[healthy], groups[name], metric, design)
               for metric in ("heldout_accuracy", "heldout_loss")}
        for name, healthy in (("missing_reset", "plain"), ("unscaled", "accumulate2"))
    }
    summary = dict(host=host(args.data), records_sha256=digest(records), outcomes=outcomes, contrasts=contrasts)
    if args.workload == "wine":
        replay = json.loads((ROOT / design["wine_control"]).read_bytes())
        records = args.out / "exploratory-wine-replay.jsonl"
        conditions = replay["conditions"]
        replay_rows = collect(records, workload, seeds(replay["control_seeds"]), conditions, design["steps"])
        replay_groups = grouped(replay_rows, conditions)
        summary["exploratory_replay"] = dict(
            records_sha256=digest(records),
            median_loss={name: median(row["heldout_loss"] for row in group.values())
                         for name, group in replay_groups.items()},
            paired_loss={control: paired_differences(replay_groups[control], replay_groups["quarter64"],
                                                     "heldout_loss", design)
                         for control in ("plain64", "plain48")},
        )
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    parser.add_argument("--workload", choices=("digits", "wine", "fashion"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(DESIGN.read_bytes())
    if design["status"] != "frozen" or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("Expected a frozen protocol on one pinned logical CPU")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    conditions = design["conditions"]
    if args.phase == "evaluation":
        evaluate(args, design, conditions)
        return

    workload = load_schedule_workload(args.workload, args.data, evaluate=False)
    records = args.out / "development.jsonl"
    rows = collect(records, workload, seeds(design["development_seeds"]),
                   {name: conditions[name] for name in HEALTHY}, design["steps"])
    cutoffs = {name: calibration([[row["loss_score"] for row in rows if row["condition"] == name]],
                                 design["population_target"]) for name in HEALTHY}
    write_json(args.out / "freeze.json", dict(host=host(args.data), workload=args.workload, cutoffs=cutoffs,
                                             development_sha256=digest(records),
                                             source_sha256={str(path.relative_to(ROOT)): digest(path)
                                                            for path in SOURCES}))


if __name__ == "__main__":
    main()
