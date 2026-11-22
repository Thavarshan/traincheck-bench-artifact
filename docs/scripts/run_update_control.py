"""Collect prospectively specified update and harness pairs on hosted CPUs."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
from statistics import median, quantiles
import time

import torch

from traincheck.heldout_workload import measure
from traincheck.schedule_study import load_schedule_workload, measure_schedule

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "docs/research/UPDATE_CONTROL.json"
SOURCES = (
    DESIGN,
    ROOT / "requirements/study-cpu.txt", Path(__file__).resolve(),
    ROOT / "traincheck/heldout_workload.py", ROOT / "traincheck/schedule_study.py",
    ROOT / "traincheck/heldout_datasets.py", ROOT / ".github/workflows/update-control.yml",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seeds(bounds):
    return range(bounds[0], bounds[1] + 1)


def run_pairs(workload, design):
    controls = design["conditions"]

    def control(case, seed):
        spec = controls[case]
        metrics = measure_schedule(workload, seed, spec, {"steps": spec["steps"]})
        if (metrics["steps"], metrics["actual_calls"]) != (spec["steps"], spec["calls"]):
            raise ValueError(f"Incomplete exposure for {case}")
        if not all(math.isfinite(metrics[key]) for key in ("loss_score", "heldout_accuracy")):
            raise ValueError(f"Nonfinite metric for {case}")
        return metrics

    def timing(case, seed):
        harness, exposure = case
        config = design["harness_exposures"][exposure]
        if harness == "original":
            measured = measure(workload, seed, "healthy", config["seconds"], config["max_steps"])
            calls = measured["optimizer_step_calls"]
            elapsed = measured["elapsed_seconds"]
        else:
            budget = {"steps": config["max_steps"]} if exposure == "fixed64" else {"seconds": config["seconds"]}
            measured = measure_schedule(workload, seed, controls["plain64"], budget)
            calls = measured["actual_calls"]
            elapsed = measured["loop_wall_seconds"]
        steps = measured["steps"]
        if not (steps == calls > 0 and elapsed > 0) or (exposure == "fixed64" and steps != config["max_steps"]):
            raise ValueError(f"Invalid timing measurement for {case}")
        return dict(steps=steps, actual_calls=calls, loop_wall_seconds=elapsed,
                    diagnose_wall_seconds=measured["diagnose_wall_seconds"],
                    heldout_accuracy=measured["heldout_accuracy"])

    def collect(phase, identities, cases, operation):
        for seed in identities:
            order = list(cases)
            random.Random(seed ^ 0xC048).shuffle(order)
            for position, case in enumerate(order):
                name = "/".join(case) if isinstance(case, tuple) else case
                row = dict(phase=phase, condition=name, seed=seed, position=position)
                started = time.perf_counter()
                try:
                    row.update(operation(case, seed), status="ok")
                except Exception as error:
                    row.update(status="failed", error=f"{type(error).__name__}: {error}")
                row["attempt_seconds"] = time.perf_counter() - started
                yield row

    yield from collect("control", seeds(design["control_seeds"]), controls, control)
    if workload.name == "digits":
        timing_cases = [(harness, exposure) for harness in ("original", "schedule")
                        for exposure in design["harness_exposures"]]
        for phase, key in (("warmup", "harness_warmup_seeds"), ("timing", "harness_seeds")):
            yield from collect(phase, seeds(design[key]), timing_cases, timing)


def summarize(rows, design):
    cells = defaultdict(dict)
    for row in rows:
        key = (row["phase"], row["condition"])
        if row["status"] != "ok" or row["seed"] in cells[key]:
            raise ValueError(f"Missing, duplicate or failed paired trial at {key}")
        cells[key][row["seed"]] = row

    def contrast(phase, left, right, key):
        left_rows, right_rows = cells[phase, left], cells[phase, right]
        expected = set(seeds(design["control_seeds" if phase == "control" else "harness_seeds"]))
        if left_rows.keys() != expected or right_rows.keys() != expected:
            raise ValueError(f"Incomplete pair set for {left} and {right}")
        return [left_rows[seed][key] - right_rows[seed][key] for seed in sorted(expected)]

    control = {}
    for left, right in (("plain64", "quarter64"), ("plain64", "plain48"), ("plain48", "quarter64")):
        deltas = contrast("control", left, right, "heldout_accuracy")
        rng = random.Random(design["bootstrap_seed"])
        sampled = (median(rng.choices(deltas, k=len(deltas))) for _ in range(design["bootstrap_resamples"]))
        quantile = quantiles(sampled, n=40, method="inclusive")
        control[f"{left}-{right}"] = dict(n=len(deltas), median_pp=100 * median(deltas),
                                           low_pp=100 * quantile[0], high_pp=100 * quantile[-1])
    per_arm = {arm: median(row["heldout_accuracy"] for row in cells["control", arm].values())
               for arm in design["conditions"]}

    timing = {}
    if any(phase == "timing" for phase, _ in cells):
        for exposure in design["harness_exposures"]:
            left, right = f"original/{exposure}", f"schedule/{exposure}"
            exposure_result = {}
            for metric in ("steps", "actual_calls", "loop_wall_seconds"):
                deltas = contrast("timing", left, right, metric)
                exposure_result[metric] = dict(original=median(row[metric] for row in cells["timing", left].values()),
                                               schedule=median(row[metric] for row in cells["timing", right].values()),
                                               original_minus_schedule=median(deltas))
            timing[exposure] = exposure_result
    return dict(median_heldout_accuracy=per_arm, accuracy_contrasts=control, harness_timing=timing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, choices=("digits", "wine", "fashion"))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    design = json.loads(DESIGN.read_bytes())
    affinity = sorted(os.sched_getaffinity(0))
    if design["status"] != "frozen" or len(affinity) != 1:
        raise ValueError("A frozen design and one pinned logical CPU are required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    workload = load_schedule_workload(args.workload, args.data, evaluate=True)
    args.out.mkdir(parents=True, exist_ok=True)
    records = args.out / "records.jsonl"
    rows = []
    with records.open("w") as stream:
        for row in run_pairs(workload, design):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    manifest = dict(workload=workload.name, record_count=len(rows),
                    failures=sum(row["status"] != "ok" for row in rows),
                    source_revision=os.environ.get("GITHUB_SHA"), workflow_run_id=os.environ.get("GITHUB_RUN_ID"),
                    workflow_attempt=os.environ.get("GITHUB_RUN_ATTEMPT"), cpu_model=cpu, cpu_affinity=affinity,
                    boot_sha256=sha256(Path("/proc/sys/kernel/random/boot_id")),
                    platform=platform.platform(), python=platform.python_version(), torch=str(torch.__version__),
                    dataset_sha256={file.name: sha256(file) for file in sorted(args.data.iterdir()) if file.is_file()},
                    source_sha256={str(file.relative_to(ROOT)): sha256(file) for file in SOURCES},
                    records_sha256=sha256(records))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = summarize(rows, design)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
