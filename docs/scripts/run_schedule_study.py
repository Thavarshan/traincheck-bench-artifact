"""Collect and validate one workload of the optimizer scheduling study."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import random
import subprocess
from pathlib import Path

import torch

from traincheck.schedule_study import HEALTHY, load_schedule_workload, measure_schedule

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "docs/research/SCHEDULE_STUDY.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def study_cells(design, evaluation):
    cells = {name: dict(schedule=name, pattern="none", denominator=1) for name in HEALTHY}
    if evaluation:
        cells["full"] = dict(schedule="plain", pattern="periodic", denominator=1)
        for pattern, denominator in itertools.product(design["patterns"], design["denominators"]):
            cells[f"{pattern}{denominator}"] = dict(schedule="plain", pattern=pattern, denominator=denominator)
    return cells


def collect(workload_name, evaluation, data, out, config_path=CONFIG):
    design = json.loads(config_path.read_bytes())
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if evaluation:
        if design["status"] != "frozen" or not design.get("thresholds"):
            raise ValueError("Evaluation requires a frozen configuration")
        for name, sha in design["frozen_files"].items():
            if digest(ROOT / name) != sha:
                raise ValueError(f"Frozen input changed: {name}")
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT)
        if dirty:
            raise ValueError("Evaluation requires a clean tracked tree")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    workload = load_schedule_workload(workload_name, data, evaluation)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = design["evaluation_seeds" if evaluation else "development_seeds"]
    cells = study_cells(design, evaluation)
    budgets = design["budgets"]
    grid = list(itertools.product(cells, budgets))
    path = out / "records.jsonl"
    n = failures = 0
    with path.open("x", encoding="utf-8") as rows:
        for seed in range(lo, hi + 1):
            order = grid.copy()
            random.Random(seed ^ 0x5CA1).shuffle(order)
            for position, (condition, budget) in enumerate(order):
                row = dict(seed=seed, condition=condition, budget=budget, position=position, status="ok")
                try:
                    row.update(measure_schedule(workload, seed, cells[condition], budgets[budget]))
                    if not math.isfinite(row["loss_score"]) or not row["steps"]:
                        raise ValueError("No evaluable loss score")
                except Exception as error:
                    row = dict(seed=seed, condition=condition, budget=budget, position=position,
                               status="failed", error=f"{type(error).__name__}: {error}")
                    failures += 1
                rows.write(json.dumps(row, allow_nan=False) + "\n")
                rows.flush()
                n += 1
    cpu_model = None
    cpu_path = Path("/proc/cpuinfo")
    if cpu_path.is_file():
        cpu_model = next((line.partition(":")[2].strip() for line in cpu_path.read_text().splitlines()
                          if line.startswith("model name")), None)
    manifest = dict(
        design=design, phase="evaluation" if evaluation else "development", workload=workload_name,
        conditions=cells, source_revision=revision, config_sha256=digest(config_path), records_sha256=digest(path),
        dataset_sha256={p.name: digest(p) for p in sorted(data.iterdir()) if p.is_file()},
        record_count=n, failures=failures, workflow_run_id=os.getenv("GITHUB_RUN_ID"),
        workflow_attempt=os.getenv("GITHUB_RUN_ATTEMPT"), python=platform.python_version(),
        torch=str(torch.__version__), platform=platform.platform(), cpu_model=cpu_model,
        threads=1, train=len(workload.train_y), validation=len(workload.validation_y), test=len(workload.test_y),
        runner_boot_sha256=digest(Path("/proc/sys/kernel/random/boot_id"))
        if Path("/proc/sys/kernel/random/boot_id").is_file() else None,
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{workload_name}: {n} attempts, {failures} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=CONFIG)
    args = parser.parse_args()
    collect(args.workload, args.phase == "evaluation", args.data, args.output, args.design)
