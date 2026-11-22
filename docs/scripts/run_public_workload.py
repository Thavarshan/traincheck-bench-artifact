#!/usr/bin/env python3
"""Record all attempts in a fixed, paired CPU design."""

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
import time
from pathlib import Path

import torch

from traincheck.heldout_workload import load_workload, measure

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "docs/research/PUBLIC_WORKLOAD_STUDY.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_numbers(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_numbers(item) for key, item in value.items()}
    return value


def run(workload_name: str, phase: str, data_dir: Path, output: Path) -> None:
    design = json.loads(DESIGN.read_bytes())
    if phase == "confirmatory" and design["status"] != "frozen":
        raise ValueError("Confirmatory runs require a committed frozen public workload design")
    if phase == "confirmatory":
        for relative, expected in design["frozen_files"].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Frozen input changed: {relative}")
    if workload_name not in design["workloads"]:
        raise ValueError(f"Unknown workload: {workload_name}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT))
    if phase == "confirmatory" and dirty:
        raise ValueError("Confirmatory runs require a clean tracked working tree")

    seed_start, seed_stop = design[f"{phase}_seed_range"]
    if phase == "confirmatory" and seed_start <= design["development_seed_range"][1]:
        raise ValueError("Development and confirmatory seed ranges must be disjoint")
    cells = list(itertools.product(design["faults"], design["budgets_seconds"]))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    workload = load_workload(workload_name, data_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = output / "records.jsonl"
    failures = 0
    started = time.perf_counter()
    with records.open("x", encoding="utf-8") as stream:
        for seed in range(seed_start, seed_stop + 1):
            order = cells.copy()
            random.Random(design["order_seed_offset"] + seed).shuffle(order)
            for position, (fault, budget) in enumerate(order):
                row = {
                    "workload": workload_name, "phase": phase, "source_revision": revision,
                    "seed": seed, "fault": fault, "budget_seconds": budget, "position": position,
                    "configuration": {"max_steps": design["max_steps"], "torch_threads": 1},
                    "python_version": platform.python_version(), "torch_version": str(torch.__version__),
                }
                attempt_started = time.perf_counter()
                try:
                    metrics = measure(workload, seed, fault, budget, design["max_steps"])
                    row.update(metrics)
                    row["status"] = "ok"
                except Exception as error:
                    failures += 1
                    row.update(status="failed", error_type=type(error).__name__, error=str(error))
                row["attempt_seconds"] = time.perf_counter() - attempt_started
                stream.write(json.dumps(clean_numbers(row), allow_nan=False) + "\n")
                stream.flush()
    cpu_info = Path("/proc/cpuinfo")
    cpu_model = None
    if cpu_info.is_file():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
                          if line.startswith("model name")), None)
    manifest = {
        "workload": workload_name, "phase": phase, "source_revision": revision,
        "design": design, "design_sha256": sha256(DESIGN), "records_sha256": sha256(records),
        "data_sha256": {path.name: sha256(path) for path in sorted(data_dir.iterdir()) if path.is_file()},
        "record_count": (seed_stop - seed_start + 1) * len(cells), "failed_runs": failures,
        "sweep_wall_seconds": time.perf_counter() - started, "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(), "host_platform": platform.platform(), "tracked_source_changes": dirty,
        "python_version": platform.python_version(), "torch_version": str(torch.__version__),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"), "workflow_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--phase", choices=("development", "confirmatory"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.workload, arguments.phase, arguments.data_dir, arguments.output)
