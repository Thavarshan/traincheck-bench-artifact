from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .experiment import FAULTS, _configuration, _git_metadata, diagnose


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="traincheck", description="Run controlled CPU training-check experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--fault", choices=FAULTS, default="healthy")
    run.add_argument("--budget-seconds", type=float, default=2.0)
    run.add_argument("--format", choices=("json", "csv"), default="json")
    run.add_argument("--output", type=Path)
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--seeds", default="1,2,3")
    sweep.add_argument("--faults", default=",".join(FAULTS))
    sweep.add_argument("--budget-seconds", type=float, default=2.0)
    sweep.add_argument("--output", type=Path, required=True)
    study = sub.add_parser(
        "study", help="Run a provenance-recorded CPU sweep as JSON Lines")
    study.add_argument("--seeds", required=True,
                       help="Comma-separated integer seeds")
    study.add_argument("--faults", default=",".join(FAULTS))
    study.add_argument("--budgets", default="0.25,1,5",
                       help="Comma-separated per-run wall-clock seconds")
    study.add_argument("--phase", choices=("exploratory",
                       "confirmatory"), default="exploratory")
    study.add_argument("--output", type=Path, required=True)
    study.add_argument("--max-steps", type=int, default=20_000,
                       help="Optimizer-iteration ceiling; must be frozen for confirmatory study")
    args = parser.parse_args(argv)
    if args.command == "run":
        if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
            parser.error("budget-seconds must be a finite positive number")
        rows = [_run_record(args.seed, args.fault, args.budget_seconds)]
        payload = json.dumps(rows[0], indent=2) + \
            "\n" if args.format == "json" else _csv(rows)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload)
        else:
            print(payload, end="")
        return 1 if rows[0]["status"] != "ok" else 0
    if args.command == "sweep":
        seeds = _parse_list(parser, args.seeds, int, "seeds")
        faults = [value.strip()
                  for value in args.faults.split(",") if value.strip()]
        if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
            parser.error("budget-seconds must be a finite positive number")
        if not faults:
            parser.error("at least one fault is required")
        invalid = set(faults) - set(FAULTS)
        if invalid:
            parser.error(f"unknown faults: {', '.join(sorted(invalid))}")
        rows = [
            _run_record(seed, fault, args.budget_seconds)
            for seed in seeds
            for fault in faults
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(_csv(rows))
        return 1 if any(row["status"] != "ok" for row in rows) else 0

    seeds = _parse_list(parser, args.seeds, int, "seeds")
    budgets = _parse_list(parser, args.budgets, float, "budgets")
    if args.max_steps <= 0 or args.max_steps > 100_000:
        parser.error("study max-steps must be from 1 to 100000")
    if len(seeds) > 100 or len(set(seeds)) != len(seeds):
        parser.error("study requires 1 to 100 unique seeds")
    if any(not math.isfinite(value) or value <= 0 or value > 60 for value in budgets):
        parser.error(
            "all budgets must be finite values greater than 0 and at most 60 seconds")
    if len(set(budgets)) != len(budgets):
        parser.error("study budgets must be unique")
    faults = [value.strip()
              for value in args.faults.split(",") if value.strip()]
    invalid = set(faults) - set(FAULTS)
    if invalid:
        parser.error(f"unknown faults: {', '.join(sorted(invalid))}")
    if not faults or len(set(faults)) != len(faults):
        parser.error("study requires a non-empty list of unique faults")
    if len(seeds) * len(faults) * sum(budgets) > 5400:
        parser.error(
            "requested wall-clock budgets exceed the 90-minute study-run limit")
    if args.phase == "confirmatory":
        repository = Path(__file__).resolve().parents[1]
        freeze_checker = repository / "docs/scripts/check_confirmatory_freeze.py"
        if not freeze_checker.is_file():
            parser.error(
                "confirmatory runs require the checked-out research repository")
        check = subprocess.run(
            [
                sys.executable,
                str(freeze_checker),
                "--seeds", args.seeds,
                "--faults", ",".join(faults),
                "--budgets", args.budgets,
                "--max-steps", str(args.max_steps),
            ],
            cwd=repository,
            check=False,
        )
        if check.returncode:
            return check.returncode
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for seed in seeds:
            for fault in faults:
                for budget in budgets:
                    row = _run_record(
                        seed, fault, budget, phase=args.phase, max_steps=args.max_steps
                    )
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    failed += row["status"] != "ok"
    print(
        f"Wrote {len(seeds) * len(faults) * len(budgets)} run records; failures: {failed}.")
    return 1 if failed else 0


def _parse_list(parser: argparse.ArgumentParser, value: str, cast: type, label: str) -> list:
    try:
        parsed = [cast(item.strip())
                  for item in value.split(",") if item.strip()]
    except ValueError:
        parser.error(f"{label} must be comma-separated {cast.__name__} values")
    if not parsed:
        parser.error(f"at least one {label} value is required")
    return parsed


def _run_record(
    seed: int, fault: str, budget: float,
    phase: str = "exploratory", max_steps: int = 120
) -> dict:
    started = time.perf_counter()
    try:
        record = diagnose(seed=seed, fault=fault,
                          budget_seconds=budget, max_steps=max_steps).to_dict()
        record["status"] = "ok"
        record["error_type"] = None
        record["error"] = None
    except Exception as error:
        revision, dirty = _git_metadata()
        record = {
            "seed": seed,
            "fault": fault,
            "budget_seconds": budget,
            "elapsed_seconds": time.perf_counter() - started,
            "diagnose_wall_seconds": None,
            "budget_overshoot_seconds": None,
            "steps": 0,
            "optimizer_step_calls": None,
            "termination_reason": "exception",
            "over_budget": None,
            "initial_loss": None,
            "final_loss": None,
            "serving_loss": None,
            "preprocessing_max_abs_delta": None,
            "gradient_nonzero": None,
            "weights_updated": None,
            "loss_decreased": None,
            "overfit_small_batch": None,
            "detected": None,
            "detections": {},
            "torch_version": None,
            "device": "cpu",
            "status": "failed",
            "runtime_scope": "failed_before_or_during_training_loop",
            "python_version": sys.version.split()[0],
            "source_revision": revision,
            "working_tree_dirty": dirty,
            "host_platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "configuration": _configuration(max_steps),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    record["phase"] = phase
    return record


def _csv(rows: list[dict]) -> str:
    import io
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(
        {
            key: json.dumps(value, sort_keys=True) if isinstance(
                value, (dict, list)) else value
            for key, value in row.items()
        }
        for row in rows
    )
    return out.getvalue()


if __name__ == "__main__":
    sys.exit(main())
