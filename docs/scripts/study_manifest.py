#!/usr/bin/env python3
"""Write the run-level manifest for a hosted CPU study sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("exploratory",
                        "confirmatory"), required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--faults", required=True)
    parser.add_argument("--budgets", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--run-id", default="unavailable")
    parser.add_argument("--run-attempt", default="1")
    parser.add_argument("--run-url", default="unavailable")
    args = parser.parse_args()
    if not args.records.is_file():
        print(f"Run record file not found: {args.records}", file=sys.stderr)
        return 1
    parsed = []
    malformed_line_count = 0
    for line in args.records.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed_line_count += 1
            continue
        if isinstance(row, dict):
            parsed.append(row)
        else:
            malformed_line_count += 1
    manifest = {
        "phase": args.phase,
        "source_revision": git_value("rev-parse", "HEAD"),
        "tracked_source_changes": bool(git_value("status", "--porcelain", "--untracked-files=no")),
        "workflow_run_id": args.run_id,
        "workflow_run_attempt": args.run_attempt,
        "workflow_run_url": args.run_url,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "requested_seeds": [int(value.strip()) for value in args.seeds.split(",") if value.strip()],
        "requested_faults": [value.strip() for value in args.faults.split(",") if value.strip()],
        "requested_budgets_seconds": [float(value.strip()) for value in args.budgets.split(",") if value.strip()],
        "requested_max_steps": args.max_steps,
        "record_count": len(parsed),
        "malformed_line_count": malformed_line_count,
        "success_count": sum(row.get("status") == "ok" for row in parsed),
        "failure_count": sum(row.get("status") != "ok" for row in parsed),
        "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
