"""Archive the pinned CPU scaler mechanism probe and validate its trace."""

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

from traincheck.scaler_probe import CASES, probe
from analyze_schedule_study import digest


def run(design_path, root):
    design = json.loads(design_path.read_bytes())
    if design["status"] != "frozen":
        raise ValueError("Expected frozen probe configuration")
    torch.set_num_threads(1)
    first, last = design["mechanism_seeds"]
    rows = []
    for seed in range(first, last + 1):
        for case in CASES:
            started = time.perf_counter()
            try:
                trace = probe(seed, case)
                rows.append(dict(seed=seed, case=case, status="ok", trace=trace,
                                 elapsed_seconds=time.perf_counter() - started))
            except Exception as error:
                rows.append(dict(seed=seed, case=case, status="failed",
                                 error=f"{type(error).__name__}: {error}",
                                 elapsed_seconds=time.perf_counter() - started))
    records = root / "scaler-records.jsonl"
    records.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    summary = {}
    for case in CASES:
        selected = [row for row in rows if row["case"] == case]
        good = [row["trace"] for row in selected if row["status"] == "ok"]
        summary[case] = dict(attempts=len(selected), failed=len(selected)-len(good))
        for label, key in (("naive_alarm", "naive_alarm"), ("optimizer_alarm", "optimizer_mismatch"),
                           ("scheduler_alarm", "scheduler_mismatch"), ("warned", "scheduler_warnings")):
            summary[case][label] = sum(any(row[key] for row in trace) for trace in good)
        summary[case]["initial_lr_changed"] = sum(trace[0]["lr_before"] != trace[0]["lr_after"] for trace in good)
    (root / "scaler-summary.json").write_text(json.dumps(dict(
        cases=summary, raw_sha256=digest(records), configuration_sha256=digest(design_path),
        run_id=os.getenv("GITHUB_RUN_ID"), source=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python=platform.python_version(), torch=str(torch.__version__), host=platform.platform(),
    ), indent=2) + "\n")
    if any(v["failed"] for v in summary.values()):
        raise ValueError("One or more mechanism probes failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.design, args.root)
