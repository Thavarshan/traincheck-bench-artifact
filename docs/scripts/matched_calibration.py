"""Freeze same-runner cutoffs and compare them on identical evaluation traces."""

import argparse
import json
import os
from pathlib import Path

from analyze_schedule_study import HEALTHY, digest, load_records
from analyze_public_workload import apply_holm, mcnemar, wilson, write_csv

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "docs/research/MATCHED_STUDY.json",
    "docs/scripts/run_schedule_study.py", "docs/scripts/analyze_schedule_study.py",
    "docs/scripts/analyze_public_workload.py", "docs/scripts/matched_calibration.py",
    "docs/scripts/run_scaler_probe.py", "traincheck/scaler_probe.py",
    "traincheck/schedule_study.py", "traincheck/heldout_workload.py",
    "traincheck/heldout_datasets.py", "requirements/study-cpu.txt",
    ".github/workflows/matched-calibration.yml",
)
HOST_FIELDS = ("runner_boot_sha256", "cpu_affinity", "cpu_model", "python", "torch",
               "dataset_sha256", "source_revision", "workflow_run_id", "workflow_attempt")


def verify_host(development, evaluation):
    if not development.get("runner_boot_sha256") or len(development.get("cpu_affinity") or []) != 1:
        raise ValueError("A pinned CPU and verified boot identity are required")
    for field in HOST_FIELDS:
        if development[field] != evaluation[field]:
            raise ValueError(f"Calibration and evaluation differ in {field}")


def freeze(root, config_path):
    design = json.loads(config_path.read_bytes())
    summary = json.loads((root / "development-analysis/summary.json").read_bytes())
    manifest = summary["manifests"][0]
    if (summary["phase"] != "development" or manifest["design"] != design
            or manifest["failures"] or not manifest["runner_boot_sha256"]
            or len(manifest["cpu_affinity"]) != 1):
        raise ValueError("Development artifact cannot be frozen")
    design.update(status="frozen", thresholds=summary["thresholds"],
                  frozen_files={source: digest(ROOT / source) for source in SOURCES},
                  development_records_sha256=manifest["records_sha256"],
                  development_summary_sha256=digest(root / "development-analysis/summary.json"))
    (root / "frozen.json").write_text(json.dumps(design, indent=2) + "\n")


def compare(root):
    design = json.loads((root / "frozen.json").read_bytes())
    dev = json.loads((root / "development/schedule-digits/manifest.json").read_bytes())
    ev = json.loads((root / "evaluation/schedule-digits/manifest.json").read_bytes())
    verify_host(dev, ev)
    if (dev["records_sha256"] != design["development_records_sha256"]
            or digest(root / "development-analysis/summary.json") != design["development_summary_sha256"]
            or ev["design"] != design
            or (os.getenv("GITHUB_SHA") and ev["source_revision"] != os.environ["GITHUB_SHA"])):
        raise ValueError("Study artifacts differ from the freeze")
    rows = load_records(root / "evaluation/schedule-digits/records.jsonl")
    rates, pairs = [], []
    for budget, cutoff in design["thresholds"]["digits"].items():
        for condition in ev["conditions"]:
            selected = [r for r in rows if r["budget"] == budget and r["condition"] == condition]
            previous = [r["loss_score"] > design["previous_thresholds"][budget]
                        if r["status"] == "ok" else None for r in selected]
            current = [r["loss_score"] > cutoff if r["status"] == "ok" else None for r in selected]
            for policy, values in (("previous", previous), ("matched", current)):
                observed = [v for v in values if v is not None]
                alarms = sum(observed)
                low, high = wilson(alarms, len(observed))
                rates.append(dict(budget=budget, condition=condition, cutoff=policy,
                                  attempts=len(selected), evaluated=len(observed), alarms=alarms,
                                  low=low, high=high))
            if budget == "wall25" and condition in HEALTHY:
                pairs.append(dict(condition=condition, **mcnemar(previous, current)))
    apply_holm(pairs)
    write_csv(root / "transfer-rates.csv", rates)
    write_csv(root / "transfer-paired.csv", pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "compare"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--design", type=Path, default=ROOT / "docs/research/MATCHED_STUDY.json")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.root, args.design)
    else:
        compare(args.root)
