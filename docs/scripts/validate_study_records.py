#!/usr/bin/env python3
"""Validate raw research records against provenance and design manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def validate(raw: bytes, manifest: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]
    if hashlib.sha256(raw).hexdigest() != manifest.get("records_sha256"):
        errors.append("raw JSONL SHA-256 mismatch")
    rows: list[dict] = []
    malformed = 0
    for number, text in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            malformed += 1
            errors.append(f"malformed JSONL row {number}")
            continue
        if not isinstance(row, dict):
            malformed += 1
            errors.append(f"non-object JSONL row {number}")
            continue
        rows.append(row)
    for key, actual in (
        ("malformed_line_count", malformed),
        ("record_count", len(rows)),
        ("success_count", sum(r.get("status") == "ok" for r in rows)),
        ("failure_count", sum(r.get("status") != "ok" for r in rows)),
    ):
        if manifest.get(key) != actual:
            errors.append(f"{key} disagrees with manifest")
    try:
        seeds = manifest["requested_seeds"]
        faults = manifest["requested_faults"]
        budgets = manifest["requested_budgets_seconds"]
        max_steps = manifest["requested_max_steps"]
        if (
            not seeds or not faults or not budgets
            or len(seeds) != len(set(seeds))
            or len(faults) != len(set(faults))
            or len(budgets) != len(set(budgets))
            or any(type(s) is not int for s in seeds)
            or any(type(f) is not str for f in faults)
            or any(type(b) not in (float, int) or not math.isfinite(b) or b <= 0 for b in budgets)
            or type(max_steps) is not int or max_steps < 1
        ):
            raise ValueError("invalid design dimensions")
        expected = Counter(
            (seed, fault, float(budget))
            for seed in seeds for fault in faults for budget in budgets
        )
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"invalid requested design: {error}")
        return errors

    observed = Counter()
    for index, row in enumerate(rows, start=1):
        if row.get("phase") != manifest.get("phase"):
            errors.append(f"row {index}: phase differs from manifest")
        if row.get("source_revision") != manifest.get("source_revision"):
            errors.append(
                f"row {index}: source revision differs from manifest")
        if row.get("status") not in ("ok", "failed"):
            errors.append(f"row {index}: invalid run status")
        if row.get("working_tree_dirty") is True:
            errors.append(f"row {index}: tracked source dirty")
        if not isinstance(row.get("configuration"), dict) or row["configuration"].get("max_steps") != max_steps:
            errors.append(f"row {index}: max_steps differs from manifest")
        seed, fault, budget = row.get("seed"), row.get(
            "fault"), row.get("budget_seconds")
        if type(seed) is not int or type(fault) is not str or type(budget) not in (int, float):
            errors.append(f"row {index}: invalid design key")
            continue
        if not math.isfinite(budget):
            errors.append(f"row {index}: nonfinite budget")
            continue
        observed[(seed, fault, float(budget))] += 1
        if row.get("status") == "ok":
            steps, elapsed, wall = row.get("steps"), row.get(
                "elapsed_seconds"), row.get("diagnose_wall_seconds")
            if type(steps) is not int or not 0 <= steps <= max_steps:
                errors.append(f"row {index}: invalid step count")
            if (
                type(elapsed) not in (float, int)
                or type(wall) not in (float, int)
                or not math.isfinite(elapsed) or not math.isfinite(wall)
                or not 0 <= elapsed <= wall
            ):
                errors.append(f"row {index}: invalid timing scope")
            # Newer source revisions distinguish attempted iterations from
            # optimizer.step() invocations, and measure overshoot magnitude.
            # Do not invalidate already archived legacy records lacking these fields.
            config = row.get("configuration")
            if isinstance(config, dict) and "step_counter_semantics" in config:
                calls = row.get("optimizer_step_calls")
                overshoot = row.get("budget_overshoot_seconds")
                if type(calls) is not int or type(steps) is not int or not 0 <= calls <= steps:
                    errors.append(f"row {index}: invalid optimizer-call count")
                if (
                    type(overshoot) not in (int, float)
                    or not math.isfinite(overshoot)
                    or overshoot < 0
                    or type(elapsed) not in (int, float)
                    or not math.isfinite(elapsed)
                    or abs(overshoot - max(0.0, elapsed - budget)) > 1e-6
                ):
                    errors.append(
                        f"row {index}: invalid budget-overshoot measurement")
            if not isinstance(row.get("detections"), dict):
                errors.append(f"row {index}: missing check outputs")
    if observed != expected:
        missing = sum((expected - observed).values())
        unexpected = sum((observed - expected).values())
        errors.append(
            f"design combinations mismatch: {missing} missing, {unexpected} duplicate/unexpected")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    issues = validate(args.records.read_bytes(), json.loads(
        args.manifest.read_text(encoding="utf-8")))
    if issues:
        for problem in issues:
            print(f"BUNDLE INVALID: {problem}")
        return 1
    print("Bundle integrity verified: hash, provenance, counts, timing and complete design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
