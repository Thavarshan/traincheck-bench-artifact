#!/usr/bin/env python3
"""Report exploratory optimization exposure, not effectiveness."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(rows: list[dict]) -> str:
    groups = defaultdict(list)
    for row in rows:
        if row.get("phase") != "exploratory" or row.get("fault") != "healthy":
            raise ValueError("pilot requires exploratory healthy-only records")
        groups[float(row["budget_seconds"])].append(row)
    lines = [
        "# Exploratory CPU budget exposure",
        "",
        "Pilot only: not confirmatory evidence or a fault-detection estimate.",
        "",
        "| Budget (s) | Successful/total | Steps median [min,max] | Loop median (s) | Diagnose median (s) | Cap terminations |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for budget, records in sorted(groups.items()):
        ok = [r for r in records if r.get("status") == "ok"]
        if not ok:
            lines.append(f"| {budget:g} | 0/{len(records)} | - | - | - | - |")
            continue
        steps = [int(r["steps"]) for r in ok]
        loops = [float(r["elapsed_seconds"]) for r in ok]
        whole = [float(r["diagnose_wall_seconds"]) for r in ok]
        caps = sum(r.get("termination_reason") == "max_steps" for r in ok)
        lines.append(
            f"| {budget:g} | {len(ok)}/{len(records)} | "
            f"{statistics.median(steps):g} [{min(steps)},{max(steps)}] | "
            f"{statistics.median(loops):.5f} | {statistics.median(whole):.5f} | {caps} |"
        )
    lines.extend(["", "## Paired seeds"])
    budgets = sorted(groups)
    if len(budgets) < 2:
        lines.append("- At least two budgets required for exposure comparison")
    for low, high in zip(budgets, budgets[1:]):
        lines.append(f"### {low:g} s to {high:g} s")
        a = {r["seed"]: r for r in groups[low] if r.get("status") == "ok"}
        b = {r["seed"]: r for r in groups[high] if r.get("status") == "ok"}
        seeds = sorted(set(a) & set(b))
        for seed in seeds:
            previous, next_steps = int(a[seed]["steps"]), int(b[seed]["steps"])
            lines.append(
                f"- Seed {seed}: {previous} to {next_steps} steps (change {next_steps-previous:+d})")
        if not seeds:
            lines.append("- No complete seed pairs")
        elif not any(b[seed]["steps"] > a[seed]["steps"] for seed in seeds):
            lines.append(
                "- **Warning:** no paired seed gained steps at the longer budget")
        if seeds and all(b[seed].get("termination_reason") == "max_steps" for seed in seeds):
            lines.append(
                "- **Exposure warning:** all longer-budget pairs hit the step ceiling")
    lines.append(
        "\nThis study-design pilot cannot establish recall, specificity, or generalization.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(s)
            for s in args.records.read_text().splitlines() if s.strip()]
    output = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
