"""Render the manuscript figures from the archived retrospective analysis tables."""

import argparse
import csv
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter


def read_csv(directory, name):
    with (directory / f"{name}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def score_plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.12), sharex=True, sharey=True)
    names = ("Plain", "Accumulation", "Declared skips", "Quarter omission")
    groups = ("plain", "accumulate2", "declared_skip4", "periodic4")
    color = "#1f506f"
    cutoffs = (("cross_host_cutoff", "--", "#9b4b14", "Cross-host shared cutoff"),
               ("matched_cutoff", "-.", "#1f674b", "Matched-runner shared cutoff"),
               ("plain_cutoff", ":", "#542f72", "Matched plain cutoff"))
    rng = random.Random(317)
    jitter = [rng.uniform(-0.22, 0.22) for _ in range(80)]
    for axis, (budget, title) in zip(axes, (("wall25", "25 ms"), ("fixed64", "64 minibatches")), strict=True):
        selected = [row for row in rows if row["budget"] == budget]
        for index, condition in enumerate(groups):
            ordered = sorted((row for row in selected if row["condition"] == condition),
                             key=lambda row: int(row["seed"]))
            axis.scatter([float(row["loss_score"]) for row in ordered],
                         [index + offset for offset in jitter], s=7, alpha=0.55,
                         color=color if index < 3 else "#92453c", linewidths=0, zorder=3)
        for key, style, hue, _ in cutoffs:
            axis.axvline(float(selected[0][key]), color=hue, linestyle=style, linewidth=1)
        axis.set_title(title)
        axis.set_yticks(range(len(names)), labels=names)
        axis.set_xlabel("Final / initial validation loss")
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#dedede", linewidth=0.5)
    axes[0].invert_yaxis()
    handles = [Line2D([0], [0], color=hue, linestyle=style, label=label)
               for _, style, hue, label in cutoffs]
    fig.legend(handles=handles, loc="lower center", ncols=3, frameon=False)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.87, bottom=0.31, wspace=0.13)
    return fig


def severity_plot(rows, operating):
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 3.35), sharex=True, sharey="row")
    budgets = (("wall25", "25 ms", "#17547f", "-"), ("fixed64", "64 minibatches", "#9b4b14", "--"))
    patterns = (("periodic", "Periodic", "o"), ("bernoulli", "Bernoulli", "^"))
    baselines = (("plain", "Plain baseline", (0, (1, 2))),
                 ("accumulate2", "Accumulation baseline", (0, (6, 2, 1, 2))))
    for col, (workload, title) in enumerate((("digits", "Digits"), ("wine", "Wine"), ("fashion", "Fashion-MNIST"))):
        axes[0, col].set_title(title, pad=4)
        for budget, _, color, style in budgets:
            for schedule, _, baseline_style in baselines:
                healthy = next(row for row in operating if
                               (row["study"], row["workload"], row["budget"], row["policy"], row["condition"])
                               == ("cross_host", workload, budget, "shared", schedule))
                axes[0, col].axhline(100 * int(healthy["alarms"]) / int(healthy["n"]),
                                    color=color, linestyle=baseline_style, alpha=0.62, linewidth=0.9)
            cells = [row for row in rows if row["workload"] == workload and row["budget"] == budget]
            full = next(row for row in cells if row["condition"] == "full")
            for pattern, _, marker in patterns:
                selected = sorted((row for row in cells if row["condition"].startswith(pattern)),
                                  key=lambda row: float(row["median_omission"]))
                x_values = [100 * float(row["median_omission"]) for row in selected]
                y_alarm = [100 * int(row["alarms"]) / int(row["realized"]) for row in selected]
                y_quality = [100 * float(row["median_accuracy_drop"]) for row in selected]
                for axis, values in zip(axes[:, col], (y_alarm, y_quality), strict=True):
                    axis.plot(x_values, values, color=color, linestyle=style, marker=marker,
                              markersize=3.5, markerfacecolor="white", linewidth=1)
            for axis, value in zip(axes[:, col], (100 * int(full["alarms"]) / int(full["realized"]),
                                                 100 * float(full["median_accuracy_drop"])), strict=True):
                axis.plot(100, value, color=color, marker="D", markersize=3.5, markerfacecolor="white")
        for axis in axes[:, col]:
            axis.set_xscale("log", base=2)
            axis.set_xticks([3.125, 6.25, 12.5, 25, 50, 100], labels=["3", "6", "12.5", "25", "50", "100"])
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_xlim(2.6, 117)
            axis.grid(color="#e4e4e4", linewidth=0.5)
            axis.set_axisbelow(True)
    axes[0, 0].set_ylabel("Alarm rate (%)")
    axes[0, 0].set_ylim(-7, 107)
    axes[0, 0].set_yticks([0, 25, 50, 75, 100])
    axes[1, 0].set_ylabel("Accuracy drop (pp)")
    for axis in axes[1]:
        axis.axhline(0, color="#818181", linewidth=0.6, zorder=0)
    handles = [Line2D([0], [0], color=color, linestyle=style, label=label)
               for _, label, color, style in budgets]
    handles += [Line2D([0], [0], color="#444444", marker=marker, linestyle="none",
                       markerfacecolor="white", label=label) for _, label, marker in patterns]
    handles.append(Line2D([0], [0], color="#444444", marker="D", linestyle="none",
                          markerfacecolor="white", label="Full"))
    handles += [Line2D([0], [0], color="#444444", linestyle=style, label=label)
                for _, label, style in baselines]
    fig.supxlabel("Median observed omission among realized faults (%)", y=0.16, fontsize=9)
    fig.legend(handles=handles, loc="lower center", ncols=4, frameon=False)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.91, bottom=0.26, wspace=0.17, hspace=0.18)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    figures = args.input / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    plots = {
        "validation-loss": score_plot(read_csv(args.input, "matched-scores")),
        "fault-severity": severity_plot(read_csv(args.input, "severity"),
                                        read_csv(args.input, "per-schedule-operating-points")),
    }
    for name, figure in plots.items():
        figure.savefig(figures / f"{name}.pdf", metadata={"Title": name, "CreationDate": None})
        plt.close(figure)
    (args.input / "plot-environment.json").write_text(json.dumps({"matplotlib": matplotlib.__version__}) + "\n")


if __name__ == "__main__":
    main()
