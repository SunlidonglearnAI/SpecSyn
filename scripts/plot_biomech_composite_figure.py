#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CONDITION_STYLE = {
    "FullSDE": {"color": "#0072B2", "lw": 2.6, "alpha": 1.0, "zorder": 5},
    "Fixed": {"color": "#666666", "lw": 1.4, "alpha": 0.85, "zorder": 2},
    "T2A": {"color": "#D55E00", "lw": 1.5, "alpha": 0.85, "zorder": 3},
    "StrengthOnly": {"color": "#CC79A7", "lw": 1.3, "alpha": 0.75, "zorder": 2},
    "AsymSDE": {"color": "#009E73", "lw": 1.3, "alpha": 0.75, "zorder": 2},
}

JOINTS = [
    ("Hip", "hip_flexion", "deg"),
    ("Knee", "knee_angle", "deg"),
    ("Ankle", "ankle_angle", "deg"),
]

MUSCLE_GROUPS = [
    ("HFL", "hip flexors"),
    ("GLU", "gluteals"),
    ("VAS", "vasti"),
    ("HAM", "hamstrings"),
    ("GAS", "gastrocnemius"),
    ("SOL", "soleus"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot composite biomechanical gait-cycle figure.")
    parser.add_argument(
        "--timeseries",
        type=Path,
        default=Path("paper/biomech_rollouts_recommended/gait_cycles/gait_cycle_timeseries.csv"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("paper/biomech_rollouts_recommended/flat/condition_comparison.csv"),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["Fixed", "T2A", "StrengthOnly", "AsymSDE", "FullSDE"],
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("paper/biomech_rollouts_recommended/fig_biomech_composite"),
    )
    parser.add_argument("--title", default="Biomechanical validation across normalized gait cycles")
    return parser.parse_args()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_timeseries(path: Path, conditions: list[str]):
    rows = read_csv(path)
    by_condition = {condition: [] for condition in conditions}
    for row in rows:
        condition = row.get("condition", "")
        if condition in by_condition:
            by_condition[condition].append(row)
    return by_condition


def cycle_matrix(rows, column: str):
    grouped = {}
    for row in rows:
        key = (row.get("rollout_file", ""), row.get("cycle_idx", ""))
        grouped.setdefault(key, []).append(row)
    curves = []
    for _key, items in grouped.items():
        items.sort(key=lambda row: as_float(row.get("gait_percent", "nan")))
        values = np.asarray([as_float(row.get(column)) for row in items], dtype=float)
        if values.size == 101 and np.isfinite(values).sum() > 50:
            curves.append(values)
    if not curves:
        return np.empty((0, 101))
    return np.vstack(curves)


def mean_sem(curves):
    if curves.size == 0:
        return None, None
    mean = np.nanmean(curves, axis=0)
    sem = np.nanstd(curves, axis=0) / np.sqrt(max(curves.shape[0], 1))
    return mean, sem


def plot_condition_curves(
    ax,
    by_condition,
    column,
    conditions,
    ylabel=None,
    full_band=True,
    normalize=False,
    linestyle="-",
    show_labels=True,
):
    x = np.linspace(0, 100, 101)
    for condition in conditions:
        style = CONDITION_STYLE.get(condition, {"color": "black", "lw": 1.2, "alpha": 0.7, "zorder": 1})
        curves = cycle_matrix(by_condition.get(condition, []), column)
        if normalize and curves.size:
            denom = np.nanmax(np.abs(curves), axis=1, keepdims=True)
            denom[denom < 1e-8] = 1.0
            curves = curves / denom
        mean, sem = mean_sem(curves)
        if mean is None:
            continue
        ax.plot(
            x,
            mean,
            color=style["color"],
            linestyle=linestyle,
            lw=style["lw"],
            alpha=style["alpha"],
            label=f"{condition} (n={curves.shape[0]})" if show_labels else None,
            zorder=style["zorder"],
        )
        if full_band and condition == "FullSDE":
            ax.fill_between(x, mean - sem, mean + sem, color=style["color"], alpha=0.18, linewidth=0)
    ax.axvspan(0, 60, color="#f2f2f2", zorder=0)
    ax.set_xlim(0, 100)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, linewidth=0.7)


def load_metric_summary(path: Path):
    if not path.exists():
        return {}
    out = {}
    for row in read_csv(path):
        condition = row.get("condition", "")
        if condition:
            out[condition] = row
    return out


def metric_value(metrics, condition, key):
    row = metrics.get(condition, {})
    return as_float(row.get(f"{key}_mean"))


def plot_metric_bars(ax, metrics, conditions):
    keys = [
        ("mean_forward_speed_mps", "speed"),
        ("step_length_m", "step"),
        ("duty_factor_asymmetry", "asym"),
        ("leg_crossing_rate", "cross"),
    ]
    x = np.arange(len(keys))
    width = 0.14
    offsets = np.linspace(-2, 2, len(conditions)) * width
    for offset, condition in zip(offsets, conditions):
        style = CONDITION_STYLE.get(condition, {"color": "black", "alpha": 0.7})
        vals = [metric_value(metrics, condition, key) for key, _label in keys]
        ax.bar(x + offset, vals, width=width, color=style["color"], alpha=style["alpha"], label=condition)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _key, label in keys])
    ax.set_title("Spatiotemporal / abnormality metrics")
    ax.grid(True, axis="y", alpha=0.22)


def main():
    args = parse_args()
    by_condition = load_timeseries(args.timeseries, args.conditions)
    metrics = load_metric_summary(args.metrics)

    fig = plt.figure(figsize=(17.5, 12.5))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.05, 1.05, 1.15, 0.95], hspace=0.38, wspace=0.25)

    for col, (title, joint, _unit) in enumerate(JOINTS):
        ax = fig.add_subplot(gs[0, col])
        right_col = f"{joint}_r_deg"
        left_col = f"{joint}_l_deg"
        plot_condition_curves(ax, by_condition, right_col, args.conditions, ylabel="angle (deg)" if col == 0 else None)
        plot_condition_curves(ax, by_condition, left_col, args.conditions, full_band=False, linestyle="--", show_labels=False)
        ax.set_title(f"{title} angle")
        if col == 0:
            ax.text(-0.16, 1.08, "A", transform=ax.transAxes, fontsize=16, fontweight="bold")

    for col, (title, joint, _unit) in enumerate(JOINTS):
        ax = fig.add_subplot(gs[1, col])
        right_col = f"{joint}_r_vel_deg_s"
        left_col = f"{joint}_l_vel_deg_s"
        plot_condition_curves(ax, by_condition, right_col, args.conditions, ylabel="angular velocity (deg/s)" if col == 0 else None)
        plot_condition_curves(ax, by_condition, left_col, args.conditions, full_band=False, linestyle="--", show_labels=False)
        ax.set_title(f"{title} angular velocity")
        if col == 0:
            ax.text(-0.16, 1.08, "B", transform=ax.transAxes, fontsize=16, fontweight="bold")

    for idx, (group, label) in enumerate(MUSCLE_GROUPS):
        ax = fig.add_subplot(gs[2 + idx // 3, idx % 3])
        plot_condition_curves(
            ax,
            by_condition,
            f"activation_{group}",
            args.conditions,
            ylabel="activation (norm.)" if idx % 3 == 0 else None,
            normalize=True,
        )
        ax.set_ylim(-0.08, 1.12)
        ax.set_title(f"{group}: {label}")
        ax.set_xlabel("gait cycle (%)" if idx >= 3 else "")
        if idx == 0:
            ax.text(-0.16, 1.08, "C", transform=ax.transAxes, fontsize=16, fontweight="bold")

    metric_ax = fig.add_axes([0.70, 0.035, 0.27, 0.14])
    plot_metric_bars(metric_ax, metrics, args.conditions)
    metric_ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.28))

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(args.conditions), frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(args.title, y=0.995, fontsize=15)
    fig.text(
        0.02,
        0.02,
        "Kinematics: solid=right leg, dashed=left leg. Shaded vertical band: nominal stance phase (0-60% gait cycle). FullSDE band: SEM.",
        fontsize=9,
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
