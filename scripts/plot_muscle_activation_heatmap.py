#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from segment_gait_cycles import (
    MUSCLE_GROUP_PATTERNS,
    filter_by_quality,
    heel_strikes,
    infer_condition,
    resample,
    rollout_files,
)


GROUP_ORDER = [
    ("ADD", ["addbrev", "addlong", "addmag"]),
    ("HFL", MUSCLE_GROUP_PATTERNS["HFL"]),
    ("GLU", MUSCLE_GROUP_PATTERNS["GLU"] + ["glmed", "glmin"]),
    ("HAM", MUSCLE_GROUP_PATTERNS["HAM"] + ["bfsh"]),
    ("RF", MUSCLE_GROUP_PATTERNS["RF"]),
    ("VAS", MUSCLE_GROUP_PATTERNS["VAS"]),
    ("GAS", MUSCLE_GROUP_PATTERNS["GAS"]),
    ("SOL", MUSCLE_GROUP_PATTERNS["SOL"]),
    ("TA", MUSCLE_GROUP_PATTERNS["TA"] + ["edl", "ehl"]),
    ("FLEX", ["fdl", "fhl"]),
    ("OTHER", []),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot actuator-level muscle activation heatmaps.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[Path("paper/biomech_rollouts_recommended/flat")],
        help="Rollout NPZ files or directories containing *_rollout.npz.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["Fixed", "T2A", "AsymSDE", "FullSDE"],
        help="Condition panels to show, in order.",
    )
    parser.add_argument("--output-prefix", type=Path, default=Path("paper/biomech_rollouts_recommended/fig_muscle_activation_heatmap"))
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--min-cycle-frames", type=int, default=25)
    parser.add_argument("--max-cycle-frames", type=int, default=260)
    parser.add_argument("--contact-column", choices=["left", "right"], default="left")
    parser.add_argument("--quality-status", choices=["all", "accepted"], default="accepted")
    parser.add_argument(
        "--normalize",
        choices=["row", "global", "none"],
        default="row",
        help="row highlights timing per muscle; global preserves relative amplitude within each condition.",
    )
    return parser.parse_args()


def muscle_group(name: str) -> str:
    lower = name.lower()
    for group, patterns in GROUP_ORDER:
        if group == "OTHER":
            continue
        if any(pattern in lower for pattern in patterns):
            return group
    return "OTHER"


def side_rank(name: str) -> int:
    if name.endswith("_r"):
        return 0
    if name.endswith("_l"):
        return 1
    return 2


def sorted_muscle_indices(names: list[str]):
    group_rank = {group: idx for idx, (group, _patterns) in enumerate(GROUP_ORDER)}
    return sorted(
        range(len(names)),
        key=lambda idx: (group_rank.get(muscle_group(names[idx]), 99), names[idx].replace("_r", "").replace("_l", ""), side_rank(names[idx])),
    )


def accepted_cycles(path: Path, samples: int, min_len: int, max_len: int, contact_column: str):
    data = np.load(path, allow_pickle=True)
    contact = data["contact"]
    side_idx = 0 if contact_column == "left" else 1
    strikes = heel_strikes(contact[:, side_idx], min_interval=min_len)
    activation = data["muscle_activation"]
    cycles = []
    for start, stop in zip(strikes[:-1], strikes[1:]):
        length = int(stop - start)
        if length < min_len or length > max_len:
            continue
        rows = [resample(activation[start:stop, idx], samples) for idx in range(activation.shape[1])]
        cycles.append(np.vstack(rows))
    return cycles, [str(name) for name in data["actuator_names"].tolist()]


def normalize_matrix(matrix: np.ndarray, mode: str):
    if mode == "none":
        return matrix
    if mode == "global":
        vmax = np.nanmax(matrix)
        return matrix / vmax if np.isfinite(vmax) and vmax > 1e-8 else matrix
    denom = np.nanmax(matrix, axis=1, keepdims=True)
    denom[~np.isfinite(denom) | (denom < 1e-8)] = 1.0
    return matrix / denom


def group_boundaries(sorted_names: list[str]):
    bounds = []
    last = None
    for idx, name in enumerate(sorted_names):
        group = muscle_group(name)
        if last is not None and group != last:
            bounds.append((idx - 0.5, group))
        last = group
    return bounds


def main():
    args = parse_args()
    files = filter_by_quality(rollout_files(args.inputs), args.quality_status)
    by_condition = {condition: [] for condition in args.conditions}
    actuator_names = None

    for path in files:
        condition = infer_condition(path)
        if condition not in by_condition:
            continue
        cycles, names = accepted_cycles(path, args.samples, args.min_cycle_frames, args.max_cycle_frames, args.contact_column)
        if not cycles:
            continue
        if actuator_names is None:
            actuator_names = names
        by_condition[condition].extend(cycles)

    if actuator_names is None:
        raise SystemExit("No usable rollout cycles found.")

    order = sorted_muscle_indices(actuator_names)
    sorted_names = [actuator_names[idx] for idx in order]
    n_conditions = len(args.conditions)
    fig, axes = plt.subplots(1, n_conditions, figsize=(4.3 * n_conditions, 14.0), sharey=True, constrained_layout=True)
    if n_conditions == 1:
        axes = [axes]

    image = None
    for ax, condition in zip(axes, args.conditions):
        cycles = by_condition.get(condition, [])
        if cycles:
            matrix = np.nanmean(np.stack(cycles, axis=0), axis=0)[order, :]
            matrix = normalize_matrix(matrix, args.normalize)
            image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_title(f"{condition}\ncycles={len(cycles)}")
        else:
            ax.set_title(f"{condition}\ncycles=0")
        ax.set_xlabel("gait cycle (%)")
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100"])
        for boundary, _group in group_boundaries(sorted_names):
            ax.axhline(boundary, color="white", linewidth=0.45, alpha=0.65)

    axes[0].set_ylabel("muscle actuator")
    tick_step = 2 if len(sorted_names) > 45 else 1
    axes[0].set_yticks(np.arange(0, len(sorted_names), tick_step))
    axes[0].set_yticklabels(sorted_names[::tick_step], fontsize=5)

    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.75, pad=0.01)
        cbar.set_label("normalized activation" if args.normalize != "none" else "activation")

    fig.suptitle("Muscle activation timing across normalized gait cycles", fontsize=14)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(args.output_prefix.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
