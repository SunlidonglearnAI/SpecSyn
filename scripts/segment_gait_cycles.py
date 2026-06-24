#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


SAGITTAL_JOINT_NAMES = [
    "hip_flexion_r",
    "hip_flexion_l",
    "knee_angle_r",
    "knee_angle_l",
    "ankle_angle_r",
    "ankle_angle_l",
]

MUSCLE_GROUP_PATTERNS = {
    "HFL": ["iliacus", "psoas"],
    "GLU": ["glmax"],
    "HAM": ["bflh", "semimem", "semiten"],
    "RF": ["recfem"],
    "VAS": ["vasint", "vaslat", "vasmed"],
    "GAS": ["gaslat", "gasmed"],
    "SOL": ["soleus"],
    "TA": ["tibant"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Segment rollout NPZ files into normalized gait cycles.")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Rollout NPZ files or directories containing *_rollout.npz.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper/gait_cycle_segments"))
    parser.add_argument("--samples", type=int, default=101, help="Samples per normalized gait cycle.")
    parser.add_argument("--min-cycle-frames", type=int, default=25)
    parser.add_argument("--max-cycle-frames", type=int, default=260)
    parser.add_argument(
        "--contact-column",
        choices=["left", "right"],
        default="left",
        help="Foot contact channel used to define successive gait cycles.",
    )
    parser.add_argument(
        "--quality-status",
        choices=["all", "accepted"],
        default="accepted",
        help="When episode_quality_filter.csv is present, keep only this rollout quality subset.",
    )
    return parser.parse_args()


def rollout_files(inputs):
    files = []
    for item in inputs:
        if item.is_dir():
            files.extend(sorted(item.rglob("*_rollout.npz")))
        elif item.name.endswith(".npz"):
            files.append(item)
    return sorted(dict.fromkeys(files))


def heel_strikes(contact: np.ndarray, min_interval: int = 20) -> np.ndarray:
    c = contact.astype(bool)
    edges = np.where((~c[:-1]) & c[1:])[0] + 1
    if edges.size == 0:
        return edges
    keep = [int(edges[0])]
    for edge in edges[1:]:
        if int(edge) - keep[-1] >= min_interval:
            keep.append(int(edge))
    return np.asarray(keep, dtype=int)


def resample(values: np.ndarray, n: int) -> np.ndarray:
    if values.size < 2:
        return np.full(n, np.nan)
    x_old = np.linspace(0.0, 1.0, values.shape[0])
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, values)


def safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")


def infer_condition(path: Path) -> str:
    parts = path.parts
    for marker in ("biomech_rollouts", "biomech_validation_walk", "biomech_validation_rough"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    if path.parent.name:
        return path.parent.name
    return "condition"


def rollout_seed(path: Path) -> str | None:
    match = re.search(r"_seed_([^_]+)_episode_", path.name)
    return match.group(1) if match else None


def accepted_seed_set(condition_dir: Path) -> set[str] | None:
    quality_path = condition_dir / "episode_quality_filter.csv"
    if not quality_path.exists():
        return None
    accepted = set()
    with quality_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("quality_status") == "accepted":
                accepted.add(str(row.get("seed", "")))
    return accepted


def filter_by_quality(files, quality_status: str):
    if quality_status == "all":
        return files
    filtered = []
    cache = {}
    for path in files:
        condition_dir = path.parent
        if condition_dir not in cache:
            cache[condition_dir] = accepted_seed_set(condition_dir)
        accepted = cache[condition_dir]
        if accepted is None:
            filtered.append(path)
            continue
        seed = rollout_seed(path)
        if seed is not None and seed in accepted:
            filtered.append(path)
    return filtered


def joint_index(joint_names: np.ndarray, name: str) -> int | None:
    names = [str(x) for x in joint_names.tolist()]
    return names.index(name) if name in names else None


def muscle_group_indices(actuator_names: np.ndarray):
    names = [str(x).lower() for x in actuator_names.tolist()]
    groups = {}
    for group, patterns in MUSCLE_GROUP_PATTERNS.items():
        idxs = [i for i, name in enumerate(names) if any(pattern in name for pattern in patterns)]
        groups[group] = idxs
    return groups


def cycle_rows_for_file(path: Path, samples: int, min_len: int, max_len: int, contact_column: str):
    data = np.load(path, allow_pickle=True)
    contact = data["contact"]
    side_idx = 0 if contact_column == "left" else 1
    strikes = heel_strikes(contact[:, side_idx], min_interval=min_len)
    joint_names = data["joint_names"]
    joint_angles = data["joint_angles"]
    joint_velocities = data["joint_velocities"]
    pelvis = data["pelvis_pos"]
    muscle_activation = data["muscle_activation"]
    actuator_names = data["actuator_names"] if "actuator_names" in data else np.asarray([])
    muscle_groups = muscle_group_indices(actuator_names) if actuator_names.size else {}
    cycle_rows = []
    signal_rows = []
    condition = infer_condition(path)

    for cycle_idx, (start, stop) in enumerate(zip(strikes[:-1], strikes[1:]), start=1):
        length = int(stop - start)
        if length < min_len or length > max_len:
            continue
        duration = float(data["time"][stop - 1] - data["time"][start]) if "time" in data else float(length) * 0.01
        distance = float(abs(pelvis[stop - 1, 1] - pelvis[start, 1]))
        mean_speed = distance / max(duration, 1e-9)
        stance_fraction = float(np.mean(contact[start:stop, side_idx]))
        opposite_stance_fraction = float(np.mean(contact[start:stop, 1 - side_idx]))
        cycle_rows.append(
            {
                "condition": condition,
                "rollout_file": str(path),
                "cycle_idx": cycle_idx,
                "start_frame": int(start),
                "stop_frame": int(stop),
                "n_frames": length,
                "duration_s": duration,
                "distance_m": distance,
                "mean_speed_mps": mean_speed,
                f"{contact_column}_stance_fraction": stance_fraction,
                "opposite_stance_fraction": opposite_stance_fraction,
            }
        )

        percent = np.linspace(0.0, 100.0, samples)
        signal = {
            "condition": condition,
            "rollout_file": str(path),
            "cycle_idx": cycle_idx,
        }
        for pct_idx, pct in enumerate(percent):
            signal_rows.append({**signal, "gait_percent": pct_idx, "gait_percent_value": pct})

        for joint in SAGITTAL_JOINT_NAMES:
            idx = joint_index(joint_names, joint)
            if idx is None:
                continue
            angle = np.rad2deg(joint_angles[start:stop, idx])
            velocity = np.rad2deg(joint_velocities[start:stop, idx])
            angle_rs = resample(angle, samples)
            vel_rs = resample(velocity, samples)
            for row, angle_value, vel_value in zip(signal_rows[-samples:], angle_rs, vel_rs):
                row[f"{joint}_deg"] = angle_value
                row[f"{joint}_vel_deg_s"] = vel_value

        mean_activation = muscle_activation[start:stop].mean(axis=1)
        mean_activation_rs = resample(mean_activation, samples)
        for row, value in zip(signal_rows[-samples:], mean_activation_rs):
            row["mean_muscle_activation"] = value

        for group, idxs in muscle_groups.items():
            if not idxs:
                continue
            values = muscle_activation[start:stop, idxs].mean(axis=1)
            values_rs = resample(values, samples)
            for row, value in zip(signal_rows[-samples:], values_rs):
                row[f"activation_{group}"] = value

    return cycle_rows, signal_rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "condition",
        "rollout_file",
        "cycle_idx",
        "gait_percent",
        "gait_percent_value",
        "start_frame",
        "stop_frame",
        "n_frames",
        "duration_s",
        "distance_m",
        "mean_speed_mps",
    ]
    fieldnames = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_cycles(cycle_rows):
    grouped = {}
    for row in cycle_rows:
        grouped.setdefault(row["condition"], []).append(row)
    out = []
    metric_keys = ["duration_s", "distance_m", "mean_speed_mps", "n_frames"]
    for condition, rows in sorted(grouped.items()):
        summary = {"condition": condition, "n_cycles": len(rows)}
        for key in metric_keys:
            vals = np.asarray([float(row[key]) for row in rows], dtype=float)
            summary[f"{key}_mean"] = float(np.nanmean(vals))
            summary[f"{key}_std"] = float(np.nanstd(vals))
        out.append(summary)
    return out


def main():
    args = parse_args()
    files = filter_by_quality(rollout_files(args.inputs), args.quality_status)
    all_cycles = []
    all_signals = []
    for path in files:
        cycle_rows, signal_rows = cycle_rows_for_file(
            path,
            samples=args.samples,
            min_len=args.min_cycle_frames,
            max_len=args.max_cycle_frames,
            contact_column=args.contact_column,
        )
        all_cycles.extend(cycle_rows)
        all_signals.extend(signal_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(all_cycles, args.output_dir / "gait_cycles.csv")
    write_csv(all_signals, args.output_dir / "gait_cycle_timeseries.csv")
    write_csv(summarize_cycles(all_cycles), args.output_dir / "gait_cycle_summary.csv")
    print(f"Processed {len(files)} rollout files")
    print(f"Extracted {len(all_cycles)} gait cycles")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
