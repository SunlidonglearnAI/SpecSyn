#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


METRICS_HIGH = [
    "return_mean",
    "distance_m_mean",
    "mean_forward_speed_mps_mean",
    "step_length_m_mean",
]

METRICS_LOW = [
    "duty_factor_asymmetry_mean",
    "leg_crossing_rate_mean",
    "pelvis_lateral_sway_std_m_mean",
    "pelvis_pitch_rms_deg_mean",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Screen FullSDE walk candidates with biomechanical rollouts.")
    parser.add_argument("--env-id", default="myoLegWalkT2Apca1-v0")
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("myosuite/agents/outputs_by_category/full_sde/myoLegWalkT2Apca1-v0/candidates"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper/full_sde_candidate_screen"))
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=0, help="Optional number of candidates to screen.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def candidate_links(candidate_dir: Path):
    items = []
    for link in sorted(candidate_dir.iterdir()):
        if not link.is_symlink():
            continue
        target = link.resolve()
        if (target / "myoLegWalkT2Apca1-v0_PPO_model.zip").exists() and (target / "myoLegWalkT2Apca1-v0_PPO_env").exists():
            items.append((link.name, target))
    return items


def read_condition_summary(path: Path):
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    if not rows:
        return {}
    row = rows[0]
    return row


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def zscore(values, higher_is_better=True):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=float)
    if finite.sum() < 2:
        return out
    mean = np.nanmean(arr[finite])
    std = np.nanstd(arr[finite])
    if std < 1e-9:
        return out
    out[finite] = (arr[finite] - mean) / std
    if not higher_is_better:
        out = -out
    return out


def rank_rows(rows):
    scores = np.zeros(len(rows), dtype=float)
    for key in METRICS_HIGH:
        scores += zscore([as_float(row.get(key)) for row in rows], higher_is_better=True)
    for key in METRICS_LOW:
        scores += zscore([as_float(row.get(key)) for row in rows], higher_is_better=False)
    for row, score in zip(rows, scores):
        row["composite_score"] = f"{score:.6f}" if math.isfinite(float(score)) else "nan"
    return sorted(rows, key=lambda row: as_float(row.get("composite_score")), reverse=True)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidate_links(args.candidate_dir)
    if args.limit > 0:
        candidates = candidates[: args.limit]
    if not candidates:
        raise SystemExit(f"No usable candidates found in {args.candidate_dir}")

    rows = []
    for idx, (candidate_name, model_dir) in enumerate(candidates, start=1):
        label = f"cand_{idx:02d}"
        out_dir = args.output_dir / label
        cmd = [
            sys.executable,
            "biomech_validate_success.py",
            "--condition",
            f"{label},{args.env_id},{model_dir}",
            "--episodes",
            str(args.episodes),
            "--num-seeds",
            str(args.num_seeds),
            "--max-steps",
            str(args.max_steps),
            "--reset-type",
            "random",
            "--filter-quality",
            "--min-frames",
            "80",
            "--min-distance",
            "0.25",
            "--min-steps",
            "2",
            "--min-speed",
            "0.05",
            "--return-mad-threshold",
            "3.0",
            "--terrain",
            "flat",
            "--device",
            args.device,
            "--output-dir",
            str(out_dir),
        ]
        print(f"[{idx}/{len(candidates)}] {candidate_name} -> {model_dir}", flush=True)
        result = subprocess.run(cmd, check=False)
        row = {
            "label": label,
            "candidate": candidate_name,
            "model_dir": str(model_dir),
            "screen_output": str(out_dir),
            "status": "ok" if result.returncode == 0 else f"failed:{result.returncode}",
        }
        comparison = out_dir / "condition_comparison.csv"
        if comparison.exists() and result.returncode == 0:
            row.update(read_condition_summary(comparison))
        rows.append(row)

    ranked = rank_rows(rows)
    fieldnames = sorted({key for row in ranked for key in row.keys()})
    preferred = [
        "label",
        "candidate",
        "status",
        "composite_score",
        "n_episodes",
        "return_mean",
        "distance_m_mean",
        "mean_forward_speed_mps_mean",
        "step_length_m_mean",
        "duty_factor_asymmetry_mean",
        "leg_crossing_rate_mean",
        "pelvis_lateral_sway_std_m_mean",
        "pelvis_pitch_rms_deg_mean",
        "model_dir",
        "screen_output",
    ]
    fieldnames = [f for f in preferred if f in fieldnames] + [f for f in fieldnames if f not in preferred]
    with (args.output_dir / "full_sde_candidate_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked)

    md = ["# FullSDE Candidate Screening", ""]
    md.append("| Rank | Candidate | Score | Episodes | Speed | Step length | Duty asym | Leg crossing | Model |")
    md.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for rank, row in enumerate(ranked, start=1):
        md.append(
            f"| {rank} | `{row.get('candidate', '')}` | {as_float(row.get('composite_score')):.2f} | "
            f"{row.get('n_episodes', '')} | {as_float(row.get('mean_forward_speed_mps_mean')):.2f} | "
            f"{as_float(row.get('step_length_m_mean')):.2f} | {as_float(row.get('duty_factor_asymmetry_mean')):.2f} | "
            f"{as_float(row.get('leg_crossing_rate_mean')):.2f} | `{row.get('model_dir', '')}` |"
        )
    (args.output_dir / "full_sde_candidate_ranking.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'full_sde_candidate_ranking.csv'}")
    print(f"Wrote {args.output_dir / 'full_sde_candidate_ranking.md'}")


if __name__ == "__main__":
    main()
