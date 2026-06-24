#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TABLE_KEYS = [
    ("return", "reward"),
    ("survived_horizon", "success"),
    ("fell_or_failed", "fall"),
    ("mean_forward_speed_mps", "speed"),
    ("step_length_m", "step"),
    ("duty_factor_asymmetry", "duty_asym"),
    ("leg_crossing_rate", "leg_cross"),
    ("pelvis_pitch_rms_deg", "pelvis_pitch"),
    ("sigma_mean", "sigma"),
    ("nu_mean", "nu"),
    ("kappa_mean", "kappa"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a paper-ready evaluation table from deterministic/stochastic rollout outputs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("paper/eval_rollouts/paper_eval_manifest.csv"),
    )
    parser.add_argument("--det-root", type=Path, default=Path("paper/eval_rollouts/deterministic"))
    parser.add_argument("--stoch-root", type=Path, default=Path("paper/eval_rollouts/stochastic"))
    parser.add_argument("--output-prefix", type=Path, default=Path("paper/eval_rollouts/paper_evaluation_table"))
    return parser.parse_args()


def load_summary(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric_mean(summary: dict, key: str):
    value = summary.get(key, {})
    if isinstance(value, dict):
        return value.get("mean", "")
    return value


def metric_std(summary: dict, key: str):
    value = summary.get(key, {})
    if isinstance(value, dict):
        return value.get("std", "")
    return ""


def fmt(value):
    if value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def main():
    args = parse_args()
    with args.manifest.open("r", newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))

    rows = []
    for item in manifest_rows:
        label = item["label"]
        det = load_summary(args.det_root / label / "metrics_summary.json")
        stoch = load_summary(args.stoch_root / label / "metrics_summary.json")
        row = {
            "method": item["method"],
            "terrain": item["terrain"],
            "training_seed": item["training_seed"],
            "env_id": item["env_id"],
        }
        for key, short in TABLE_KEYS:
            row[f"det_{short}_mean"] = metric_mean(det, key)
            row[f"det_{short}_std"] = metric_std(det, key)
            row[f"stoch_{short}_mean"] = metric_mean(stoch, key)
            row[f"stoch_{short}_std"] = metric_std(stoch, key)
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else ["method", "terrain", "training_seed", "env_id"]
    csv_path = args.output_prefix.with_suffix(".csv")
    md_path = args.output_prefix.with_suffix(".md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Paper Evaluation Table", ""]
    lines.append("| Method | Terrain | Seed | Det reward | Det success | Det speed | Stoch reward | Stoch success | Stoch speed | Sigma | Nu | Kappa |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['terrain']} | {row['training_seed']} | "
            f"{fmt(row['det_reward_mean'])} | {fmt(row['det_success_mean'])} | {fmt(row['det_speed_mean'])} | "
            f"{fmt(row['stoch_reward_mean'])} | {fmt(row['stoch_success_mean'])} | {fmt(row['stoch_speed_mean'])} | "
            f"{fmt(row['stoch_sigma_mean'])} | {fmt(row['stoch_nu_mean'])} | {fmt(row['stoch_kappa_mean'])} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
