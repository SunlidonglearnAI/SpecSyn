#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run batched deterministic or stochastic policy evaluations from a manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("paper/eval_rollouts/paper_eval_manifest.csv"),
    )
    parser.add_argument("--mode", choices=["deterministic", "stochastic"], required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output-root", type=Path, default=Path("paper/eval_rollouts"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def default_episodes(mode: str) -> int:
    return 20 if mode == "deterministic" else 50


def main():
    args = parse_args()
    episodes = args.episodes or default_episodes(args.mode)

    with args.manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit > 0:
        rows = rows[: args.limit]

    mode_dir = args.output_root / args.mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(rows, start=1):
        label = row["label"]
        out_dir = mode_dir / label
        cmd = [
            args.python_bin,
            "biomech_validate_success.py",
            "--condition",
            f"{label},{row['env_id']},{row['model_path']},{row['vecnormalize_path']}",
            "--episodes",
            str(episodes),
            "--max-steps",
            str(args.max_steps),
            "--reset-type",
            "random",
            "--terrain",
            row["terrain"],
            "--device",
            args.device,
            "--output-dir",
            str(out_dir),
        ]
        if args.mode == "stochastic":
            cmd.append("--stochastic")
        print(f"[{idx}/{len(rows)}] {label} -> {out_dir}", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
