#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


GROUP_METHOD = {
    "fixed": "Fixed",
    "direct_t2a": "T2A",
    "full_sde": "FullSDE",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a seed-level evaluation manifest for paper rollouts.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("myosuite/agents/outputs_by_category/MANIFEST.csv"),
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["fixed", "direct_t2a", "full_sde", "symmetry", "single_property", "latent_k"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/eval_rollouts/paper_eval_manifest.csv"),
    )
    return parser.parse_args()


def as_int(text: str, default: int = -10**9) -> int:
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def infer_terrain(env_id: str) -> str:
    if "RoughTerrain" in env_id:
        return "rough"
    if "StairTerrain" in env_id:
        return "stairs"
    if "HillyTerrain" in env_id:
        return "hilly"
    return "flat"


def infer_method(group: str, env_id: str) -> str:
    if group in GROUP_METHOD:
        return GROUP_METHOD[group]
    if group == "symmetry":
        return "AsymSDE" if "pca0" in env_id else "Symmetry"
    if group == "single_property":
        if "Strength" in env_id:
            return "StrengthOnly"
        if "Velocity" in env_id:
            return "VelocityOnly"
        if "Stiffness" in env_id:
            return "StiffnessOnly"
        return "SingleProperty"
    if group == "latent_k":
        for k in ("K3", "K5", "K7", "K9"):
            if k in env_id:
                return k
        return "LatentK"
    return group


def valid_row(row: dict[str, str]) -> bool:
    model_path = Path(row.get("model_path", ""))
    vec_path = Path(row.get("vecnormalize_path", ""))
    env_id = row.get("env", "")
    if not env_id or not model_path.exists() or not vec_path.exists():
        return False
    # Avoid rows where the manifest env and stored model basename disagree.
    if not model_path.name.startswith(env_id):
        return False
    return True


def main():
    args = parse_args()
    with args.manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    best_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        group = row.get("group", "")
        if group not in args.groups:
            continue
        if not valid_row(row):
            continue
        seed = row.get("seed", "")
        env_id = row.get("env", "")
        key = (group, env_id, seed)
        current = best_by_key.get(key)
        if current is None or as_int(row.get("score")) > as_int(current.get("score")):
            best_by_key[key] = row

    out_rows = []
    for (group, env_id, seed), row in sorted(best_by_key.items()):
        terrain = infer_terrain(env_id)
        method = infer_method(group, env_id)
        label = f"{method}__{terrain}__seed_{seed}"
        out_rows.append(
            {
                "label": label,
                "method": method,
                "group": group,
                "terrain": terrain,
                "training_seed": seed,
                "env_id": env_id,
                "score": row.get("score", ""),
                "model_path": row.get("model_path", ""),
                "vecnormalize_path": row.get("vecnormalize_path", ""),
                "original_run_dir": row.get("original_run_dir", ""),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "method",
        "group",
        "terrain",
        "training_seed",
        "env_id",
        "score",
        "model_path",
        "vecnormalize_path",
        "original_run_dir",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {args.output}")
    print(f"Rows: {len(out_rows)}")


if __name__ == "__main__":
    main()
