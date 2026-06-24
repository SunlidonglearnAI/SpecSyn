#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHOD_LABELS = {
    "fixed": "Fixed",
    "direct_t2a": "T2A",
    "full_sde": "FullSDE",
}

GROUPS = ("fixed", "direct_t2a", "full_sde")

ENV_TO_TERRAIN = {
    "myoLegWalk-v0": "flat",
    "myoLegWalkT2A-v0": "flat",
    "myoLegWalkT2Apca1-v0": "flat",
    "myoLegRoughTerrainWalk-v0": "rough",
    "myoLegRoughTerrainT2A-v0": "rough",
    "myoLegRoughTerrainT2Apca1-v0": "rough",
    "myoLegStairTerrainWalk-v0": "stairs",
    "myoLegStairTerrainT2A-v0": "stairs",
    "myoLegStairTerrainT2Apca1-v0": "stairs",
    "myoLegHillyTerrainWalk-v0": "hilly",
    "myoLegHillyTerrainWalkT2A-v0": "hilly",
    "myoLegHillyTerrainWalkT2Apca1-v0": "hilly",
}

GROUP_ENVS = {
    "fixed": {
        "flat": "myoLegWalk-v0",
        "rough": "myoLegRoughTerrainWalk-v0",
        "stairs": "myoLegStairTerrainWalk-v0",
        "hilly": "myoLegHillyTerrainWalk-v0",
    },
    "direct_t2a": {
        "flat": "myoLegWalkT2A-v0",
        "rough": "myoLegRoughTerrainT2A-v0",
        "stairs": "myoLegStairTerrainT2A-v0",
        "hilly": "myoLegHillyTerrainWalkT2A-v0",
    },
    "full_sde": {
        "flat": "myoLegWalkT2Apca1-v0",
        "rough": "myoLegRoughTerrainT2Apca1-v0",
        "stairs": "myoLegStairTerrainT2Apca1-v0",
        "hilly": "myoLegHillyTerrainWalkT2Apca1-v0",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a curated biomechanical-validation checkpoint shortlist.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("myosuite/agents/outputs_by_category/MANIFEST.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("paper/biomech_validation_model_shortlist.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("paper/biomech_validation_model_shortlist.md"),
    )
    return parser.parse_args()


def read_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_primary(rows):
    by_key = {}
    for row in rows:
        group = row["group"]
        env = row["env"]
        key = (group, env)
        if key not in by_key:
            by_key[key] = []
        by_key[key].append(row)

    selected = []
    for group in GROUPS:
        for terrain, env in GROUP_ENVS[group].items():
            candidates = by_key.get((group, env), [])
            if not candidates:
                raise RuntimeError(f"Missing shortlist candidate for {group} / {env}")

            candidates.sort(
                key=lambda row: (
                    row.get("is_recommended") == "yes",
                    int(row.get("score") or 0),
                    row.get("referenced_by_report", "") != "",
                ),
                reverse=True,
            )
            primary = candidates[0].copy()
            primary["terrain"] = terrain
            primary["method"] = METHOD_LABELS[group]
            primary["status"] = "primary"
            primary["note"] = "Highest-scored current candidate."
            primary["backup_seed"] = candidates[1]["seed"] if len(candidates) > 1 else ""
            primary["backup_model_path"] = candidates[1]["model_path"] if len(candidates) > 1 else ""

            if group == "full_sde" and terrain == "flat":
                primary["status"] = "provisional"
                primary["note"] = (
                    "Current best existing FullSDE flat checkpoint, but prior visual inspection suggested gait may be "
                    "confounded by the older flat reward weights. Use cautiously; replace with reward-v2 retrain when ready."
                )
            elif primary.get("referenced_by_report"):
                primary["note"] = "Already used by an existing validation/export report and still the top current candidate."

            selected.append(primary)
    return selected


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "group",
        "terrain",
        "status",
        "env",
        "seed",
        "score",
        "model_path",
        "vecnormalize_path",
        "original_run_dir",
        "backup_seed",
        "backup_model_path",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_md(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Biomechanical Validation Model Shortlist",
        "",
        "Primary shortlist for biomechanics export and gait-cycle analysis across `Fixed / T2A / FullSDE`.",
        "",
        "| Method | Terrain | Status | Seed | Score | Env ID | Note |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['terrain']} | {row['status']} | {row['seed']} | {row['score']} | "
            f"`{row['env']}` | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Paths",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['method']} / {row['terrain']}",
                f"- model: `{row['model_path']}`",
                f"- vecnormalize: `{row['vecnormalize_path']}`",
                f"- run dir: `{row['original_run_dir']}`",
                (
                    f"- backup seed `{row['backup_seed']}`: `{row['backup_model_path']}`"
                    if row.get("backup_model_path")
                    else "- backup: none recorded"
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rows = read_rows(args.manifest)
    rows = [row for row in rows if row.get("group") in GROUPS and row.get("env") in ENV_TO_TERRAIN]
    selected = select_primary(rows)
    write_csv(args.output_csv, selected)
    write_md(args.output_md, selected)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
