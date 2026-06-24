#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


EXPERIMENT_GROUPS = {
    "fixed": [
        "myoLegWalk-v0",
        "myoLegRoughTerrainWalk-v0",
        "myoLegStairTerrainWalk-v0",
        "myoLegHillyTerrainWalk-v0",
    ],
    "direct_t2a": [
        "myoLegWalkT2A-v0",
        "myoLegRoughTerrainT2A-v0",
        "myoLegStairTerrainT2A-v0",
        "myoLegHillyTerrainWalkT2A-v0",
    ],
    "full_sde": [
        "myoLegWalkT2Apca1-v0",
        "myoLegRoughTerrainT2Apca1-v0",
        "myoLegStairTerrainT2Apca1-v0",
        "myoLegHillyTerrainWalkT2Apca1-v0",
    ],
    "single_property": [
        "myoLegWalkT2AStrength-v0",
        "myoLegWalkT2AVelocity-v0",
        "myoLegWalkT2AStiffness-v0",
    ],
    "symmetry": [
        "myoLegWalkT2Apca0-v0",
        "myoLegRoughTerrainT2Apca0-v0",
    ],
    "latent_k": [
        "myoLegWalkT2ApcaK3-v0",
        "myoLegWalkT2ApcaK5-v0",
        "myoLegWalkT2ApcaK7-v0",
        "myoLegWalkT2ApcaK9-v0",
    ],
}

LEGACY_ENV_ALIASES = {
    "myoLegWalkT2Apca-v0": "myoLegWalkT2Apca1-v0",
    "myoLegRoughTerrainT2Apca-v0": "myoLegRoughTerrainT2Apca1-v0",
    "myoLegStairTerrainT2Apca-v0": "myoLegStairTerrainT2Apca1-v0",
    "myoLegHillyTerrainWalkT2Apca-v0": "myoLegHillyTerrainWalkT2Apca1-v0",
    "myoLegWalkT2A1-v0": "myoLegWalkT2Apca1-v0",
    "myoLegRoughTerrainT2A1-v0": "myoLegRoughTerrainT2Apca1-v0",
    "myoLegStairTerrainT2A1-v0": "myoLegStairTerrainT2Apca1-v0",
    "myoLegHillyTerrainWalkT2A1-v0": "myoLegHillyTerrainWalkT2Apca1-v0",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit trained RL model candidates for CBS/SDE experiments.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def env_to_group():
    out = {}
    for group, envs in EXPERIMENT_GROUPS.items():
        for env in envs:
            out[env] = group
    return out


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def first_yaml_scalar(text: str, key: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text)
    return match.group(1).strip() if match else ""


def infer_env_from_model_name(model_path: Path) -> str:
    name = model_path.name
    for suffix in ("_PPO_model.zip", "_SAC_model.zip", "_cpg_model.zip"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name.removesuffix(".zip")


def find_vec_path(model_path: Path, env_id: str) -> Path | None:
    parent = model_path.parent
    candidates = [
        parent / f"{env_id}_PPO_env",
        parent / f"{env_id}_SAC_env",
        parent / f"{env_id}_cpg_env",
    ]
    candidates.extend(sorted(p for p in parent.glob(f"{env_id}_*_env") if p.is_file()))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def report_model_refs(root: Path):
    refs = defaultdict(list)
    for report in sorted(root.glob("paper/biomech_validation_*/*/REPORT.md")):
        text = read_text(report)
        match = re.search(r"^- Model:\s+`(.+?)`", text, flags=re.MULTILINE)
        if not match:
            continue
        raw = match.group(1)
        path = Path(raw)
        if not path.exists():
            try:
                path = root / path.relative_to(root)
            except ValueError:
                pass
        refs[str(path)].append(str(report.relative_to(root)))
    return refs


def score_candidate(row):
    score = 0
    reasons = []
    if row["referenced_by_report"]:
        score += 100
        reasons.append("used_in_existing_biomech_report")
    if row["has_vecnormalize"] == "yes":
        score += 20
        reasons.append("has_vecnormalize")
    if row["matched_current_matrix"] == "yes":
        score += 15
        reasons.append("exact_current_env")
    elif row["canonical_env"] in env_to_group():
        score += 8
        reasons.append("legacy_alias_for_current_env")
    if "/outputs/pass/" in row["model_path"]:
        score += 6
        reasons.append("under_outputs_pass")
    if "/outputs/walk_ori/" in row["model_path"]:
        score += 5
        reasons.append("under_walk_ori")
    if row["model_size_mb"]:
        size_mb = float(row["model_size_mb"])
        if size_mb >= 3.0:
            score += 4
            reasons.append("nontrivial_zip_size")
    return score, ";".join(reasons)


def collect_rows(root: Path):
    group_by_env = env_to_group()
    refs = report_model_refs(root)
    rows = []
    outputs = root / "myosuite" / "agents" / "outputs"
    for model_path in sorted(outputs.rglob("*_model.zip")):
        env_from_name = infer_env_from_model_name(model_path)
        config_text = read_text(model_path.parent / "job_config.json")
        if not config_text:
            config_text = read_text(model_path.parent / ".hydra" / "config.yaml")
        config_env = first_yaml_scalar(config_text, "env")
        seed = first_yaml_scalar(config_text, "seed")
        total_timesteps = first_yaml_scalar(config_text, "total_timesteps")
        env_id = config_env or env_from_name
        canonical_env = LEGACY_ENV_ALIASES.get(env_id, env_id)
        vec_path = find_vec_path(model_path, env_from_name) or find_vec_path(model_path, env_id)
        progress_paths = sorted(model_path.parent.glob("results_*/progress.csv"))
        row = {
            "group": group_by_env.get(canonical_env, ""),
            "env_id": env_id,
            "canonical_env": canonical_env,
            "matched_current_matrix": "yes" if env_id in group_by_env else "no",
            "seed": seed,
            "total_timesteps": total_timesteps,
            "model_size_mb": f"{model_path.stat().st_size / (1024 * 1024):.2f}",
            "has_vecnormalize": "yes" if vec_path else "no",
            "has_progress_csv": "yes" if progress_paths else "no",
            "progress_csv_nonempty": "yes" if any(p.stat().st_size > 0 for p in progress_paths) else "no",
            "referenced_by_report": ";".join(refs.get(str(model_path), [])),
            "model_path": str(model_path),
            "vecnormalize_path": str(vec_path) if vec_path else "",
        }
        row["score"], row["score_reasons"] = score_candidate(row)
        rows.append(row)
    rows.sort(key=lambda r: (r["canonical_env"], -int(r["score"]), r["model_path"]))
    return rows


def write_summary(rows, summary_path: Path):
    grouped = defaultdict(list)
    for row in rows:
        if row["group"]:
            grouped[row["canonical_env"]].append(row)

    lines = [
        "# RL Model Candidate Audit",
        "",
        "This file is generated by `scripts/audit_rl_model_candidates.py`.",
        "",
        "## Recommended Candidates",
        "",
        "| Group | Env | Score | Seed | Model | Evidence |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for group, envs in EXPERIMENT_GROUPS.items():
        for env in envs:
            candidates = grouped.get(env, [])
            if not candidates:
                lines.append(f"| {group} | {env} |  |  | MISSING | no candidate found |")
                continue
            best = candidates[0]
            evidence = best["score_reasons"] or "candidate_exists"
            lines.append(
                f"| {group} | {env} | {best['score']} | {best['seed']} | "
                f"`{best['model_path']}` | {evidence} |"
            )

    lines.extend(["", "## Coverage", ""])
    for group, envs in EXPERIMENT_GROUPS.items():
        found = sum(1 for env in envs if grouped.get(env))
        lines.append(f"- {group}: {found}/{len(envs)} envs have at least one candidate")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    root = args.root.resolve()
    output = args.output or root / "paper" / "rl_model_candidate_audit.csv"
    summary = args.summary or root / "paper" / "rl_model_candidate_audit.md"
    rows = collect_rows(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "group",
            "env_id",
            "canonical_env",
            "matched_current_matrix",
            "seed",
            "total_timesteps",
            "score",
            "score_reasons",
            "model_size_mb",
            "has_vecnormalize",
            "has_progress_csv",
            "progress_csv_nonempty",
            "referenced_by_report",
            "model_path",
            "vecnormalize_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_summary(rows, summary)
    print(f"Wrote {len(rows)} candidates to {output}")
    print(f"Wrote summary to {summary}")


if __name__ == "__main__":
    main()
