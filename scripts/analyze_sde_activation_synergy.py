#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from segment_gait_cycles import filter_by_quality, heel_strikes, infer_condition, resample, rollout_files


ANATOMICAL_GROUPS = {
    "ADD": ["addbrev", "addlong", "addmag"],
    "HFL": ["iliacus", "psoas"],
    "GLU": ["glmax", "glmed", "glmin"],
    "HAM": ["bflh", "bfsh", "semimem", "semiten"],
    "RF": ["recfem"],
    "VAS": ["vasint", "vaslat", "vasmed"],
    "GAS": ["gaslat", "gasmed"],
    "SOL": ["soleus"],
    "TA": ["tibant", "edl", "ehl"],
    "FLEX": ["fdl", "fhl"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Quantify activation synergy alignment with SDE basis.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[Path("paper/biomech_rollouts_recommended/flat")],
    )
    parser.add_argument("--basis", type=Path, default=Path("myosuite/agents/synergy_W_basis5.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/biomech_rollouts_recommended/sde_activation_synergy"))
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--min-cycle-frames", type=int, default=25)
    parser.add_argument("--max-cycle-frames", type=int, default=260)
    parser.add_argument("--contact-column", choices=["left", "right"], default="left")
    parser.add_argument("--quality-status", choices=["all", "accepted"], default="accepted")
    return parser.parse_args()


def base_name(name: str):
    return name[:-2] if name.endswith(("_l", "_r")) else name


def group_indices(names):
    groups = {}
    for idx, name in enumerate(names):
        groups.setdefault(base_name(name), []).append(idx)
    return groups


def anatomical_indices(names):
    lower = [name.lower() for name in names]
    out = {}
    for group, patterns in ANATOMICAL_GROUPS.items():
        out[group] = [idx for idx, name in enumerate(lower) if any(pattern in name for pattern in patterns)]
    return out


def accepted_cycle_matrices(path: Path, samples: int, min_len: int, max_len: int, contact_column: str):
    data = np.load(path, allow_pickle=True)
    contact = data["contact"]
    side_idx = 0 if contact_column == "left" else 1
    strikes = heel_strikes(contact[:, side_idx], min_interval=min_len)
    activation = data["muscle_activation"]
    names = [str(name) for name in data["actuator_names"].tolist()]
    cycles = []
    for start, stop in zip(strikes[:-1], strikes[1:]):
        length = int(stop - start)
        if length < min_len or length > max_len:
            continue
        matrix = np.vstack([resample(activation[start:stop, idx], samples) for idx in range(activation.shape[1])])
        cycles.append(matrix)
    return cycles, names


def group_average_matrix(matrix_80: np.ndarray, names):
    groups = group_indices(names)
    ordered = list(groups.keys())
    out = np.vstack([np.nanmean(matrix_80[idxs, :], axis=0) for idxs in groups.values()])
    return out, ordered


def projection_fraction(group_matrix: np.ndarray, basis: np.ndarray):
    # Basis columns are PCA components and should already be orthonormal, but QR keeps this robust.
    q, _ = np.linalg.qr(basis)
    fractions = []
    for col in range(group_matrix.shape[1]):
        x = group_matrix[:, col].astype(float)
        x = x - np.nanmean(x)
        denom = float(np.dot(x, x))
        if denom <= 1e-12:
            continue
        proj = q @ (q.T @ x)
        fractions.append(float(np.dot(proj, proj) / denom))
    return float(np.nanmean(fractions)) if fractions else float("nan")


def mean_within_group_corr(matrix_80: np.ndarray, names):
    anat = anatomical_indices(names)
    vals = []
    by_group = {}
    for group, idxs in anat.items():
        if len(idxs) < 2:
            continue
        curves = matrix_80[idxs, :]
        row_std = np.nanstd(curves, axis=1)
        keep = row_std > 1e-8
        curves = curves[keep]
        if curves.shape[0] < 2:
            continue
        corr = np.corrcoef(curves)
        upper = corr[np.triu_indices_from(corr, k=1)]
        upper = upper[np.isfinite(upper)]
        if upper.size:
            value = float(np.nanmean(upper))
            by_group[group] = value
            vals.append(value)
    return (float(np.nanmean(vals)) if vals else float("nan")), by_group


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = ["condition", "n_cycles", "sde_basis_projection_fraction_mean", "within_anatomical_group_corr_mean"]
    fieldnames = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    basis = np.load(args.basis).astype(float)
    files = filter_by_quality(rollout_files(args.inputs), args.quality_status)
    rows = []
    per_condition = {}
    for path in files:
        condition = infer_condition(path)
        cycles, names = accepted_cycle_matrices(
            path,
            args.samples,
            args.min_cycle_frames,
            args.max_cycle_frames,
            args.contact_column,
        )
        for matrix_80 in cycles:
            group_matrix, group_names = group_average_matrix(matrix_80, names)
            if group_matrix.shape[0] != basis.shape[0]:
                continue
            proj = projection_fraction(group_matrix, basis)
            corr, by_group = mean_within_group_corr(matrix_80, names)
            row = {
                "condition": condition,
                "rollout_file": str(path),
                "sde_basis_projection_fraction": proj,
                "within_anatomical_group_corr": corr,
            }
            row.update({f"corr_{group}": value for group, value in by_group.items()})
            rows.append(row)
            per_condition.setdefault(condition, []).append(row)

    summary = []
    for condition, items in sorted(per_condition.items()):
        row = {"condition": condition, "n_cycles": len(items)}
        for key in ["sde_basis_projection_fraction", "within_anatomical_group_corr"]:
            vals = np.asarray([float(item.get(key, np.nan)) for item in items], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(vals))
            row[f"{key}_std"] = float(np.nanstd(vals))
        group_keys = sorted({key for item in items for key in item if key.startswith("corr_")})
        for key in group_keys:
            vals = np.asarray([float(item.get(key, np.nan)) for item in items], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(vals))
        summary.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "activation_synergy_by_cycle.csv")
    write_csv(summary, args.output_dir / "activation_synergy_summary.csv")
    print(f"Wrote {args.output_dir / 'activation_synergy_by_cycle.csv'}")
    print(f"Wrote {args.output_dir / 'activation_synergy_summary.csv'}")


if __name__ == "__main__":
    main()
