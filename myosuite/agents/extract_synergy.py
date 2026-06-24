import argparse
import collections
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import myosuite  # noqa: F401 - registers MyoSuite environments
import numpy as np
from sklearn.decomposition import PCA


AGENTS_DIR = Path(__file__).resolve().parent
MYOBASE_DIR = AGENTS_DIR.parent / "envs" / "myo" / "myobase"


def _actuator_name(model, idx):
    try:
        name = model.actuator(int(idx)).name
    except AttributeError:
        name = model.actuator_id2name(int(idx))
    return name if name else f"M{idx}"


def _reset_env(env, seed=None):
    try:
        return env.reset(seed=seed)
    except TypeError:
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        return env.reset()


def _step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = result
    return obs, reward, bool(done), info


def collect_group_length_history(env_id, steps, seed, reset_interval):
    print("=== SDE spectral-basis extraction ===", flush=True)
    print(f"Environment: {env_id}", flush=True)
    print(f"Steps: {steps}", flush=True)
    print(f"Seed: {seed}", flush=True)

    env = gym.make(env_id)
    _reset_env(env, seed=seed)
    env.action_space.seed(seed)

    model = env.unwrapped.sim.model
    muscle_names = [_actuator_name(model, i) for i in range(model.nu)]

    muscle_groups = collections.OrderedDict()
    for idx, name in enumerate(muscle_names):
        base_name = name[:-2] if name.endswith(("_l", "_r")) else name
        muscle_groups.setdefault(base_name, []).append(idx)

    group_names = list(muscle_groups.keys())
    group_length_history = np.zeros((steps, len(group_names)), dtype=np.float64)

    print(
        f"Found {len(muscle_names)} actuators mapped to "
        f"{len(group_names)} bilateral muscle groups.",
        flush=True,
    )

    for t in range(steps):
        action = env.action_space.sample()
        _obs, _reward, done, _info = _step_env(env, action)

        actuator_lengths = np.asarray(env.unwrapped.sim.data.actuator_length)
        for group_idx, base_name in enumerate(group_names):
            indices = muscle_groups[base_name]
            group_length_history[t, group_idx] = float(np.mean(actuator_lengths[indices]))

        if done or ((t + 1) % reset_interval == 0):
            _reset_env(env)

        if (t + 1) % max(1, steps // 20) == 0 or (t + 1) == steps:
            print(f"Collected {t + 1}/{steps} steps.", flush=True)

    env.close()
    return muscle_names, group_names, muscle_groups, group_length_history


def fit_and_save_basis(
    env_id,
    steps,
    seed,
    k_values,
    output_dir,
    mirror_dir,
    reset_interval,
):
    max_k = max(k_values)
    output_dir = Path(output_dir).resolve()
    mirror_dir = Path(mirror_dir).resolve() if mirror_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)
    if mirror_dir:
        mirror_dir.mkdir(parents=True, exist_ok=True)

    muscle_names, group_names, muscle_groups, history = collect_group_length_history(
        env_id=env_id,
        steps=steps,
        seed=seed,
        reset_interval=reset_interval,
    )

    feature_mean = np.mean(history, axis=0)
    feature_std = np.std(history, axis=0) + 1e-8
    normalized = (history - feature_mean) / feature_std

    print(f"Fitting PCA with max_k={max_k}...", flush=True)
    pca = PCA(n_components=max_k, svd_solver="full")
    pca.fit(normalized)

    metadata = {
        "env_id": env_id,
        "steps": steps,
        "seed": seed,
        "reset_interval": reset_interval,
        "data_source": "random_rollout_nominal_model",
        "symmetry_mode": "bilateral_group_average",
        "num_actuators": len(muscle_names),
        "num_groups": len(group_names),
        "muscle_names": muscle_names,
        "group_names": group_names,
        "muscle_groups": {k: [int(i) for i in v] for k, v in muscle_groups.items()},
        "k_values": [int(k) for k in k_values],
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance_ratio": np.cumsum(
            pca.explained_variance_ratio_
        ).tolist(),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
    }

    for k in k_values:
        w_basis = pca.components_[:k].T.astype(np.float32)
        filename = f"synergy_W_basis{k}.npy"
        out_path = output_dir / filename
        np.save(out_path, w_basis)
        if mirror_dir:
            np.save(mirror_dir / filename, w_basis)
        explained = 100.0 * float(np.sum(pca.explained_variance_ratio_[:k]))
        print(f"Saved {filename}: shape={w_basis.shape}, variance={explained:.2f}%", flush=True)

    metadata_path = output_dir / "synergy_basis_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez(
        output_dir / "synergy_basis_preprocess.npz",
        feature_mean=feature_mean,
        feature_std=feature_std,
        explained_variance_ratio=pca.explained_variance_ratio_,
        components=pca.components_,
    )

    if mirror_dir:
        (mirror_dir / "synergy_basis_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        np.savez(
            mirror_dir / "synergy_basis_preprocess.npz",
            feature_mean=feature_mean,
            feature_std=feature_std,
            explained_variance_ratio=pca.explained_variance_ratio_,
            components=pca.components_,
        )

    print(f"Metadata: {metadata_path}", flush=True)
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract bilateral SDE spectral basis from MyoLeg muscle-length dynamics."
    )
    parser.add_argument("--env", default="myoLegWalk-v0", help="Nominal environment ID.")
    parser.add_argument("--steps", type=int, default=50000, help="Random rollout steps.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[3, 5, 7, 9],
        help="Latent dimensions to export.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(MYOBASE_DIR),
        help="Primary output directory used by SDE environments.",
    )
    parser.add_argument(
        "--mirror-dir",
        default=str(AGENTS_DIR),
        help="Optional mirror directory for scripts launched from agents/.",
    )
    parser.add_argument(
        "--reset-interval",
        type=int,
        default=500,
        help="Force environment reset every N random steps.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fit_and_save_basis(
        env_id=args.env,
        steps=args.steps,
        seed=args.seed,
        k_values=sorted(set(args.k)),
        output_dir=args.output_dir,
        mirror_dir=args.mirror_dir,
        reset_interval=args.reset_interval,
    )
