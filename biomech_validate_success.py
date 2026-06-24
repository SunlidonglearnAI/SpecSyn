"""Biomechanical validation for trained VIB/distilled MyoLeg policies.

The script reloads a success checkpoint with the same observation wrapper and
VecNormalize statistics used during training, rolls out the policy, and exports
kinematic, gait-timing, and muscle-activation summaries.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "glfw")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "myosuite" / "agents").is_dir():
            return candidate
    return start


ROOT = find_repo_root(Path(__file__).resolve().parent)
AGENTS_DIR = ROOT / "myosuite" / "agents"
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import myosuite  # noqa: E402,F401
from myosuite.utils import gym  # noqa: E402

try:
    from sb3_job_script_new import FatDictObservationWrapper  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - optional wrapper from older experiments
    FatDictObservationWrapper = None


@dataclass(frozen=True)
class Condition:
    label: str
    env_id: str
    model_path: Path
    vecnormalize_path: Path


JOINTS = [
    "hip_flexion_r",
    "hip_flexion_l",
    "hip_adduction_r",
    "hip_adduction_l",
    "hip_rotation_r",
    "hip_rotation_l",
    "knee_angle_r",
    "knee_angle_l",
    "ankle_angle_r",
    "ankle_angle_l",
    "subtalar_angle_r",
    "subtalar_angle_l",
]

SAGITTAL_JOINTS = [
    ("Hip flexion", "hip_flexion_r", "hip_flexion_l"),
    ("Knee flexion", "knee_angle_r", "knee_angle_l"),
    ("Ankle angle", "ankle_angle_r", "ankle_angle_l"),
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

TERRAIN_NAMES = {0: "flat", 1: "rough", 2: "hilly", 3: "stairs"}
KNOWN_TERRAINS = {"flat", "rough", "hilly", "stairs", "mixed"}


def quat_to_roll_pitch_yaw(q: np.ndarray) -> Tuple[float, float, float]:
    """Convert MuJoCo root quaternion [w, x, y, z] to roll, pitch, yaw."""
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def maybe_wrap_env(env, mode: str):
    if mode == "none":
        return env
    if mode == "fat":
        if FatDictObservationWrapper is None:
            raise ImportError("FatDictObservationWrapper requested, but sb3_job_script_new.py is not available.")
        return FatDictObservationWrapper(env)
    if mode == "auto" and FatDictObservationWrapper is not None:
        return FatDictObservationWrapper(env)
    return env


def load_saved_obs_dim(vecnormalize_path: Path) -> int | None:
    try:
        with vecnormalize_path.open("rb") as handle:
            saved = pickle.load(handle)
    except Exception:
        return None

    observation_space = getattr(saved, "observation_space", None)
    shape = getattr(observation_space, "shape", None)
    if shape and len(shape) == 1:
        return int(shape[0])

    obs_rms = getattr(saved, "obs_rms", None)
    mean = getattr(obs_rms, "mean", None)
    if mean is not None and getattr(mean, "shape", None):
        return int(mean.shape[0])
    return None


def align_env_obs_with_saved_stats(env, expected_dim: int | None):
    if expected_dim is None:
        return

    raw = env.unwrapped
    current_dim = int(np.prod(raw.observation_space.shape))
    if current_dim == expected_dim:
        return

    design_dim = 0
    if hasattr(raw, "current_scales"):
        design_dim = int(np.prod(np.asarray(raw.current_scales).shape))
    current_keys = list(getattr(raw, "obs_keys", []))
    updated_keys = list(current_keys)

    if design_dim and expected_dim == current_dim + design_dim and "design_params" not in updated_keys:
        updated_keys.append("design_params")
    elif (
        design_dim
        and expected_dim == current_dim + design_dim + 1
        and "design_params" not in updated_keys
    ):
        updated_keys.extend(["design_params", "is_design_phase"])
    elif expected_dim == current_dim + 1 and "is_design_phase" not in updated_keys:
        updated_keys.append("is_design_phase")

    if updated_keys == current_keys:
        return

    raw.obs_keys = updated_keys
    raw.key_idx = {}
    raw.ordered_obs_keys = None
    raw.initialized = False
    obs = raw.get_obs()
    if obs.shape[0] != expected_dim:
        raise ValueError(
            f"Failed to align observation space for {type(raw).__name__}: "
            f"expected {expected_dim}, got {obs.shape[0]} after obs_keys patch."
        )

    box = gym.spaces.Box(
        low=-10.0 * np.ones(obs.shape[0], dtype=np.float32),
        high=10.0 * np.ones(obs.shape[0], dtype=np.float32),
        dtype=np.float32,
    )
    raw.observation_space = box
    env.observation_space = box


def make_eval_env(env_id: str, vecnormalize_path: Path, obs_wrapper: str = "auto"):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    holder = {}
    expected_dim = load_saved_obs_dim(vecnormalize_path)

    def _make():
        env = gym.make(env_id)
        align_env_obs_with_saved_stats(env, expected_dim)
        wrapped = maybe_wrap_env(env, obs_wrapper)
        holder["wrapped"] = wrapped
        holder["raw"] = env.unwrapped
        return wrapped

    dummy = DummyVecEnv([_make])
    vec = VecNormalize.load(str(vecnormalize_path), dummy)
    vec.training = False
    vec.norm_reward = False
    return vec, holder


def normalize_single_obs(vec: VecNormalize, obs):
    if isinstance(obs, dict):
        batched = {k: np.asarray(v)[None, ...] for k, v in obs.items()}
    else:
        batched = np.asarray(obs)[None, ...]
    return vec.normalize_obs(batched)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def latest_success_dir() -> Path:
    candidates = sorted((AGENTS_DIR / "outputs").glob("*/success"))
    if not candidates:
        raise FileNotFoundError("No myosuite/agents/outputs/*/success directory found.")
    return candidates[-1]


def read_job_env(success_dir: Path) -> str:
    job_path = success_dir / "job_config.json"
    if not job_path.exists():
        return "myoLegLongTerrainWalkT2Apcanew1-v0"
    text = job_path.read_text()
    for line in text.splitlines():
        if line.strip().startswith("env:"):
            return line.split(":", 1)[1].strip()
    return "myoLegLongTerrainWalkT2Apcanew1-v0"


def model_and_vec_paths(success_dir: Path) -> Tuple[Path, Path]:
    model = success_dir / "myoLegLongTerrainWalkT2Apcanew1-v0_cpg_model.zip"
    vec = success_dir / "myoLegLongTerrainWalkT2Apcanew1-v0_cpg_env"
    if not model.exists():
        models = sorted(success_dir.glob("*_cpg_model.zip"))
        if not models:
            raise FileNotFoundError(f"No *_cpg_model.zip found in {success_dir}")
        model = models[-1]
    if not vec.exists():
        vecs = sorted(p for p in success_dir.glob("*_cpg_env") if p.is_file())
        if not vecs:
            raise FileNotFoundError(f"No *_cpg_env VecNormalize file found in {success_dir}")
        vec = vecs[-1]
    return model, vec


def infer_model_and_vec(env_id: str, model_dir: Path, algo: str = "PPO") -> Tuple[Path, Path]:
    model = model_dir / f"{env_id}_{algo}_model.zip"
    vec = model_dir / f"{env_id}_{algo}_env"
    if not model.exists() and (model_dir / "best_model.zip").exists():
        model = model_dir / "best_model.zip"
    if not model.exists():
        models = sorted(model_dir.glob(f"{env_id}_*_model.zip"))
        if models:
            model = models[-1]
    if not vec.exists():
        vec_candidates = sorted(p for p in model_dir.glob(f"{env_id}_*_env") if p.is_file())
        if vec_candidates:
            vec = vec_candidates[-1]
    if not model.exists():
        raise FileNotFoundError(f"No model found for {env_id} in {model_dir}")
    if not vec.exists():
        raise FileNotFoundError(f"No VecNormalize file found for {env_id} in {model_dir}")
    return model, vec


def parse_condition(text: str, algo: str = "PPO") -> Condition:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) not in (3, 4):
        raise ValueError(
            "--condition must be 'LABEL,ENV_ID,MODEL_DIR' or "
            "'LABEL,ENV_ID,MODEL_ZIP,VECNORMALIZE_FILE'"
        )
    label, env_id = parts[0], parts[1]
    if len(parts) == 3:
        model_path, vec_path = infer_model_and_vec(env_id, Path(parts[2]).expanduser().resolve(), algo=algo)
    else:
        model_path = Path(parts[2]).expanduser().resolve()
        vec_path = Path(parts[3]).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        if not vec_path.exists():
            raise FileNotFoundError(f"VecNormalize file does not exist: {vec_path}")
    return Condition(label=label, env_id=env_id, model_path=model_path, vecnormalize_path=vec_path)


def default_success_condition(success_dir: Path, env_id: str | None = None) -> Condition:
    model_path, vec_path = model_and_vec_paths(success_dir)
    resolved_env_id = env_id or read_job_env(success_dir)
    return Condition(
        label=success_dir.name if success_dir.name != "success" else success_dir.parent.name,
        env_id=resolved_env_id,
        model_path=model_path,
        vecnormalize_path=vec_path,
    )


def get_joint_angle(raw_env, joint_name: str) -> float:
    m = raw_env.sim.model
    if m.joint_name2id(joint_name) < 0:
        return float("nan")
    jid = m.joint_name2id(joint_name)
    qadr = m.jnt_qposadr[jid]
    return float(raw_env.sim.data.qpos[qadr])


def get_joint_velocity(raw_env, joint_name: str) -> float:
    m = raw_env.sim.model
    if m.joint_name2id(joint_name) < 0:
        return float("nan")
    jid = m.joint_name2id(joint_name)
    dadr = m.jnt_dofadr[jid]
    return float(raw_env.sim.data.qvel[dadr])


def get_actuator_names(raw_env) -> List[str]:
    names = []
    for idx in range(raw_env.sim.model.nu):
        try:
            names.append(raw_env.sim.model.actuator(int(idx)).name)
        except AttributeError:
            names.append(raw_env.sim.model.actuator_id2name(int(idx)))
    return names


def group_indices(names: List[str], side: str) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    suffix = f"_{side}"
    lower_names = [n.lower() for n in names]
    for group, patterns in MUSCLE_GROUP_PATTERNS.items():
        idxs = [
            i
            for i, name in enumerate(lower_names)
            if name.endswith(suffix) and any(p in name for p in patterns)
        ]
        out[group] = idxs
    return out


def hfield_height(raw_env, x: float, y: float) -> float:
    model = raw_env.sim.model
    if model.nhfield < 1:
        return 0.0
    nrow = int(model.hfield_nrow[0])
    ncol = int(model.hfield_ncol[0])
    size_x = float(model.hfield_size[0, 0])
    size_y = float(model.hfield_size[0, 1])
    z_top = float(model.hfield_size[0, 2])
    z_bottom = float(model.hfield_size[0, 3])
    cx = int(np.clip(((x + size_x) / (2 * size_x) * ncol), 0, ncol - 1))
    cy = int(np.clip(((y + size_y) / (2 * size_y) * nrow), 0, nrow - 1))
    return float(model.hfield_data[cy * ncol + cx] * (z_top + z_bottom) - z_bottom)


def _model_name(model, kind: str, idx: int) -> str:
    try:
        return getattr(model, kind)(idx).name
    except AttributeError:
        return getattr(model, f"{kind}_id2name")(idx)


def contact_state(raw_env) -> Tuple[bool, bool]:
    model = raw_env.sim.model
    data = raw_env.sim.data
    left = False
    right = False
    ground_names = {"terrain", "floor", "ground"}
    foot_bodies = {
        "l": {"toes_l", "calcn_l", "talus_l"},
        "r": {"toes_r", "calcn_r", "talus_r"},
    }

    for i in range(data.ncon):
        c = data.contact[i]
        geoms = [int(c.geom1), int(c.geom2)]
        gnames = [_model_name(model, "geom", g) for g in geoms]
        bnames = [_model_name(model, "body", int(model.geom_bodyid[g])) for g in geoms]

        for side, bodies in foot_bodies.items():
            side_hit = False
            for foot_slot, ground_slot in [(0, 1), (1, 0)]:
                foot_body = bnames[foot_slot]
                ground_geom = gnames[ground_slot]
                ground_body = bnames[ground_slot]
                if foot_body in bodies and (ground_geom in ground_names or ground_body == "world"):
                    side_hit = True
            if side == "l":
                left = left or side_hit
            else:
                right = right or side_hit
    return left, right


def foot_state(raw_env):
    model = raw_env.sim.model
    data = raw_env.sim.data
    talus_l = model.body_name2id("talus_l")
    talus_r = model.body_name2id("talus_r")
    pelvis = model.body_name2id("pelvis")
    lpos = data.body_xpos[talus_l].copy()
    rpos = data.body_xpos[talus_r].copy()
    ppos = data.body_xpos[pelvis].copy()
    lh = lpos[2] - hfield_height(raw_env, lpos[0], lpos[1])
    rh = rpos[2] - hfield_height(raw_env, rpos[0], rpos[1])
    contact_l, contact_r = contact_state(raw_env)
    # Height fallback for pathological contact-buffer gaps.
    if not contact_l:
        contact_l = lh < 0.075
    if not contact_r:
        contact_r = rh < 0.075
    return lpos, rpos, ppos, lh, rh, contact_l, contact_r


def record_frame(raw_env, action, reward, done, info) -> Dict[str, object]:
    qpos = raw_env.sim.data.qpos.copy()
    qvel = raw_env.sim.data.qvel.copy()
    lpos, rpos, ppos, lh, rh, cl, cr = foot_state(raw_env)
    roll, pitch, yaw = quat_to_roll_pitch_yaw(qpos[3:7])
    joint_values = {j: get_joint_angle(raw_env, j) for j in JOINTS}
    joint_velocities = {j: get_joint_velocity(raw_env, j) for j in JOINTS}
    names = get_actuator_names(raw_env)
    act = raw_env.sim.data.act.copy()
    length = raw_env.sim.data.actuator_length.copy()
    force = raw_env.sim.data.actuator_force.copy()
    velocity = raw_env.sim.data.actuator_velocity.copy()
    scales = getattr(raw_env, "current_scales", None)
    if scales is None:
        scales = np.ones((raw_env.sim.model.nu, 3), dtype=np.float32)
    else:
        scales = np.asarray(scales, dtype=np.float32).copy()
    terrain_attr = str(getattr(raw_env, "terrain", ""))
    terrain_idx = int(getattr(raw_env, "_current_terrain_type", -1))
    if terrain_attr in KNOWN_TERRAINS:
        terrain_name = terrain_attr
    else:
        terrain_name = TERRAIN_NAMES.get(terrain_idx, f"unknown_{terrain_idx}")
    return {
        "time": float(raw_env.sim.data.time),
        "qpos": qpos,
        "qvel": qvel,
        "pelvis_pos": ppos,
        "foot_l": lpos,
        "foot_r": rpos,
        "foot_l_rel_height": float(lh),
        "foot_r_rel_height": float(rh),
        "contact_l": bool(cl),
        "contact_r": bool(cr),
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "joints": joint_values,
        "joint_velocities": joint_velocities,
        "act": act,
        "length": length,
        "force": force,
        "actuator_velocity": velocity,
        "scales": scales,
        "action": np.asarray(action).copy(),
        "reward": float(reward),
        "done": bool(done),
        "terrain_idx": terrain_idx,
        "terrain_attr": terrain_attr,
        "terrain_name": terrain_name,
        "info_phase": str(info.get("phase", "")) if isinstance(info, dict) else "",
        "actuator_names": names,
    }


def adapt_action_for_env(raw_env, action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    env_action_dim = int(np.prod(raw_env.action_space.shape))
    if action.shape[0] == env_action_dim:
        return action

    control_dim = getattr(raw_env, "control_dim", None)
    design_dim = getattr(raw_env, "design_dim", None)
    design_steps = int(getattr(raw_env, "design_steps", 0))
    design_step_counter = int(getattr(raw_env, "design_step_counter", design_steps))
    if control_dim is None or design_dim is None:
        return action

    control_dim = int(control_dim)
    design_dim = int(design_dim)
    legacy_dim = max(control_dim, design_dim)
    if action.shape[0] != legacy_dim or env_action_dim != control_dim + design_dim:
        return action

    expanded = np.zeros(env_action_dim, dtype=np.float32)
    if design_step_counter < design_steps:
        expanded[control_dim : control_dim + design_dim] = action[:design_dim]
        expanded[: min(control_dim, action.shape[0])] = action[: min(control_dim, action.shape[0])]
    else:
        expanded[:control_dim] = action[:control_dim]
    return expanded


def rollout(
    model,
    vec_env,
    holder,
    max_steps: int,
    deterministic: bool,
    terrain: str | None = None,
    seed: int | None = None,
    reset_type: str | None = None,
):
    wrapped = holder["wrapped"]
    raw = holder["raw"]
    if seed is not None:
        seed_everything(seed)
        for env_obj in (wrapped, raw):
            if hasattr(env_obj, "seed"):
                try:
                    env_obj.seed(seed)
                except TypeError:
                    pass
    if terrain is not None:
        raw.terrain = terrain
    if reset_type and reset_type != "env-default" and hasattr(raw, "reset_type"):
        raw.reset_type = reset_type
    reset_out = wrapped.reset()
    obs_raw = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    obs = normalize_single_obs(vec_env, obs_raw)
    frames = []
    total_reward = 0.0
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        env_action = adapt_action_for_env(raw, action[0])
        step_out = wrapped.step(env_action)
        obs_raw, reward, term, trunc, info = step_out
        done = bool(term or trunc)
        obs = normalize_single_obs(vec_env, obs_raw)
        reward = float(reward)
        total_reward += reward
        frames.append(record_frame(raw, env_action, reward, done, info))
        if done:
            break
    return frames, total_reward


def heel_strikes(contact: np.ndarray, min_interval: int = 45) -> np.ndarray:
    c = contact.astype(bool)
    edges = np.where((~c[:-1]) & c[1:])[0] + 1
    if edges.size == 0:
        return edges
    keep = [int(edges[0])]
    for edge in edges[1:]:
        if int(edge) - keep[-1] >= min_interval:
            keep.append(int(edge))
    return np.asarray(keep, dtype=int)


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def compute_metrics(frames: List[Dict[str, object]], max_steps: int | None = None) -> Dict[str, object]:
    if not frames:
        return {}

    t = np.arange(len(frames), dtype=float) * 0.01
    # Prefer env time when available and monotonic.
    times = np.asarray([f["time"] for f in frames], dtype=float)
    if np.all(np.diff(times) >= -1e-9) and np.nanmax(times) > 0:
        t = times - times[0]
    duration = max(float(t[-1] - t[0]), 1e-9)

    pelvis = np.vstack([f["pelvis_pos"] for f in frames])
    contact_l = np.asarray([f["contact_l"] for f in frames], dtype=bool)
    contact_r = np.asarray([f["contact_r"] for f in frames], dtype=bool)
    hs_l = heel_strikes(contact_l)
    hs_r = heel_strikes(contact_r)
    n_steps = int(len(hs_l) + len(hs_r))
    distance = float(abs(pelvis[-1, 1] - pelvis[0, 1]))
    mean_speed = distance / duration
    cadence = 60.0 * n_steps / duration if duration > 0 else float("nan")
    step_length = distance / n_steps if n_steps > 0 else float("nan")

    joint_summary = {}
    for j in JOINTS:
        vals = np.asarray([f["joints"][j] for f in frames], dtype=float)
        vels = np.asarray([f["joint_velocities"][j] for f in frames], dtype=float)
        vals_deg = np.rad2deg(vals)
        vels_deg = np.rad2deg(vels)
        joint_summary[j] = {
            "mean_deg": safe_mean(vals_deg),
            "min_deg": float(np.nanmin(vals_deg)),
            "max_deg": float(np.nanmax(vals_deg)),
            "rom_deg": float(np.nanmax(vals_deg) - np.nanmin(vals_deg)),
            "vel_rms_deg_s": float(np.sqrt(np.nanmean(vels_deg**2))),
            "vel_peak_abs_deg_s": float(np.nanmax(np.abs(vels_deg))),
        }

    foot_l = np.vstack([f["foot_l"] for f in frames])
    foot_r = np.vstack([f["foot_r"] for f in frames])
    lateral_sep = foot_l[:, 0] - foot_r[:, 0]
    reference_sign = np.sign(np.nanmedian(lateral_sep[: max(5, min(50, len(lateral_sep)))]))
    if reference_sign == 0:
        reference_sign = 1.0
    leg_crossing_rate = float(np.mean(np.sign(lateral_sep) != reference_sign))
    close_lateral_rate = float(np.mean(np.abs(lateral_sep) < 0.03))
    clearance = np.vstack([[f["foot_l_rel_height"], f["foot_r_rel_height"]] for f in frames])

    terrain_counts: Dict[str, int] = {}
    for f in frames:
        name = str(f.get("terrain_name") or TERRAIN_NAMES.get(int(f["terrain_idx"]), f"unknown_{f['terrain_idx']}"))
        terrain_counts[name] = terrain_counts.get(name, 0) + 1

    reward_total = float(np.sum([f["reward"] for f in frames]))
    final_scales = np.asarray(frames[-1].get("scales"), dtype=float)
    if final_scales.ndim == 2 and final_scales.shape[1] >= 3:
        sigma_values = final_scales[:, 0]
        nu_values = final_scales[:, 1]
        kappa_values = final_scales[:, 2]
    else:
        sigma_values = np.asarray([np.nan], dtype=float)
        nu_values = np.asarray([np.nan], dtype=float)
        kappa_values = np.asarray([np.nan], dtype=float)

    survived_horizon = int(max_steps is not None and len(frames) >= max_steps)
    fell_or_failed = int(max_steps is not None and len(frames) < max_steps)
    metrics = {
        "n_frames": len(frames),
        "duration_s": duration,
        "return": reward_total,
        "distance_m": distance,
        "mean_forward_speed_mps": mean_speed,
        "cadence_steps_per_min": cadence,
        "step_count": n_steps,
        "step_length_m": step_length,
        "duty_factor_l": float(np.mean(contact_l)),
        "duty_factor_r": float(np.mean(contact_r)),
        "duty_factor_asymmetry": float(abs(np.mean(contact_l) - np.mean(contact_r))),
        "double_support_fraction": float(np.mean(contact_l & contact_r)),
        "no_contact_fraction": float(np.mean((~contact_l) & (~contact_r))),
        "leg_crossing_rate": leg_crossing_rate,
        "close_lateral_feet_rate": close_lateral_rate,
        "foot_clearance_std_l_m": float(np.std(clearance[:, 0])),
        "foot_clearance_std_r_m": float(np.std(clearance[:, 1])),
        "pelvis_lateral_sway_std_m": float(np.std(pelvis[:, 0])),
        "pelvis_height_mean_m": float(np.mean(pelvis[:, 2])),
        "pelvis_height_std_m": float(np.std(pelvis[:, 2])),
        "pelvis_height_min_m": float(np.min(pelvis[:, 2])),
        "pelvis_roll_rms_deg": float(np.sqrt(np.mean(np.rad2deg([f["roll"] for f in frames]) ** 2))),
        "pelvis_pitch_rms_deg": float(np.sqrt(np.mean(np.rad2deg([f["pitch"] for f in frames]) ** 2))),
        "survived_horizon": survived_horizon,
        "fell_or_failed": fell_or_failed,
        "sigma_mean": float(np.nanmean(sigma_values)),
        "sigma_std": float(np.nanstd(sigma_values)),
        "nu_mean": float(np.nanmean(nu_values)),
        "nu_std": float(np.nanstd(nu_values)),
        "kappa_mean": float(np.nanmean(kappa_values)),
        "kappa_std": float(np.nanstd(kappa_values)),
        "heel_strikes_l": hs_l.tolist(),
        "heel_strikes_r": hs_r.tolist(),
        "terrain_counts": terrain_counts,
        "joint_summary": joint_summary,
    }
    return metrics


def cycles_from_left_strikes(frames: List[Dict[str, object]], min_len: int = 20):
    contact_l = np.asarray([f["contact_l"] for f in frames], dtype=bool)
    hs = heel_strikes(contact_l)
    cycles = [(int(a), int(b)) for a, b in zip(hs[:-1], hs[1:]) if b - a >= min_len]
    return cycles


def resample(values: np.ndarray, n: int = 101) -> np.ndarray:
    if values.size < 2:
        return np.full(n, np.nan)
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, values)


def write_timeseries(frames: List[Dict[str, object]], path: Path):
    fields = [
        "idx",
        "time",
        "reward",
        "pelvis_x",
        "pelvis_y",
        "pelvis_z",
        "foot_l_x",
        "foot_l_y",
        "foot_l_z",
        "foot_r_x",
        "foot_r_y",
        "foot_r_z",
        "foot_l_rel_height",
        "foot_r_rel_height",
        "contact_l",
        "contact_r",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "terrain_idx",
    ] + [f"{j}_deg" for j in JOINTS] + [f"{j}_vel_deg_s" for j in JOINTS]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, frame in enumerate(frames):
            row = {
                "idx": i,
                "time": frame["time"],
                "reward": frame["reward"],
                "pelvis_x": frame["pelvis_pos"][0],
                "pelvis_y": frame["pelvis_pos"][1],
                "pelvis_z": frame["pelvis_pos"][2],
                "foot_l_x": frame["foot_l"][0],
                "foot_l_y": frame["foot_l"][1],
                "foot_l_z": frame["foot_l"][2],
                "foot_r_x": frame["foot_r"][0],
                "foot_r_y": frame["foot_r"][1],
                "foot_r_z": frame["foot_r"][2],
                "foot_l_rel_height": frame["foot_l_rel_height"],
                "foot_r_rel_height": frame["foot_r_rel_height"],
                "contact_l": int(frame["contact_l"]),
                "contact_r": int(frame["contact_r"]),
                "roll_deg": math.degrees(frame["roll"]),
                "pitch_deg": math.degrees(frame["pitch"]),
                "yaw_deg": math.degrees(frame["yaw"]),
                "terrain_idx": frame["terrain_idx"],
            }
            for j in JOINTS:
                row[f"{j}_deg"] = math.degrees(frame["joints"][j])
                row[f"{j}_vel_deg_s"] = math.degrees(frame["joint_velocities"][j])
            writer.writerow(row)


def write_npz(frames: List[Dict[str, object]], path: Path):
    if not frames:
        return
    np.savez_compressed(
        path,
        time=np.asarray([f["time"] for f in frames], dtype=float),
        reward=np.asarray([f["reward"] for f in frames], dtype=float),
        done=np.asarray([f["done"] for f in frames], dtype=bool),
        qpos=np.vstack([f["qpos"] for f in frames]),
        qvel=np.vstack([f["qvel"] for f in frames]),
        pelvis_pos=np.vstack([f["pelvis_pos"] for f in frames]),
        foot_l=np.vstack([f["foot_l"] for f in frames]),
        foot_r=np.vstack([f["foot_r"] for f in frames]),
        foot_rel_height=np.vstack([[f["foot_l_rel_height"], f["foot_r_rel_height"]] for f in frames]),
        contact=np.vstack([[f["contact_l"], f["contact_r"]] for f in frames]).astype(bool),
        joint_angles=np.vstack([[f["joints"][j] for j in JOINTS] for f in frames]),
        joint_velocities=np.vstack([[f["joint_velocities"][j] for j in JOINTS] for f in frames]),
        muscle_activation=np.vstack([f["act"] for f in frames]),
        muscle_length=np.vstack([f["length"] for f in frames]),
        muscle_force=np.vstack([f["force"] for f in frames]),
        muscle_velocity=np.vstack([f["actuator_velocity"] for f in frames]),
        muscle_scales=np.stack([f["scales"] for f in frames]),
        action=np.vstack([f["action"] for f in frames]),
        joint_names=np.asarray(JOINTS),
        actuator_names=np.asarray(frames[0]["actuator_names"]),
    )


def save_cycle_joint_plot(episodes, output_path: Path):
    x = np.linspace(0, 100, 101)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, (title, r_name, l_name) in zip(axes, SAGITTAL_JOINTS):
        all_r, all_l = [], []
        for frames, _metrics in episodes:
            for a, b in cycles_from_left_strikes(frames):
                r = np.rad2deg(np.asarray([f["joints"][r_name] for f in frames[a:b]], dtype=float))
                l = np.rad2deg(np.asarray([f["joints"][l_name] for f in frames[a:b]], dtype=float))
                all_r.append(resample(r))
                all_l.append(resample(l))
        if all_r:
            r_arr = np.vstack(all_r)
            l_arr = np.vstack(all_l)
            for arr, color, label in [(r_arr, "#b23b3b", "right"), (l_arr, "#2f6db3", "left")]:
                mean = np.nanmean(arr, axis=0)
                std = np.nanstd(arr, axis=0)
                ax.plot(x, mean, color=color, label=label)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
        else:
            for frames, _metrics in episodes:
                vals_r = np.rad2deg(np.asarray([f["joints"][r_name] for f in frames]))
                vals_l = np.rad2deg(np.asarray([f["joints"][l_name] for f in frames]))
                ax.plot(np.linspace(0, 100, len(vals_r)), vals_r, "#b23b3b", alpha=0.6, label="right")
                ax.plot(np.linspace(0, 100, len(vals_l)), vals_l, "#2f6db3", alpha=0.6, label="left")
        ax.set_ylabel("angle (deg)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", frameon=False)
    axes[-1].set_xlabel("left gait cycle (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_gait_timing_plot(episodes, output_path: Path):
    fig, axes = plt.subplots(len(episodes), 1, figsize=(9, max(2.5, 2.0 * len(episodes))), sharex=False)
    if len(episodes) == 1:
        axes = [axes]
    for ep_idx, (frames, _metrics) in enumerate(episodes):
        t = np.arange(len(frames)) * 0.01
        pelvis = np.vstack([f["pelvis_pos"] for f in frames])
        cl = np.asarray([f["contact_l"] for f in frames], dtype=float)
        cr = np.asarray([f["contact_r"] for f in frames], dtype=float)
        ax = axes[ep_idx]
        ax.plot(t, pelvis[:, 1] - pelvis[0, 1], color="black", lw=1.2, label="pelvis y displacement")
        ax.fill_between(t, -0.15, -0.05, where=cl > 0, color="#2f6db3", alpha=0.35, label="left contact")
        ax.fill_between(t, -0.30, -0.20, where=cr > 0, color="#b23b3b", alpha=0.35, label="right contact")
        ax.set_ylabel(f"ep {ep_idx + 1}")
        ax.grid(True, alpha=0.2)
    axes[0].legend(loc="best", frameon=False, ncol=3)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_muscle_plot(episodes, output_path: Path):
    x = np.linspace(0, 100, 101)
    group_names = list(MUSCLE_GROUP_PATTERNS.keys())
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
    axes = axes.ravel()
    if not episodes:
        return
    actuator_names = episodes[0][0][0]["actuator_names"]
    groups_r = group_indices(actuator_names, "r")
    groups_l = group_indices(actuator_names, "l")
    for ax, group in zip(axes, group_names):
        curves = {"right": [], "left": []}
        for frames, _metrics in episodes:
            act_mat = np.vstack([f["act"] for f in frames])
            for a, b in cycles_from_left_strikes(frames):
                for side, groups in [("right", groups_r), ("left", groups_l)]:
                    idxs = groups[group]
                    if not idxs:
                        continue
                    vals = act_mat[a:b, idxs].mean(axis=1)
                    denom = np.nanmax(np.abs(vals))
                    if denom > 1e-8:
                        vals = vals / denom
                    curves[side].append(resample(vals))
        for side, color in [("right", "#b23b3b"), ("left", "#2f6db3")]:
            if curves[side]:
                arr = np.vstack(curves[side])
                mean = np.nanmean(arr, axis=0)
                std = np.nanstd(arr, axis=0)
                ax.plot(x, mean, color=color, label=side)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16)
        ax.set_title(group)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)
    axes[0].legend(loc="best", frameon=False)
    for ax in axes[-2:]:
        ax.set_xlabel("left gait cycle (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_summary_bar(metrics_list: List[Dict[str, object]], output_path: Path):
    labels = [
        ("speed", "mean_forward_speed_mps", "m/s"),
        ("cadence", "cadence_steps_per_min", "steps/min"),
        ("step length", "step_length_m", "m"),
        ("duty L", "duty_factor_l", ""),
        ("duty R", "duty_factor_r", ""),
        ("double support", "double_support_fraction", ""),
        ("pelvis z std", "pelvis_height_std_m", "m"),
        ("pitch RMS", "pelvis_pitch_rms_deg", "deg"),
    ]
    means, stds = [], []
    for _label, key, _unit in labels:
        vals = np.asarray([m.get(key, np.nan) for m in metrics_list], dtype=float)
        means.append(np.nanmean(vals))
        stds.append(np.nanstd(vals))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color="#526d82", alpha=0.88, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n({u})" if u else a for a, _k, u in labels], rotation=0)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def terrain_color(name: str) -> str:
    palette = {
        "flat": "#2f6db3",
        "rough": "#b23b3b",
        "hilly": "#4f8a3b",
        "stairs": "#9b6bb3",
        "mixed": "#d08a2d",
        "random": "#526d82",
    }
    return palette.get(name, "#526d82")


def best_by_terrain(episodes, key: str = "distance_m"):
    best = {}
    for frames, metrics in episodes:
        terrain = str(metrics.get("terrain_requested", "random"))
        value = float(metrics.get(key, float("nan")))
        if math.isnan(value):
            continue
        if terrain not in best or value > float(best[terrain][1].get(key, -np.inf)):
            best[terrain] = (frames, metrics)
    return best


def write_best_by_terrain_csv(best, path: Path):
    keys = [
        "seed",
        "duration_s",
        "return",
        "distance_m",
        "mean_forward_speed_mps",
        "cadence_steps_per_min",
        "step_count",
        "step_length_m",
        "duty_factor_l",
        "duty_factor_r",
        "double_support_fraction",
        "no_contact_fraction",
        "pelvis_height_std_m",
        "pelvis_roll_rms_deg",
        "pelvis_pitch_rms_deg",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["terrain"] + keys)
        writer.writeheader()
        for terrain, (_frames, metrics) in best.items():
            row = {"terrain": terrain}
            row.update({k: metrics.get(k, np.nan) for k in keys})
            writer.writerow(row)


def representative_curve(frames, key: str) -> np.ndarray:
    curves = []
    for a, b in cycles_from_left_strikes(frames):
        vals = np.rad2deg(np.asarray([f["joints"][key] for f in frames[a:b]], dtype=float))
        curves.append(resample(vals))
    if curves:
        return np.nanmean(np.vstack(curves), axis=0)
    vals = np.rad2deg(np.asarray([f["joints"][key] for f in frames], dtype=float))
    return resample(vals)


def representative_muscle_curve(frames, group: str, side: str, actuator_names: List[str]) -> np.ndarray | None:
    groups = group_indices(actuator_names, side)
    idxs = groups.get(group, [])
    if not idxs:
        return None
    act_mat = np.vstack([f["act"] for f in frames])
    curves = []
    for a, b in cycles_from_left_strikes(frames):
        vals = act_mat[a:b, idxs].mean(axis=1)
        curves.append(resample(vals))
    if curves:
        curve = np.nanmean(np.vstack(curves), axis=0)
    else:
        curve = resample(act_mat[:, idxs].mean(axis=1))
    denom = np.nanmax(np.abs(curve))
    if denom > 1e-8:
        curve = curve / denom
    return curve


def save_best_spatiotemporal_plot(best, output_path: Path):
    metrics = [
        ("distance_m", "distance (m)"),
        ("mean_forward_speed_mps", "speed (m/s)"),
        ("cadence_steps_per_min", "cadence (steps/min)"),
        ("step_length_m", "step length (m)"),
        ("duty_factor_l", "duty L"),
        ("duty_factor_r", "duty R"),
        ("double_support_fraction", "double support"),
        ("pelvis_height_std_m", "pelvis z std (m)"),
    ]
    terrains = list(best.keys())
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.ravel()
    for ax, (key, label) in zip(axes, metrics):
        vals = [best[t][1].get(key, np.nan) for t in terrains]
        ax.bar(terrains, vals, color=[terrain_color(t) for t in terrains], alpha=0.88)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_best_joint_terrain_plot(best, output_path: Path):
    x = np.linspace(0, 100, 101)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for row, (title, r_name, l_name) in enumerate(SAGITTAL_JOINTS):
        for col, (side_label, key) in enumerate([("right", r_name), ("left", l_name)]):
            ax = axes[row, col]
            for terrain, (frames, metrics) in best.items():
                curve = representative_curve(frames, key)
                ax.plot(
                    x,
                    curve,
                    lw=1.8,
                    color=terrain_color(terrain),
                    label=f"{terrain} seed={metrics.get('seed', '')}",
                )
            ax.set_title(f"{title} - {side_label}")
            ax.set_ylabel("angle (deg)")
            ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("left gait cycle (%)")
    axes[0, 0].legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_best_contact_terrain_plot(best, output_path: Path):
    terrains = list(best.keys())
    fig, ax = plt.subplots(figsize=(11, max(4, 0.8 * len(terrains) + 1.5)))
    for row, terrain in enumerate(terrains):
        frames, _metrics = best[terrain]
        x = np.linspace(0, 100, len(frames))
        cl = np.asarray([f["contact_l"] for f in frames], dtype=bool)
        cr = np.asarray([f["contact_r"] for f in frames], dtype=bool)
        base = row * 1.0
        ax.fill_between(x, base + 0.05, base + 0.35, where=cl, color="#2f6db3", alpha=0.5)
        ax.fill_between(x, base + 0.45, base + 0.75, where=cr, color="#b23b3b", alpha=0.5)
        ax.text(-2.5, base + 0.4, terrain, ha="right", va="center", color=terrain_color(terrain))
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("episode progress (%)")
    ax.set_title("Best rollout contact timing by terrain (blue=L, red=R)")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_best_muscle_terrain_plot(best, output_path: Path):
    groups = ["HFL", "GLU", "VAS", "GAS", "SOL", "TA"]
    x = np.linspace(0, 100, 101)
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    axes = axes.ravel()
    for ax, group in zip(axes, groups):
        for terrain, (frames, _metrics) in best.items():
            actuator_names = frames[0]["actuator_names"]
            curves = []
            for side in ["l", "r"]:
                curve = representative_muscle_curve(frames, group, side, actuator_names)
                if curve is not None:
                    curves.append(curve)
            if curves:
                ax.plot(x, np.nanmean(np.vstack(curves), axis=0), color=terrain_color(terrain), lw=1.8, label=terrain)
        ax.set_title(group)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)
    axes[0].legend(loc="best", frameon=False, fontsize=8)
    for ax in axes[-2:]:
        ax.set_xlabel("left gait cycle (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def aggregate_metrics(metrics_list: List[Dict[str, object]]) -> Dict[str, object]:
    scalar_keys = [
        "duration_s",
        "return",
        "distance_m",
        "mean_forward_speed_mps",
        "cadence_steps_per_min",
        "step_count",
        "step_length_m",
        "duty_factor_l",
        "duty_factor_r",
        "duty_factor_asymmetry",
        "double_support_fraction",
        "no_contact_fraction",
        "leg_crossing_rate",
        "close_lateral_feet_rate",
        "foot_clearance_std_l_m",
        "foot_clearance_std_r_m",
        "pelvis_lateral_sway_std_m",
        "pelvis_height_mean_m",
        "pelvis_height_std_m",
        "pelvis_height_min_m",
        "pelvis_roll_rms_deg",
        "pelvis_pitch_rms_deg",
        "survived_horizon",
        "fell_or_failed",
        "sigma_mean",
        "sigma_std",
        "nu_mean",
        "nu_std",
        "kappa_mean",
        "kappa_std",
    ]
    summary = {}
    for key in scalar_keys:
        vals = np.asarray([m.get(key, np.nan) for m in metrics_list], dtype=float)
        summary[key] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    terrain_counts: Dict[str, int] = {}
    for m in metrics_list:
        for terrain, count in m.get("terrain_counts", {}).items():
            terrain_counts[terrain] = terrain_counts.get(terrain, 0) + int(count)
    summary["terrain_counts"] = terrain_counts

    joint_summary = {}
    for j in JOINTS:
        joint_summary[j] = {}
        for subkey in ["mean_deg", "min_deg", "max_deg", "rom_deg", "vel_rms_deg_s", "vel_peak_abs_deg_s"]:
            vals = np.asarray([m["joint_summary"][j][subkey] for m in metrics_list], dtype=float)
            joint_summary[j][subkey] = {
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
            }
    summary["joint_summary"] = joint_summary
    return summary


def write_metrics_csv(metrics_list: List[Dict[str, object]], path: Path):
    keys = [
        "condition",
        "env_id",
        "seed",
        "run_idx",
        "duration_s",
        "return",
        "distance_m",
        "mean_forward_speed_mps",
        "cadence_steps_per_min",
        "step_count",
        "step_length_m",
        "duty_factor_l",
        "duty_factor_r",
        "duty_factor_asymmetry",
        "double_support_fraction",
        "no_contact_fraction",
        "leg_crossing_rate",
        "close_lateral_feet_rate",
        "foot_clearance_std_l_m",
        "foot_clearance_std_r_m",
        "pelvis_lateral_sway_std_m",
        "pelvis_height_mean_m",
        "pelvis_height_std_m",
        "pelvis_height_min_m",
        "pelvis_roll_rms_deg",
        "pelvis_pitch_rms_deg",
        "survived_horizon",
        "fell_or_failed",
        "sigma_mean",
        "sigma_std",
        "nu_mean",
        "nu_std",
        "kappa_mean",
        "kappa_std",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "terrain_requested"] + keys)
        writer.writeheader()
        for i, m in enumerate(metrics_list):
            row = {"episode": i + 1, "terrain_requested": m.get("terrain_requested", "random")}
            row.update({k: m.get(k, np.nan) for k in keys})
            writer.writerow(row)


def finite_metric(metrics: Dict[str, object], key: str, default: float = float("nan")) -> float:
    try:
        value = float(metrics.get(key, default))
    except (TypeError, ValueError):
        return default
    return value


def filter_episode_quality(
    episodes,
    min_frames: int,
    min_distance: float,
    min_steps: int,
    min_speed: float,
    return_mad_threshold: float,
):
    rows = []
    accepted = []
    for frames, metrics in episodes:
        reasons = []
        if len(frames) < min_frames:
            reasons.append(f"frames<{min_frames}")
        if finite_metric(metrics, "distance_m", 0.0) < min_distance:
            reasons.append(f"distance<{min_distance}")
        if finite_metric(metrics, "step_count", 0.0) < min_steps:
            reasons.append(f"steps<{min_steps}")
        if finite_metric(metrics, "mean_forward_speed_mps", 0.0) < min_speed:
            reasons.append(f"speed<{min_speed}")
        rows.append({"frames": frames, "metrics": metrics, "reasons": reasons})

    provisional_returns = np.asarray(
        [finite_metric(row["metrics"], "return") for row in rows if not row["reasons"]],
        dtype=float,
    )
    provisional_returns = provisional_returns[np.isfinite(provisional_returns)]
    median_return = float(np.nanmedian(provisional_returns)) if provisional_returns.size else float("nan")
    mad_return = (
        float(np.nanmedian(np.abs(provisional_returns - median_return)))
        if provisional_returns.size
        else float("nan")
    )

    if return_mad_threshold > 0 and provisional_returns.size >= 4 and np.isfinite(mad_return):
        if mad_return <= 1e-9:
            lower_return = median_return
        else:
            lower_return = median_return - return_mad_threshold * 1.4826 * mad_return
        for row in rows:
            if row["reasons"]:
                continue
            ret = finite_metric(row["metrics"], "return")
            if np.isfinite(ret) and ret < lower_return:
                row["reasons"].append(f"return_mad_low<{lower_return:.3f}")

    for row in rows:
        metrics = row["metrics"]
        metrics["quality_status"] = "accepted" if not row["reasons"] else "rejected"
        metrics["quality_reasons"] = ";".join(row["reasons"])
        if not row["reasons"]:
            accepted.append((row["frames"], metrics))

    return accepted, rows, {"median_return": median_return, "mad_return": mad_return}


def write_quality_filter_csv(rows, path: Path):
    fields = [
        "quality_status",
        "quality_reasons",
        "condition",
        "env_id",
        "terrain_requested",
        "seed",
        "run_idx",
        "n_frames",
        "return",
        "distance_m",
        "mean_forward_speed_mps",
        "step_count",
        "leg_crossing_rate",
        "duty_factor_asymmetry",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            out = {
                "quality_status": metrics.get("quality_status", ""),
                "quality_reasons": metrics.get("quality_reasons", ""),
                "n_frames": len(row["frames"]),
            }
            out.update({k: metrics.get(k, np.nan) for k in fields if k not in out})
            writer.writerow(out)


def write_report(summary: Dict[str, object], output_path: Path, model_path: Path, vec_path: Path):
    s = summary
    lines = []
    lines.append("# Biomechanical validation summary")
    lines.append("")
    lines.append(f"- Model: `{model_path}`")
    lines.append(f"- VecNormalize: `{vec_path}`")
    if s.get("eval_mode"):
        lines.append(f"- Evaluation mode: `{s['eval_mode']}`")
    if s.get("quality_filter"):
        q = s["quality_filter"]
        lines.append(
            f"- Quality filter: accepted {q.get('accepted_episodes', 0)}/"
            f"{q.get('raw_episodes', 0)} episodes"
        )
    lines.append(f"- Terrain samples: {s.get('terrain_counts', {})}")
    lines.append("")
    lines.append("## Spatiotemporal gait")
    lines.append("")
    for key, label in [
        ("mean_forward_speed_mps", "Mean forward speed (m/s)"),
        ("cadence_steps_per_min", "Cadence (steps/min)"),
        ("step_length_m", "Step length (m)"),
        ("duty_factor_l", "Duty factor L"),
        ("duty_factor_r", "Duty factor R"),
        ("duty_factor_asymmetry", "Duty-factor asymmetry"),
        ("double_support_fraction", "Double-support fraction"),
        ("no_contact_fraction", "No-contact fraction"),
        ("leg_crossing_rate", "Leg-crossing rate"),
        ("close_lateral_feet_rate", "Close lateral feet rate"),
        ("pelvis_lateral_sway_std_m", "Pelvis lateral sway std (m)"),
        ("pelvis_height_std_m", "Pelvis height std (m)"),
        ("pelvis_pitch_rms_deg", "Pelvis pitch RMS (deg)"),
        ("pelvis_roll_rms_deg", "Pelvis roll RMS (deg)"),
    ]:
        v = s[key]
        lines.append(f"- {label}: {v['mean']:.3f} ± {v['std']:.3f}")
    lines.append("")
    lines.append("## Sagittal joint ROM")
    lines.append("")
    for label, r_name, l_name in SAGITTAL_JOINTS:
        r_rom = s["joint_summary"][r_name]["rom_deg"]
        l_rom = s["joint_summary"][l_name]["rom_deg"]
        lines.append(
            f"- {label}: right {r_rom['mean']:.1f} ± {r_rom['std']:.1f} deg, "
            f"left {l_rom['mean']:.1f} ± {l_rom['std']:.1f} deg"
        )
    if s.get("by_requested_terrain"):
        lines.append("")
        lines.append("## Terrain-wise summary")
        lines.append("")
        lines.append("| Terrain | Episodes | Duration (s) | Speed (m/s) | Distance (m) | Cadence (steps/min) | Step length (m) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for terrain, terrain_summary in s["by_requested_terrain"].items():
            lines.append(
                f"| {terrain} | {terrain_summary['n_episodes']} | "
                f"{terrain_summary['duration_s']['mean']:.2f} ± {terrain_summary['duration_s']['std']:.2f} | "
                f"{terrain_summary['mean_forward_speed_mps']['mean']:.2f} ± {terrain_summary['mean_forward_speed_mps']['std']:.2f} | "
                f"{terrain_summary['distance_m']['mean']:.2f} ± {terrain_summary['distance_m']['std']:.2f} | "
                f"{terrain_summary['cadence_steps_per_min']['mean']:.1f} ± {terrain_summary['cadence_steps_per_min']['std']:.1f} | "
                f"{terrain_summary['step_length_m']['mean']:.2f} ± {terrain_summary['step_length_m']['std']:.2f} |"
            )
    if s.get("best_by_terrain"):
        lines.append("")
        lines.append("## Best rollout by terrain")
        lines.append("")
        lines.append("| Terrain | Seed | Distance (m) | Duration (s) | Speed (m/s) | Cadence (steps/min) | Step length (m) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for terrain, metrics in s["best_by_terrain"].items():
            lines.append(
                f"| {terrain} | {metrics.get('seed', '')} | "
                f"{metrics.get('distance_m', float('nan')):.2f} | "
                f"{metrics.get('duration_s', float('nan')):.2f} | "
                f"{metrics.get('mean_forward_speed_mps', float('nan')):.2f} | "
                f"{metrics.get('cadence_steps_per_min', float('nan')):.1f} | "
                f"{metrics.get('step_length_m', float('nan')):.2f} |"
            )
    lines.append("")
    lines.append("## Generated files")
    lines.append("")
    lines.append("- `metrics_summary.json`: all scalar and joint ROM summaries")
    lines.append("- `metrics_by_episode.csv`: per-episode gait metrics")
    lines.append("- `episode_*_timeseries.csv`: joint/foot/pelvis time series")
    lines.append("- `joint_trajectories_gait_cycle.png`: hip/knee/ankle trajectories")
    lines.append("- `muscle_activation_envelopes.png`: major muscle-group activations")
    lines.append("- `gait_timing_contacts.png`: foot contacts and pelvis displacement")
    if s.get("best_by_terrain"):
        lines.append("- `best_by_terrain.csv`: farthest rollout selected for each terrain")
        lines.append("- `terrain_best_spatiotemporal.png`: best-rollout scalar comparison by terrain")
        lines.append("- `terrain_best_joint_trajectories.png`: terrain-conditioned joint trajectory comparison")
        lines.append("- `terrain_best_contact_timing.png`: terrain-conditioned foot-contact timing comparison")
        lines.append("- `terrain_best_muscle_activation.png`: terrain-conditioned muscle activation comparison")
    output_path.write_text("\n".join(lines))


COMPARISON_KEYS = [
    ("return", "return"),
    ("distance_m", "distance (m)"),
    ("mean_forward_speed_mps", "speed (m/s)"),
    ("cadence_steps_per_min", "cadence"),
    ("step_length_m", "step length (m)"),
    ("duty_factor_asymmetry", "duty asym"),
    ("double_support_fraction", "double support"),
    ("leg_crossing_rate", "leg crossing"),
    ("pelvis_lateral_sway_std_m", "pelvis sway (m)"),
    ("pelvis_pitch_rms_deg", "pitch RMS"),
]


def summarize_by_condition(metrics_list: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out = {}
    labels = sorted({str(m.get("condition", "condition")) for m in metrics_list})
    for label in labels:
        items = [m for m in metrics_list if str(m.get("condition", "condition")) == label]
        summary = aggregate_metrics(items)
        summary["n_episodes"] = len(items)
        out[label] = summary
    return out


def write_condition_comparison_csv(by_condition: Dict[str, Dict[str, object]], path: Path):
    fields = ["condition", "n_episodes"]
    for key, _label in COMPARISON_KEYS:
        fields.extend([f"{key}_mean", f"{key}_std"])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for condition, summary in by_condition.items():
            row = {"condition": condition, "n_episodes": summary.get("n_episodes", 0)}
            for key, _label in COMPARISON_KEYS:
                val = summary.get(key, {})
                row[f"{key}_mean"] = val.get("mean", np.nan)
                row[f"{key}_std"] = val.get("std", np.nan)
            writer.writerow(row)


def save_condition_comparison_plot(by_condition: Dict[str, Dict[str, object]], output_path: Path):
    conditions = list(by_condition.keys())
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
    axes = axes.ravel()
    x = np.arange(len(conditions))
    for ax, (key, label) in zip(axes, COMPARISON_KEYS):
        means = [by_condition[c].get(key, {}).get("mean", np.nan) for c in conditions]
        stds = [by_condition[c].get(key, {}).get("std", np.nan) for c in conditions]
        ax.bar(x, means, yerr=stds, color="#526d82", alpha=0.88, capsize=3)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def condition_joint_curves(episodes, condition: str, joint_name: str) -> List[np.ndarray]:
    curves = []
    for frames, metrics in episodes:
        if str(metrics.get("condition", "")) != condition:
            continue
        for a, b in cycles_from_left_strikes(frames):
            vals = np.rad2deg(np.asarray([f["joints"][joint_name] for f in frames[a:b]], dtype=float))
            curves.append(resample(vals))
    return curves


def save_condition_joint_comparison_plot(episodes, output_path: Path):
    conditions = sorted({str(metrics.get("condition", "condition")) for _frames, metrics in episodes})
    x = np.linspace(0, 100, 101)
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    cmap = plt.get_cmap("tab10")
    for row, (title, r_name, l_name) in enumerate(SAGITTAL_JOINTS):
        for col, (side, joint_name) in enumerate([("right", r_name), ("left", l_name)]):
            ax = axes[row, col]
            for idx, condition in enumerate(conditions):
                curves = condition_joint_curves(episodes, condition, joint_name)
                if not curves:
                    continue
                arr = np.vstack(curves)
                mean = np.nanmean(arr, axis=0)
                std = np.nanstd(arr, axis=0)
                color = cmap(idx % 10)
                ax.plot(x, mean, color=color, lw=1.9, label=condition)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.13)
            ax.set_title(f"{title} - {side}")
            ax.set_ylabel("angle (deg)")
            ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("left gait cycle (%)")
    axes[0, 0].legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-dir", type=Path, default=None)
    parser.add_argument("--env-id", type=str, default=None)
    parser.add_argument(
        "--condition",
        action="append",
        default=None,
        help=(
            "Paper comparison condition. Format: LABEL,ENV_ID,MODEL_DIR or "
            "LABEL,ENV_ID,MODEL_ZIP,VECNORMALIZE_FILE. Repeat for FullSDE, StrengthOnly, Fixed, T2A, etc."
        ),
    )
    parser.add_argument("--algo", type=str, default="PPO", help="Algorithm name used when inferring files from MODEL_DIR.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--obs-wrapper", choices=["auto", "none", "fat"], default="auto")
    parser.add_argument(
        "--reset-type",
        choices=["env-default", "init", "random", "none"],
        default="env-default",
        help="Override env.reset_type before each rollout when the environment supports it.",
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-seeds", type=int, default=1, help="Number of sequential seeds to test per terrain.")
    parser.add_argument("--seed-base", type=int, default=0, help="First seed when --seeds is not provided.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit seed list to test per terrain.")
    parser.add_argument(
        "--select-best-per-terrain",
        action="store_true",
        help="Select the farthest rollout for each terrain and create terrain-comparison plots.",
    )
    parser.add_argument("--best-key", type=str, default="distance_m", help="Metric used for best rollout selection.")
    parser.add_argument(
        "--filter-quality",
        action="store_true",
        help="Use only quality-filtered episodes for summaries and plots; raw rollouts are still saved.",
    )
    parser.add_argument("--min-frames", type=int, default=80, help="Reject episodes shorter than this many frames.")
    parser.add_argument("--min-distance", type=float, default=0.25, help="Reject episodes with less forward distance.")
    parser.add_argument("--min-steps", type=int, default=2, help="Reject episodes with fewer detected steps.")
    parser.add_argument("--min-speed", type=float, default=0.05, help="Reject episodes below this mean speed.")
    parser.add_argument(
        "--return-mad-threshold",
        type=float,
        default=3.0,
        help="Reject low-return outliers below median - threshold*1.4826*MAD within each condition. Use 0 to disable.",
    )
    parser.add_argument(
        "--terrain",
        nargs="+",
        choices=["random", "flat", "rough", "hilly", "stairs", "mixed"],
        default=None,
        help="Optional fixed terrain(s). With multiple values, --episodes is run per terrain.",
    )
    args = parser.parse_args()

    if args.condition:
        conditions = [parse_condition(text, algo=args.algo) for text in args.condition]
        output_dir = args.output_dir or (ROOT / "biomech_validation_outputs")
        success_dir = None
    else:
        success_dir = (args.success_dir or latest_success_dir()).resolve()
        conditions = [default_success_condition(success_dir, env_id=args.env_id)]
        output_dir = args.output_dir or (success_dir / "biomech_validation")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if success_dir is not None:
        print(f"success_dir={success_dir}")
    print(f"output_dir={output_dir}")
    print("conditions:")
    for condition in conditions:
        print(f"  {condition.label}: env={condition.env_id}")
        print(f"    model={condition.model_path}")
        print(f"    vecnormalize={condition.vecnormalize_path}")

    requested_terrains = args.terrain or [None]
    if args.seeds is not None:
        seed_values = args.seeds
    elif args.num_seeds > 0:
        seed_values = list(range(args.seed_base, args.seed_base + args.num_seeds))
    else:
        seed_values = [None]
    all_episodes = []
    all_metrics = []
    all_raw_metrics = []
    ep_global = 0

    for condition in conditions:
        condition_dir = output_dir / condition.label.replace(" ", "_")
        condition_dir.mkdir(parents=True, exist_ok=True)
        from stable_baselines3 import PPO

        model = PPO.load(str(condition.model_path), device=args.device)
        condition_episodes = []
        condition_metrics = []

        for terrain in requested_terrains:
            terrain_label = terrain or "random"
            for seed in seed_values:
                for ep in range(args.episodes):
                    ep_global += 1
                    run_seed = None if seed is None else int(seed + ep * 100000)
                    vec_env, holder = make_eval_env(
                        condition.env_id,
                        condition.vecnormalize_path,
                        obs_wrapper=args.obs_wrapper,
                    )
                    frames, total_reward = rollout(
                        model,
                        vec_env,
                        holder,
                        max_steps=args.max_steps,
                        deterministic=not args.stochastic,
                        terrain=terrain,
                        seed=run_seed,
                        reset_type=args.reset_type,
                    )
                    metrics = compute_metrics(frames, max_steps=args.max_steps)
                    metrics["return"] = total_reward
                    metrics["terrain_requested"] = terrain_label
                    metrics["condition"] = condition.label
                    metrics["env_id"] = condition.env_id
                    metrics["seed"] = run_seed
                    metrics["run_idx"] = ep_global
                    condition_metrics.append(metrics)
                    condition_episodes.append((frames, metrics))

                    seed_label = "none" if run_seed is None else str(run_seed)
                    ts_stem = f"{terrain_label}_seed_{seed_label}_episode_{ep + 1:02d}"
                    write_timeseries(frames, condition_dir / f"{ts_stem}_timeseries.csv")
                    write_npz(frames, condition_dir / f"{ts_stem}_rollout.npz")
                    vec_env.close()
                    print(
                        f"{condition.label} episode {ep_global} ({terrain_label}, seed={seed_label}, "
                        f"repeat={ep + 1}/{args.episodes}): "
                        f"frames={len(frames)} return={total_reward:.1f} "
                        f"distance={metrics.get('distance_m', float('nan')):.2f}m "
                        f"speed={metrics.get('mean_forward_speed_mps', float('nan')):.2f}m/s "
                        f"steps={metrics.get('step_count', 0)} terrains={metrics.get('terrain_counts', {})}"
                    )

        write_metrics_csv(condition_metrics, condition_dir / "metrics_by_episode_raw.csv")
        if args.filter_quality:
            accepted_condition_episodes, quality_rows, quality_stats = filter_episode_quality(
                condition_episodes,
                min_frames=args.min_frames,
                min_distance=args.min_distance,
                min_steps=args.min_steps,
                min_speed=args.min_speed,
                return_mad_threshold=args.return_mad_threshold,
            )
            write_quality_filter_csv(quality_rows, condition_dir / "episode_quality_filter.csv")
            condition_episodes_for_stats = accepted_condition_episodes
            condition_metrics_for_stats = [metrics for _frames, metrics in accepted_condition_episodes]
            if not condition_metrics_for_stats:
                print(f"WARNING: {condition.label} quality filter rejected all episodes; using raw episodes for stats.")
                condition_episodes_for_stats = condition_episodes
                condition_metrics_for_stats = condition_metrics
        else:
            quality_rows = []
            quality_stats = {}
            condition_episodes_for_stats = condition_episodes
            condition_metrics_for_stats = condition_metrics

        all_raw_metrics.extend(condition_metrics)
        all_metrics.extend(condition_metrics_for_stats)
        all_episodes.extend(condition_episodes_for_stats)

        condition_summary = aggregate_metrics(condition_metrics_for_stats)
        condition_summary["condition"] = condition.label
        condition_summary["env_id"] = condition.env_id
        condition_summary["eval_mode"] = "stochastic" if args.stochastic else "deterministic"
        condition_summary["quality_filter"] = {
            "enabled": bool(args.filter_quality),
            "raw_episodes": len(condition_metrics),
            "accepted_episodes": len(condition_metrics_for_stats),
            "rejected_episodes": max(0, len(condition_metrics) - len(condition_metrics_for_stats)),
            "min_frames": args.min_frames,
            "min_distance": args.min_distance,
            "min_steps": args.min_steps,
            "min_speed": args.min_speed,
            "return_mad_threshold": args.return_mad_threshold,
            **quality_stats,
        }
        if args.terrain:
            by_terrain = {}
            for terrain in requested_terrains:
                terrain_label = terrain or "random"
                terrain_metrics = [
                    m for m in condition_metrics_for_stats if m.get("terrain_requested") == terrain_label
                ]
                if not terrain_metrics:
                    continue
                terrain_summary = aggregate_metrics(terrain_metrics)
                terrain_summary["n_episodes"] = len(terrain_metrics)
                by_terrain[terrain_label] = terrain_summary
            condition_summary["by_requested_terrain"] = by_terrain

        if args.select_best_per_terrain:
            best = best_by_terrain(condition_episodes_for_stats, key=args.best_key)
            condition_summary["best_by_terrain"] = {terrain: metrics for terrain, (_frames, metrics) in best.items()}
            write_best_by_terrain_csv(best, condition_dir / "best_by_terrain.csv")
            (condition_dir / "best_by_terrain_summary.json").write_text(
                json.dumps(condition_summary["best_by_terrain"], indent=2)
            )
            save_best_spatiotemporal_plot(best, condition_dir / "terrain_best_spatiotemporal.png")
            save_best_joint_terrain_plot(best, condition_dir / "terrain_best_joint_trajectories.png")
            save_best_contact_terrain_plot(best, condition_dir / "terrain_best_contact_timing.png")
            save_best_muscle_terrain_plot(best, condition_dir / "terrain_best_muscle_activation.png")

        (condition_dir / "metrics_summary.json").write_text(json.dumps(condition_summary, indent=2))
        write_metrics_csv(condition_metrics_for_stats, condition_dir / "metrics_by_episode.csv")
        save_cycle_joint_plot(condition_episodes_for_stats, condition_dir / "joint_trajectories_gait_cycle.png")
        save_muscle_plot(condition_episodes_for_stats, condition_dir / "muscle_activation_envelopes.png")
        save_gait_timing_plot(condition_episodes_for_stats, condition_dir / "gait_timing_contacts.png")
        save_summary_bar(condition_metrics_for_stats, condition_dir / "spatiotemporal_summary.png")
        write_report(
            condition_summary,
            condition_dir / "REPORT.md",
            condition.model_path,
            condition.vecnormalize_path,
        )

    summary = aggregate_metrics(all_metrics)
    summary["eval_mode"] = "stochastic" if args.stochastic else "deterministic"
    summary["conditions"] = [condition.label for condition in conditions]
    summary["quality_filter"] = {
        "enabled": bool(args.filter_quality),
        "raw_episodes": len(all_raw_metrics),
        "accepted_episodes": len(all_metrics),
        "rejected_episodes": max(0, len(all_raw_metrics) - len(all_metrics)),
    }
    summary["by_condition"] = summarize_by_condition(all_metrics)
    if args.terrain:
        by_terrain = {}
        for terrain in requested_terrains:
            terrain_label = terrain or "random"
            terrain_metrics = [m for m in all_metrics if m.get("terrain_requested") == terrain_label]
            if not terrain_metrics:
                continue
            terrain_summary = aggregate_metrics(terrain_metrics)
            terrain_summary["n_episodes"] = len(terrain_metrics)
            by_terrain[terrain_label] = terrain_summary
        summary["by_requested_terrain"] = by_terrain

    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    write_metrics_csv(all_metrics, output_dir / "metrics_by_episode.csv")
    write_metrics_csv(all_raw_metrics, output_dir / "metrics_by_episode_raw.csv")
    write_condition_comparison_csv(summary["by_condition"], output_dir / "condition_comparison.csv")
    save_condition_comparison_plot(summary["by_condition"], output_dir / "condition_spatiotemporal_comparison.png")
    save_condition_joint_comparison_plot(all_episodes, output_dir / "condition_joint_trajectories.png")
    save_summary_bar(all_metrics, output_dir / "spatiotemporal_summary_all_conditions.png")

    print("\nSummary:")
    for key in [
        "mean_forward_speed_mps",
        "cadence_steps_per_min",
        "step_length_m",
        "duty_factor_l",
        "duty_factor_r",
        "double_support_fraction",
        "pelvis_height_std_m",
        "pelvis_pitch_rms_deg",
    ]:
        v = summary[key]
        print(f"  {key}: {v['mean']:.3f} ± {v['std']:.3f}")
    print("\nCondition comparison:")
    for condition, condition_summary in summary["by_condition"].items():
        speed = condition_summary["mean_forward_speed_mps"]["mean"]
        crossing = condition_summary["leg_crossing_rate"]["mean"]
        duty_asym = condition_summary["duty_factor_asymmetry"]["mean"]
        print(f"  {condition}: speed={speed:.3f} m/s, leg_crossing={crossing:.3f}, duty_asym={duty_asym:.3f}")
    print(f"\nSaved validation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
