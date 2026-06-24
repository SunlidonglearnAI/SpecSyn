#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
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


STATUS_ORDER = ["completed", "running", "failed", "pending"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize CBS RL training progress for the 63-task experiment grid."
    )
    parser.add_argument("run_id", help="Run ID under myosuite/agents/outputs_cbs and paper/rl_training_logs.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[101, 202, 303],
        help="Seed list used for the run.",
    )
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Repository root.",
    )
    return parser.parse_args()


def tmux_session_state(session_name):
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return "tmux-unavailable"
    return "active" if result.returncode == 0 else "inactive"


def master_process_state(pid_path):
    if not pid_path.exists():
        return "missing", None
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return "invalid", None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "exited", pid
    except PermissionError:
        return "alive-no-access", pid
    return "alive", pid


def read_log_status(log_path):
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if re.search(r"END .* status=0\b", text):
        return "ended_ok"
    if re.search(r"END .* status=(?!0)\d+\b", text):
        return "ended_error"
    if "Traceback" in text or "failed." in text.lower():
        return "error"
    return "started"


def progress_snapshot(progress_path):
    if not progress_path.exists() or progress_path.stat().st_size == 0:
        return None
    try:
        with progress_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return None
    if not rows:
        return None
    row = rows[-1]
    snapshot = {}
    for key in ("time/total_timesteps", "time/time_elapsed", "time/fps", "eval/mean_reward"):
        if key in row and row[key] != "":
            snapshot[key] = row[key]
    return snapshot


def latest_mtime(paths):
    mtimes = []
    for path in paths:
        if path.exists():
            mtimes.append(path.stat().st_mtime)
    return max(mtimes) if mtimes else None


def infer_status(job):
    if job["model_path"].exists():
        return "completed"
    log_state = read_log_status(job["log_path"])
    if log_state in {"ended_error", "error"}:
        return "failed"
    if job["output_dir"].exists() or log_state == "started":
        return "running"
    return "pending"


def format_age(timestamp):
    if timestamp is None:
        return "-"
    seconds = max(0, int(time.time() - timestamp))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def build_jobs(root, run_id, seeds):
    log_root = root / "paper" / "rl_training_logs" / run_id
    output_root = root / "myosuite" / "agents" / "outputs_cbs" / run_id
    jobs = []
    for category, env_ids in EXPERIMENT_GROUPS.items():
        for env_id in env_ids:
            for seed in seeds:
                output_dir = output_root / category / env_id / f"seed_{seed}"
                progress_dir = output_dir / f"results_{env_id}"
                jobs.append(
                    {
                        "category": category,
                        "env_id": env_id,
                        "seed": seed,
                        "output_dir": output_dir,
                        "progress_path": progress_dir / "progress.csv",
                        "model_path": output_dir / f"{env_id}_PPO_model.zip",
                        "log_path": log_root / f"{category}__{env_id}__seed_{seed}.log",
                    }
                )
    return jobs, log_root, output_root


def submitted_count(master_log_path):
    if not master_log_path.exists():
        return 0
    try:
        text = master_log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return len(re.findall(r"^submitted\s+", text, flags=re.MULTILINE))


def main():
    args = parse_args()
    jobs, log_root, output_root = build_jobs(args.root, args.run_id, args.seeds)

    session_name = args.run_id.replace("-", "_")
    session_state = tmux_session_state(session_name)
    master_log_path = log_root / "master.log"
    master_pid_state, master_pid = master_process_state(log_root / "master.pid")

    overall = Counter()
    per_category = defaultdict(Counter)
    active_rows = []
    failed_rows = []

    for job in jobs:
        status = infer_status(job)
        overall[status] += 1
        per_category[job["category"]][status] += 1

        snapshot = progress_snapshot(job["progress_path"])
        updated = latest_mtime([job["progress_path"], job["log_path"], job["model_path"]])
        job["snapshot"] = snapshot
        job["updated"] = updated
        job["status"] = status

        if status == "running":
            active_rows.append(job)
        elif status == "failed":
            failed_rows.append(job)

    total_jobs = len(jobs)
    completed = overall["completed"]
    print(f"Run ID: {args.run_id}")
    print(f"tmux session: {session_state}")
    print(f"master pid: {master_pid if master_pid is not None else '-'} ({master_pid_state})")
    print(f"Output root: {output_root}")
    print(f"Log root: {log_root}")
    print(f"Total jobs: {total_jobs}")
    print(f"Submitted by launcher: {submitted_count(master_log_path)}/{total_jobs}")
    print(
        "Overall: "
        + ", ".join(f"{status}={overall[status]}" for status in STATUS_ORDER)
        + f", done={completed}/{total_jobs} ({completed / total_jobs:.1%})"
    )

    print("\nBy category:")
    for category in EXPERIMENT_GROUPS:
        counts = per_category[category]
        summary = ", ".join(f"{status}={counts[status]}" for status in STATUS_ORDER)
        print(f"  {category:16s} {summary}")

    if active_rows:
        print("\nRunning jobs:")
        active_rows.sort(key=lambda item: (item["category"], item["env_id"], item["seed"]))
        for job in active_rows[:12]:
            snap = job["snapshot"] or {}
            extras = []
            if "time/total_timesteps" in snap:
                extras.append(f"steps={snap['time/total_timesteps']}")
            if "time/fps" in snap:
                extras.append(f"fps={snap['time/fps']}")
            if "eval/mean_reward" in snap:
                extras.append(f"eval={snap['eval/mean_reward']}")
            extras.append(f"updated={format_age(job['updated'])}")
            print(
                f"  {job['category']}/{job['env_id']}/seed_{job['seed']}: "
                + ", ".join(extras)
            )
        if len(active_rows) > 12:
            print(f"  ... {len(active_rows) - 12} more running jobs")

    if failed_rows:
        print("\nFailed jobs:")
        failed_rows.sort(key=lambda item: (item["category"], item["env_id"], item["seed"]))
        for job in failed_rows[:12]:
            print(f"  {job['category']}/{job['env_id']}/seed_{job['seed']}")

    pending_rows = [job for job in jobs if job["status"] == "pending"]
    if pending_rows:
        print("\nNext pending jobs:")
        pending_rows.sort(key=lambda item: (item["category"], item["env_id"], item["seed"]))
        for job in pending_rows[:12]:
            print(f"  {job['category']}/{job['env_id']}/seed_{job['seed']}")


if __name__ == "__main__":
    main()
