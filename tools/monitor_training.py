"""Bounded, token-free first-hour monitor. Never judges early driving performance."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import shutil
import time


def inspect_health(health, now, previous, last_progress):
    problems = []
    if now - health["timestamp"] > 180:
        problems.append("stale_heartbeat")
    if health["actors_alive"] != health["actors_expected"]:
        problems.append("missing_actor")
    if any(health.get(k) is not None and not math.isfinite(health[k])
           for k in ("alpha", "q_loss", "policy_loss")):
        problems.append("nonfinite_loss")
    progress = (health["transitions"], health["updates"])
    if progress != previous:
        last_progress = now
    elif now - last_progress > 300:
        problems.append("no_progress")
    return problems, progress, last_progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", action="append", nargs=2, metavar=("UNIT", "DIRECTORY"), required=True)
    parser.add_argument("--duration", type=float, default=3600)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    deadline = time.monotonic() + args.duration
    states = {unit: {"previous": None, "last_progress": started, "restarts": 0}
              for unit, _ in args.session}
    args.output.mkdir(parents=True, exist_ok=True)
    error = None
    try:
        # No model calls, scheduled agent wakeups or messages to other services.
        while time.monotonic() < deadline:
            snapshot = {"timestamp": time.time(), "sessions": {}}
            meminfo = {line.split(':')[0]: int(line.split()[1])
                       for line in Path('/proc/meminfo').read_text().splitlines()}
            snapshot["memory_available_GiB"] = meminfo["MemAvailable"] / 1024**2
            snapshot["disk_free_GiB"] = shutil.disk_usage(args.output).free / 1024**3
            for unit, directory in args.session:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                now = time.time()
                state = states[unit]
                problems, health, active = [], None, "unknown"
                try:
                    active = subprocess.run(["systemctl", "--user", "is-active", unit],
                                            capture_output=True, text=True,
                                            timeout=min(10, remaining)).stdout.strip()
                except (OSError, subprocess.TimeoutExpired) as exc:
                    problems.append("status_query_" + type(exc).__name__)
                try:
                    health = json.loads((Path(directory) / "health.json").read_text())
                    issues, state["previous"], state["last_progress"] = inspect_health(
                        health, now, state["previous"], state["last_progress"])
                    problems.extend(issues)
                except (OSError, ValueError, KeyError, TypeError):
                    if now - started > 180:
                        problems.append("missing_or_invalid_heartbeat")
                if active not in ("active", "activating") and now - started > 180:
                    problems.append("service_inactive")
                remaining = deadline - time.monotonic()
                if problems and now - started > 180 and state["restarts"] < 1 and remaining > 0:
                    # Count before calling: timeout must not cause unlimited retries.
                    state["restarts"] += 1
                    state["last_progress"] = now
                    try:
                        result = subprocess.run(["systemctl", "--user", "restart", unit],
                                                capture_output=True, text=True,
                                                timeout=min(90, remaining))
                        problems.append(f"restart_returncode={result.returncode}")
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        problems.append("restart_" + type(exc).__name__)
                snapshot["sessions"][unit] = {"active": active, "health": health,
                                              "problems": problems, "restarts": state["restarts"]}
            with (args.output / "checks.jsonl").open("a") as handle:
                handle.write(json.dumps(snapshot) + "\n")
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(args.interval, remaining))
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        (args.output / "finished.json").write_text(json.dumps({
            "started_at": started, "finished_at": time.time(), "states": states,
            "monitor_stopped": True, "training_stop_requested": False, "error": error,
        }, indent=2))


if __name__ == "__main__":
    main()
