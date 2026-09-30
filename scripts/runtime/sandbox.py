from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    wall_seconds: int
    memory_mb: int
    cpus: float
    pids: int


def clean_environment(python_path: str | None = None) -> dict[str, str]:
    allowed = ("SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "TEMP", "TMP")
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    env.update({"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"})
    if python_path:
        env["WORLDSOLVER_PYTHON"] = python_path
        python_dir = str(Path(python_path).resolve().parent)
        additions = [
            python_dir,
            str(Path(python_dir) / "Scripts"),
            str(Path(python_dir) / "Library/bin"),
        ]
        env["PATH"] = os.pathsep.join([*additions, env.get("PATH", "")])
    return env


@dataclass(slots=True)
class ResourceUsage:
    wall_seconds: float = 0.0
    peak_rss_bytes: int = 0
    cpu_seconds: float = 0.0
    exit_code: int | None = None
    timed_out: bool = False


def run_monitored(
    command: list[str], cwd: Path, timeout_seconds: float, output_path: Path, env: dict[str, str]
) -> ResourceUsage:
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, env=env)
    usage = ResourceUsage()
    try:
        root = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        usage.exit_code = process.wait()
        usage.wall_seconds = time.monotonic() - started
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(asdict(usage), indent=2, sort_keys=True), encoding="utf-8"
        )
        return usage
    try:
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > timeout_seconds:
                usage.timed_out = True
                _kill_tree(root)
                break
            rss, cpu = 0, 0.0
            try:
                processes = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                break
            for child in processes:
                try:
                    rss += child.memory_info().rss
                    times = child.cpu_times()
                    cpu += times.user + times.system
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            usage.peak_rss_bytes = max(usage.peak_rss_bytes, rss)
            usage.cpu_seconds = max(usage.cpu_seconds, cpu)
            time.sleep(0.1)
    except KeyboardInterrupt:
        _kill_tree(root)
        process.wait()
        raise
    try:
        usage.exit_code = process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _kill_tree(root)
        usage.exit_code = process.wait()
    usage.wall_seconds = time.monotonic() - started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(usage), indent=2, sort_keys=True), encoding="utf-8")
    return usage


def _kill_tree(root: psutil.Process) -> None:
    try:
        children = root.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for process in children:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        root.kill()
    except psutil.NoSuchProcess:
        pass
