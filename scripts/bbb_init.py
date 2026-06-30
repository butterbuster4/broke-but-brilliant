#!/usr/bin/env python
"""Initialize a local Broke but Brilliant reproduction context."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


UNKNOWN = "unknown"


def bytes_to_gib(value: int | None) -> float | str:
    if value is None:
        return UNKNOWN
    return round(value / (1024**3), 2)


def run_command(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def run_command_with_status(args: list[str], cwd: Path) -> tuple[bool, str | None]:
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False, None
    return result.returncode == 0, result.stdout.strip() or None


def find_git_root(target: Path) -> Path | None:
    git_root = run_command(["git", "rev-parse", "--show-toplevel"], target)
    if git_root:
        return Path(git_root).resolve()

    for parent in [target, *target.parents]:
        if (parent / ".git").exists():
            return parent.resolve()
    return None


def get_cpu_model() -> str:
    system = platform.system().lower()

    if system == "linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()

    if system == "darwin":
        value = run_command(["sysctl", "-n", "machdep.cpu.brand_string"], Path.cwd())
        if value:
            return value

    if system == "windows":
        value = platform.processor()
        if value:
            return value

    return platform.processor() or UNKNOWN


def get_ram_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total)
    except Exception:
        pass

    if platform.system().lower() == "windows":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages)
    except (AttributeError, OSError, ValueError):
        return None


def get_torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "installed": False,
        "version": UNKNOWN,
        "build": UNKNOWN,
        "cuda_available": False,
        "cuda_version": UNKNOWN,
        "gpus": [],
    }

    try:
        import torch  # type: ignore
    except Exception as exc:
        info["import_error"] = type(exc).__name__
        return info

    info["installed"] = True
    version = getattr(torch, "__version__", UNKNOWN)
    info["version"] = version
    if isinstance(version, str) and "+" in version:
        info["build"] = version.rsplit("+", 1)[1]
    elif getattr(torch.version, "cuda", None):
        info["build"] = "cuda"
    else:
        info["build"] = UNKNOWN
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_version"] = getattr(torch.version, "cuda", None) or UNKNOWN

    if info["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["gpus"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_vram_gib": bytes_to_gib(int(props.total_memory)),
                }
            )

    return info


def get_nvidia_smi_gpus(target: Path) -> tuple[bool, list[dict[str, Any]]]:
    ok, output = run_command_with_status(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        target,
    )
    if not ok or not output:
        return False, []

    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        memory_mib: int | str
        try:
            memory_mib = int(parts[2])
        except ValueError:
            memory_mib = UNKNOWN
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "total_vram_mib": memory_mib,
                "driver_version": parts[3],
            }
        )

    return True, gpus


def get_windows_gpus(target: Path) -> list[dict[str, Any]]:
    if platform.system().lower() != "windows":
        return []

    ok, output = run_command_with_status(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
        ],
        target,
    )
    if not ok or not output:
        return []

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return []

    entries = parsed if isinstance(parsed, list) else [parsed]
    gpus: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("Name") or UNKNOWN
        adapter_ram = entry.get("AdapterRAM")
        try:
            adapter_ram_gib = bytes_to_gib(int(adapter_ram)) if adapter_ram is not None else UNKNOWN
        except (TypeError, ValueError):
            adapter_ram_gib = UNKNOWN
        gpus.append({"name": name, "adapter_ram_gib": adapter_ram_gib})

    return gpus


def interpret_gpu_environment(
    torch_info: dict[str, Any],
    nvidia_smi_available: bool,
    nvidia_smi_gpus: list[dict[str, Any]],
    windows_gpus: list[dict[str, Any]],
    cuda_visible_devices: str | None,
) -> tuple[bool, str]:
    torch_gpus = torch_info.get("gpus") or []
    any_gpu_detected = bool(nvidia_smi_gpus or windows_gpus or torch_gpus)
    nvidia_seen = bool(nvidia_smi_gpus) or any(
        "nvidia" in str(gpu.get("name", "")).lower() for gpu in windows_gpus + torch_gpus
    )
    torch_installed = bool(torch_info.get("installed"))
    torch_cuda_available = bool(torch_info.get("cuda_available"))
    torch_cuda_runtime = torch_info.get("cuda_version") not in (None, "", UNKNOWN)
    torch_build = str(torch_info.get("build", UNKNOWN)).lower()
    cuda_hidden = cuda_visible_devices in ("", "-1", "none", "None")

    if not any_gpu_detected:
        return False, "no GPU hardware detected by nvidia-smi, Windows GPU detection, or PyTorch"

    if nvidia_seen and torch_cuda_available:
        return True, "NVIDIA GPU exists and PyTorch CUDA works"

    if nvidia_seen and torch_installed and ("cpu" in torch_build or not torch_cuda_runtime):
        return True, "NVIDIA GPU exists but PyTorch appears to be CPU-only"

    if nvidia_seen and torch_installed and torch_cuda_runtime and not torch_cuda_available:
        if cuda_hidden:
            return True, "GPU exists but CUDA_VISIBLE_DEVICES appears to hide CUDA devices from PyTorch"
        return True, "GPU exists but driver/CUDA/PyTorch environment is likely misconfigured"

    if nvidia_seen and not nvidia_smi_available:
        return True, "NVIDIA GPU is visible to the OS, but nvidia-smi is unavailable; driver or PATH may be misconfigured"

    if nvidia_seen:
        return True, "NVIDIA GPU hardware detected, but PyTorch CUDA status is unavailable or not working"

    return True, "GPU hardware detected, but no NVIDIA CUDA-capable GPU was identified"


def detect_runtime_hints(target: Path) -> list[str]:
    hints: list[str] = []
    env = os.environ
    target_text = str(target).lower()

    if env.get("COLAB_RELEASE_TAG") or env.get("COLAB_GPU"):
        hints.append("colab")
    if env.get("KAGGLE_URL_BASE") or Path("/kaggle").exists():
        hints.append("kaggle")
    if Path("/.dockerenv").exists():
        hints.append("docker")
    if "microsoft" in Path("/proc/version").read_text(errors="ignore").lower() if Path("/proc/version").exists() else False:
        hints.append("wsl")
    if "autodl" in target_text or Path("/root/autodl-tmp").exists() or Path("/root/autodl-fs").exists():
        hints.append("autodl-like")

    return sorted(set(hints)) or ["unknown"]


def build_profile(target: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(target)
    git_root = find_git_root(target)
    torch_info = get_torch_info()
    nvidia_smi_available, nvidia_smi_gpus = get_nvidia_smi_gpus(target)
    windows_gpus = get_windows_gpus(target)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_hardware_detected, gpu_interpretation = interpret_gpu_environment(
        torch_info,
        nvidia_smi_available,
        nvidia_smi_gpus,
        windows_gpus,
        cuda_visible_devices,
    )
    ram_bytes = get_ram_bytes()

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "privacy_note": "Does not intentionally collect secrets, tokens, usernames, or environment variable dumps.",
        "working_directory": str(target),
        "os": {
            "system": platform.system() or UNKNOWN,
            "release": platform.release() or UNKNOWN,
            "version": platform.version() or UNKNOWN,
            "platform": platform.platform() or UNKNOWN,
        },
        "python": {
            "version": platform.python_version(),
            "executable_basename": Path(sys.executable).name,
        },
        "cpu": {
            "model": get_cpu_model(),
            "core_count_logical": os.cpu_count() or UNKNOWN,
        },
        "ram": {
            "total_gib": bytes_to_gib(ram_bytes),
        },
        "disk_for_working_directory": {
            "total_gib": bytes_to_gib(disk.total),
            "free_gib": bytes_to_gib(disk.free),
        },
        "git": {
            "inside_repository": git_root is not None,
            "root": str(git_root) if git_root else UNKNOWN,
        },
        "pytorch": torch_info,
        "gpu_hardware_detected": gpu_hardware_detected,
        "nvidia_smi_available": nvidia_smi_available,
        "nvidia_smi_gpus": nvidia_smi_gpus,
        "windows_gpus": windows_gpus,
        "torch_cuda_available": torch_info["cuda_available"],
        "torch_build": torch_info["build"],
        "cuda_runtime_from_torch": torch_info["cuda_version"],
        "cuda_visible_devices": "set" if cuda_visible_devices is not None else "not set",
        "gpu_interpretation": gpu_interpretation,
        "runtime_hints": detect_runtime_hints(target),
    }


def collect_missing_info(profile: dict[str, Any]) -> list[str]:
    missing: list[str] = []

    if profile["cpu"]["model"] == UNKNOWN:
        missing.append("CPU model")
    if profile["ram"]["total_gib"] == UNKNOWN:
        missing.append("RAM total")
    if not profile["pytorch"]["installed"]:
        missing.append("PyTorch version and CUDA/GPU details from PyTorch")
    if profile["gpu_hardware_detected"] and not profile["nvidia_smi_available"]:
        missing.append("NVIDIA driver status from nvidia-smi")
    if profile["gpu_hardware_detected"] and not profile["torch_cuda_available"]:
        missing.append("Working PyTorch CUDA environment")
    if profile["runtime_hints"] == ["unknown"]:
        missing.append("cloud/runtime provider")

    return missing


def write_missing_info(path: Path, missing: list[str]) -> None:
    lines = [
        "# Missing Reproduction Context",
        "",
        "The initialization script could not detect every field automatically.",
        "",
    ]

    if missing:
        lines.append("## Missing or Unknown")
        lines.extend(f"- {item}" for item in missing)
    else:
        lines.append("No major hardware/environment fields were missing.")

    lines.extend(
        [
            "",
            "## Still Needed From User or Paper",
            "",
            "- Paper PDF/arXiv link or method section",
            "- Official author repository link, if available",
            "- Target table, figure, metric, or claim",
            "- Dataset access path and expected splits",
            "- Time, storage, and budget constraints",
            "- Whether you have access to cloud GPUs such as AutoDL, Colab, Kaggle, RunPod, Vast.ai, or university servers",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_repro_context(path: Path, profile: dict[str, Any]) -> None:
    torch_info = profile["pytorch"]
    torch_gpu_lines = torch_info["gpus"] or []
    smi_gpu_lines = profile["nvidia_smi_gpus"] or []
    windows_gpu_lines = profile["windows_gpus"] or []
    if smi_gpu_lines:
        gpu_summary = "\n".join(
            f"- nvidia-smi GPU {gpu['index']}: {gpu['name']} ({gpu['total_vram_mib']} MiB), driver {gpu['driver_version']}"
            for gpu in smi_gpu_lines
        )
    elif torch_gpu_lines:
        gpu_summary = "\n".join(
            f"- PyTorch GPU {gpu['index']}: {gpu['name']} ({gpu['total_vram_gib']} GiB)"
            for gpu in torch_gpu_lines
        )
    elif windows_gpu_lines:
        gpu_summary = "\n".join(
            f"- Windows GPU: {gpu['name']} ({gpu['adapter_ram_gib']} GiB adapter RAM)"
            for gpu in windows_gpu_lines
        )
    else:
        gpu_summary = "- No GPU hardware detected by available probes"

    lines = [
        "# Reproduction Context",
        "",
        "## Hardware Facts",
        "",
        f"- OS: {profile['os']['platform']}",
        f"- Python: {profile['python']['version']}",
        f"- CPU: {profile['cpu']['model']}",
        f"- Logical CPU cores: {profile['cpu']['core_count_logical']}",
        f"- RAM: {profile['ram']['total_gib']} GiB",
        f"- Working directory disk: {profile['disk_for_working_directory']['free_gib']} GiB free / {profile['disk_for_working_directory']['total_gib']} GiB total",
        f"- Runtime hints: {', '.join(profile['runtime_hints'])}",
        f"- GPU interpretation: {profile['gpu_interpretation']}",
        "",
        "## PyTorch Facts",
        "",
        f"- PyTorch installed: {torch_info['installed']}",
        f"- PyTorch version: {torch_info['version']}",
        f"- PyTorch build: {profile['torch_build']}",
        f"- CUDA available through PyTorch: {torch_info['cuda_available']}",
        f"- CUDA version reported by PyTorch: {torch_info['cuda_version']}",
        f"- nvidia-smi available: {profile['nvidia_smi_available']}",
        f"- CUDA_VISIBLE_DEVICES: {profile['cuda_visible_devices']}",
        "",
        "## GPUs",
        "",
        gpu_summary,
        "",
        "## Repository Facts",
        "",
        f"- Current working directory: {profile['working_directory']}",
        f"- Inside Git repository: {profile['git']['inside_repository']}",
        f"- Git root: {profile['git']['root']}",
        "",
        "## Paper Facts",
        "",
        "- Not collected by this script. Add paper-grounded facts before planning.",
        "",
        "## Estimates",
        "",
        "- None yet. Any future estimates must be explicitly labeled as estimates.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def ensure_gitignore(git_root: Path | None) -> None:
    if git_root is None:
        return

    gitignore = git_root / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    lines = [line.strip() for line in existing.splitlines()]
    if ".bbb/" in lines or ".bbb" in lines:
        return

    suffix = "" if not existing or existing.endswith(("\n", "\r")) else "\n"
    gitignore.write_text(f"{existing}{suffix}.bbb/\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize Broke but Brilliant reproduction context.")
    parser.add_argument(
        "--target",
        default=".",
        help="Target repository or working directory. Defaults to the current directory.",
    )
    args = parser.parse_args()

    target = Path(args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    bbb_dir = target / ".bbb"
    bbb_dir.mkdir(exist_ok=True)

    profile = build_profile(target)
    missing = collect_missing_info(profile)

    (bbb_dir / "hardware_profile.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_missing_info(bbb_dir / "missing_info.md", missing)
    write_repro_context(bbb_dir / "repro_context.md", profile)

    git_root = Path(profile["git"]["root"]) if profile["git"]["inside_repository"] else None
    ensure_gitignore(git_root)

    print(f"Wrote {bbb_dir / 'hardware_profile.json'}")
    print(f"Wrote {bbb_dir / 'missing_info.md'}")
    print(f"Wrote {bbb_dir / 'repro_context.md'}")
    if git_root:
        print(f"Ensured .bbb/ is ignored in {git_root / '.gitignore'}")
    else:
        print("Target is not inside a Git repository; .gitignore was not changed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
