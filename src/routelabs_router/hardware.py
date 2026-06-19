import os
import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class MachineProfile:
    os_name: str
    arch: str
    cpu_count: int
    memory_gb: float | None
    accelerator: str = "cpu"
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None


@dataclass(frozen=True)
class LocalModelRecommendation:
    profile: str
    model: str
    embedding_model: str
    reason: str
    pull_commands: list[str]
    config_hint: dict[str, str]


def detect_machine_profile() -> MachineProfile:
    os_name = platform.system() or "unknown"
    arch = platform.machine() or "unknown"
    cpu_count = os.cpu_count() or 1
    memory_gb = _detect_memory_gb()
    gpu_name, gpu_memory_gb = _detect_nvidia_gpu()
    accelerator = "nvidia-gpu" if gpu_name else "cpu"

    if not gpu_name and os_name == "Darwin" and arch in {"arm64", "aarch64"}:
        accelerator = "apple-silicon"
        gpu_name = "Apple Silicon unified memory"
        gpu_memory_gb = memory_gb

    return MachineProfile(
        os_name=os_name,
        arch=arch,
        cpu_count=cpu_count,
        memory_gb=memory_gb,
        accelerator=accelerator,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
    )


def recommend_local_model(
    profile: MachineProfile,
    workload: str = "general",
) -> LocalModelRecommendation:
    memory_gb = profile.memory_gb or 0
    effective_memory_gb = max(memory_gb, profile.gpu_memory_gb or 0)

    if workload == "coding":
        model, tier = _coding_model_for(effective_memory_gb)
    elif workload == "agent":
        model, tier = _agent_model_for(effective_memory_gb)
    else:
        model, tier = _general_model_for(effective_memory_gb)

    embedding_model = "embeddinggemma"
    reason = (
        f"{tier} recommendation for {profile.cpu_count} CPU core(s), "
        f"{_format_memory(profile.memory_gb)} RAM"
    )
    if profile.gpu_name:
        reason += f", {profile.gpu_name}"
        if profile.gpu_memory_gb is not None:
            reason += f" ({profile.gpu_memory_gb:.1f} GB available memory)"

    return LocalModelRecommendation(
        profile=tier,
        model=model,
        embedding_model=embedding_model,
        reason=reason,
        pull_commands=[
            "ollama serve",
            f"ollama pull {model}",
            f"ollama pull {embedding_model}",
        ],
        config_hint={
            "providers.local.default": "ollama",
            "providers.local.ollama.model": model,
            "providers.local.ollama.embedding_model": embedding_model,
        },
    )


def _general_model_for(memory_gb: float) -> tuple[str, str]:
    if memory_gb >= 48:
        return "qwen3:8b", "performance"
    if memory_gb >= 16:
        return "qwen3:4b", "balanced"
    if memory_gb >= 8:
        return "llama3.2:3b", "small"
    return "llama3.2:1b", "tiny"


def _coding_model_for(memory_gb: float) -> tuple[str, str]:
    if memory_gb >= 32:
        return "qwen2.5-coder:7b", "coding-performance"
    if memory_gb >= 16:
        return "qwen2.5-coder:3b", "coding-balanced"
    if memory_gb >= 8:
        return "qwen2.5-coder:1.5b", "coding-small"
    return "llama3.2:1b", "tiny"


def _agent_model_for(memory_gb: float) -> tuple[str, str]:
    if memory_gb >= 32:
        return "qwen3:8b", "agent-performance"
    if memory_gb >= 12:
        return "qwen3:4b", "agent-balanced"
    return "llama3.2:3b", "agent-small"


def _detect_memory_gb() -> float | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(page_size, int) or not isinstance(page_count, int):
        return None
    return round((page_size * page_count) / (1024**3), 1)


def _detect_nvidia_gpu() -> tuple[str | None, float | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None, None
    if result.returncode != 0 or not result.stdout.strip():
        return None, None
    first_line = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    if not parts:
        return None, None
    gpu_name = parts[0]
    gpu_memory_gb = None
    if len(parts) > 1:
        try:
            gpu_memory_gb = round(float(parts[1]) / 1024, 1)
        except ValueError:
            gpu_memory_gb = None
    return gpu_name, gpu_memory_gb


def _format_memory(memory_gb: float | None) -> str:
    if memory_gb is None:
        return "unknown"
    return f"{memory_gb:.1f} GB"
