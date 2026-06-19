from routelabs_router.hardware import MachineProfile, recommend_local_model


def test_recommend_local_model_for_small_cpu_machine() -> None:
    recommendation = recommend_local_model(
        MachineProfile(
            os_name="Linux",
            arch="x86_64",
            cpu_count=4,
            memory_gb=8.0,
        )
    )

    assert recommendation.model == "llama3.2:3b"
    assert recommendation.embedding_model == "embeddinggemma"
    assert "ollama pull llama3.2:3b" in recommendation.pull_commands


def test_recommend_local_model_for_apple_silicon_balanced_machine() -> None:
    recommendation = recommend_local_model(
        MachineProfile(
            os_name="Darwin",
            arch="arm64",
            cpu_count=10,
            memory_gb=24.0,
            accelerator="apple-silicon",
            gpu_name="Apple Silicon unified memory",
            gpu_memory_gb=24.0,
        ),
        workload="agent",
    )

    assert recommendation.model == "qwen3:4b"
    assert recommendation.profile == "agent-balanced"


def test_recommend_local_model_for_coding_gpu_machine() -> None:
    recommendation = recommend_local_model(
        MachineProfile(
            os_name="Linux",
            arch="x86_64",
            cpu_count=16,
            memory_gb=32.0,
            accelerator="nvidia-gpu",
            gpu_name="NVIDIA RTX",
            gpu_memory_gb=12.0,
        ),
        workload="coding",
    )

    assert recommendation.model == "qwen2.5-coder:7b"
    assert recommendation.profile == "coding-performance"
