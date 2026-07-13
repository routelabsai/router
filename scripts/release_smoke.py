from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and smoke-test the installed RouteLabs wheel."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python 3.11+ interpreter to use for building and the temp venv.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary directory for debugging.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python = Path(args.python)

    tmp_path = Path(tempfile.mkdtemp(prefix="routelabs-router-release-smoke-"))
    try:
        _run_smoke(repo_root=repo_root, python=python, tmp_path=tmp_path)
    finally:
        if args.keep_temp:
            print(f"Temp dir kept: {tmp_path}")
        else:
            shutil.rmtree(tmp_path, ignore_errors=True)


def _run_smoke(repo_root: Path, python: Path, tmp_path: Path) -> None:
    wheel_dir = tmp_path / "dist"
    venv_dir = tmp_path / "venv"
    output_config = tmp_path / "router.yaml"

    _run(
        [
            str(python),
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(wheel_dir),
        ],
        cwd=repo_root,
    )
    wheel = _single_wheel(wheel_dir)

    venv.EnvBuilder(with_pip=True).create(venv_dir)
    venv_python = _venv_python(venv_dir)
    router = _venv_script(venv_dir, "router")

    _run([str(venv_python), "-m", "pip", "install", str(wheel)], cwd=tmp_path)
    profiles = _run([str(router), "profiles"], cwd=tmp_path).stdout
    _assert_contains(profiles, "qwen-agent-mesh")
    _assert_contains(profiles, "local=ollama/qwen3:4b, roles=5")

    init_output = _run(
        [
            str(router),
            "init",
            "--profile",
            "qwen-agent-mesh",
            "--output",
            str(output_config),
            "--force",
        ],
        cwd=tmp_path,
    ).stdout
    _assert_contains(init_output, "Profile: qwen-agent-mesh")
    _assert_contains(output_config.read_text(encoding="utf-8"), "devstral:latest")

    import_output = _run(
        [
            str(venv_python),
            "-c",
            "import routelabs_router.server.app as app; print(app.app.title)",
        ],
        cwd=tmp_path,
    ).stdout
    _assert_contains(import_output, "RouteLabs Router")

    print("Release smoke passed")
    print(f"Wheel: {wheel}")


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command))
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def _single_wheel(wheel_dir: Path) -> Path:
    wheels = sorted(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one wheel in {wheel_dir}, found {wheels}")
    return wheels[0]


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_script(venv_dir: Path, name: str) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def _assert_contains(value: str, expected: str) -> None:
    if expected not in value:
        raise AssertionError(f"expected {expected!r} in output:\n{value}")


if __name__ == "__main__":
    main()
