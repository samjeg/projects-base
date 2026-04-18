#!/usr/bin/env python
"""Validate that every package/CLI the base image promises actually works.

Run on any pod:    python /workspace/scripts/check_env.py
Or over SSH:       ssh root@POD 'python -' < scripts/check_env.py

Exit code is non-zero if any check fails, so this can be wired into CI or
a smoke-test step before launching real jobs.

Keep the IMPORT_CHECKS and CLI_CHECKS lists aligned with the pip installs
and apt installs in the Dockerfile. When the base grows a new tool, add
it here so broken transitive deps fail the check loudly.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass


# (import_name, display_name) — display_name differs when pypi name != import name
IMPORT_CHECKS: list[tuple[str, str]] = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("PIL", "pillow"),
    ("cv2", "opencv-python-headless"),
    ("imageio", "imageio"),
    ("imageio_ffmpeg", "imageio-ffmpeg"),
    ("av", "av"),
    ("soundfile", "soundfile"),
    ("librosa", "librosa"),
    ("einops", "einops"),
    ("safetensors", "safetensors"),
    ("huggingface_hub", "huggingface_hub"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("diffusers", "diffusers"),
    ("sentencepiece", "sentencepiece"),
    ("google.protobuf", "protobuf"),
    ("tqdm", "tqdm"),
    ("rich", "rich"),
    ("pydantic", "pydantic"),
    ("httpx", "httpx"),
    ("dotenv", "python-dotenv"),
    ("typer", "typer"),
    ("whisper", "openai-whisper"),
    ("rembg", "rembg"),
    ("rembg.cli", "rembg.cli"),
    ("onnxruntime", "onnxruntime-gpu"),
]

# (args, display_name) — runs each CLI with a help/version flag; exit 0 == ok.
CLI_CHECKS: list[tuple[list[str], str]] = [
    (["python", "--version"], "python"),
    (["ffmpeg", "-version"], "ffmpeg"),
    (["ffprobe", "-version"], "ffprobe"),
    (["whisper", "--help"], "whisper"),
    (["rembg", "--help"], "rembg"),
    (["huggingface-cli", "--help"], "huggingface-cli"),
    (["accelerate", "--help"], "accelerate"),
]

# sshd -V exits 1 by design, so we just check it's on PATH, not executable success.
PATH_ONLY_CHECKS: list[str] = ["sshd"]


@dataclass
class Result:
    name: str
    ok: bool
    detail: str = ""


def check_import(import_name: str, display_name: str) -> Result:
    try:
        mod = importlib.import_module(import_name)
    except Exception as exc:
        return Result(display_name, False, f"{type(exc).__name__}: {exc}")
    version = getattr(mod, "__version__", "")
    return Result(display_name, True, f"v{version}" if version else "ok")


def check_cli(args: list[str], display_name: str) -> Result:
    binary = args[0]
    if shutil.which(binary) is None:
        return Result(display_name, False, "not on PATH")
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return Result(display_name, False, "timed out")
    except Exception as exc:
        return Result(display_name, False, f"{type(exc).__name__}: {exc}")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:]
        return Result(display_name, False, f"exit {proc.returncode}: {tail[0] if tail else ''}")
    return Result(display_name, True, "exit 0")


def check_path_only(binary: str) -> Result:
    path = shutil.which(binary)
    if path is None:
        return Result(binary, False, "not on PATH")
    return Result(binary, True, path)


def check_gpu() -> list[Result]:
    try:
        import torch
    except Exception as exc:
        return [Result("torch (for gpu check)", False, str(exc))]
    results = [
        Result(
            "torch.cuda.is_available",
            torch.cuda.is_available(),
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU",
        )
    ]
    try:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        results.append(
            Result(
                "onnxruntime CUDAExecutionProvider",
                "CUDAExecutionProvider" in providers,
                ", ".join(providers),
            )
        )
    except Exception as exc:
        results.append(Result("onnxruntime.providers", False, str(exc)))
    return results


def format_row(result: Result) -> str:
    mark = "OK  " if result.ok else "FAIL"
    return f"  [{mark}] {result.name:<34} {result.detail}"


def main() -> int:
    sections: list[tuple[str, list[Result]]] = [
        ("Python imports", [check_import(i, d) for i, d in IMPORT_CHECKS]),
        ("CLI entry points", [check_cli(a, d) for a, d in CLI_CHECKS]),
        ("PATH-only checks", [check_path_only(b) for b in PATH_ONLY_CHECKS]),
        ("GPU runtime", check_gpu()),
    ]

    total_fail = 0
    total_count = 0
    for title, results in sections:
        print(f"\n== {title} ==")
        for r in results:
            print(format_row(r))
            total_count += 1
            if not r.ok:
                total_fail += 1

    print()
    if total_fail == 0:
        print(f"All {total_count} checks passed.")
        return 0
    print(f"{total_fail} of {total_count} check(s) failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
