#!/usr/bin/env python
"""Scan a child project directory for every import, subprocess invocation,
ffmpeg codec/filter, and shell-tool reference across .py and .md files.

Cross-references against what projects-base is known to provide. Anything
that's not stdlib, not in the base, and not obviously system-ubiquitous
gets flagged before you push.

Run from projects-base repo root:
    python scripts/scan_child_deps.py ../tiktok-man-auto
"""

from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# What the base image ships (keep aligned with projects-base Dockerfile).
# ---------------------------------------------------------------------------

BASE_PYTHON_PACKAGES: set[str] = {
    # PyTorch stack
    "torch", "torchvision", "torchaudio",
    # Core numerical / imaging / audio
    "numpy", "scipy", "PIL", "cv2", "imageio", "imageio_ffmpeg", "av",
    "soundfile", "librosa",
    # ML ecosystem
    "einops", "safetensors", "huggingface_hub", "transformers",
    "accelerate", "diffusers", "sentencepiece", "google",  # google.protobuf
    # Utilities
    "tqdm", "rich", "pydantic", "httpx", "dotenv", "typer",
    # Video/audio ML tooling
    "whisper", "rembg", "onnxruntime",
    # CLI deps pulled in via rembg[cli]
    "aiohttp", "asyncer", "click", "fastapi", "filetype", "gradio",
    "python_multipart", "sniffio", "uvicorn", "watchdog",
}

BASE_CLI_TOOLS: set[str] = {
    # System binaries
    "ffmpeg", "ffprobe", "python", "python3", "python3.13",
    "git", "git-lfs", "bash", "sh", "curl", "wget", "sshd", "ssh-keygen",
    # Python-installed CLIs
    "whisper", "rembg", "huggingface-cli", "accelerate",
}

# ffmpeg codecs/filters known to ship in Ubuntu 22.04's ffmpeg 4.4.2 package.
# Not exhaustive — just what covers typical video editing workflows.
BASE_FFMPEG_FEATURES: set[str] = {
    # video encoders
    "libx264", "libx265", "libvpx", "libvpx-vp9", "libaom-av1",
    "mpeg4", "mjpeg", "prores", "prores_ks", "png", "gif",
    # audio encoders
    "aac", "libmp3lame", "libopus", "libvorbis", "flac", "pcm_s16le",
    # common filters (exhaustive list would be huge; these are the ones
    # most commonly surprised by)
    "scale", "crop", "cropdetect", "pad", "overlay", "drawtext",
    "subtitles", "ass", "chromakey", "colorkey", "format", "fps",
    "setpts", "concat", "xfade", "fade", "eq", "hue", "vflip", "hflip",
    "trim", "atrim", "asetpts", "amix", "adelay", "apad", "volume",
}

# Things Ubuntu 22.04 ffmpeg does NOT ship with by default — flag these loudly.
KNOWN_MISSING_IN_UBUNTU_FFMPEG: set[str] = {
    "h264_nvenc", "hevc_nvenc", "h264_cuvid", "hevc_cuvid",
    "libsvtav1", "libfdk_aac",
}

STDLIB_MODULES: set[str] = set(sys.stdlib_module_names)


# ---------------------------------------------------------------------------
# Scanners
# ---------------------------------------------------------------------------


def scan_python_imports(py_files: list[Path]) -> dict[str, list[Path]]:
    """Return {top_level_module: [files importing it]}."""
    imports: dict[str, list[Path]] = defaultdict(list)
    for path in py_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.name.split(".")[0]].append(path)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports[node.module.split(".")[0]].append(path)
    return imports


def scan_subprocess_tools(py_files: list[Path]) -> dict[str, list[Path]]:
    """Find every binary passed as argv[0] to subprocess calls."""
    tools: dict[str, list[Path]] = defaultdict(list)
    # Matches subprocess.run(["tool", ...]) or subprocess.Popen(["tool", ...])
    pattern = re.compile(
        r"subprocess\.\w+\s*\(\s*\[\s*[\"']([^\"']+)[\"']",
    )
    for path in py_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            tool = match.group(1)
            # Skip things that are clearly dynamic (sys.executable etc.) by
            # filtering on plausible binary names.
            if re.match(r"^[a-zA-Z][a-zA-Z0-9_.-]*$", tool):
                tools[tool].append(path)
    return tools


def scan_ffmpeg_usage(files: list[Path]) -> dict[str, set[str]]:
    """Extract codecs / filters mentioned in ffmpeg command strings across
    both .py and .md files. Heuristic but catches the common cases."""
    found: dict[str, set[str]] = {"codecs": set(), "filters": set()}

    # -c:v, -c:a, -vcodec, -acodec, -codec: all give us a codec token next
    codec_pattern = re.compile(r"-(?:c:[va]|vcodec|acodec|codec)\s+([a-zA-Z][\w-]*)")
    # -vf, -af, -filter:v, -filter:a: followed by a filter chain. Extract
    # each filter name (chars before = or , or ;).
    filter_pattern = re.compile(r"-(?:vf|af|filter:[va]|filter_complex)\s+[\"']?([^\"'\s][^\"']*)")

    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in codec_pattern.finditer(text):
            found["codecs"].add(m.group(1))
        for m in filter_pattern.finditer(text):
            chain = m.group(1)
            # Filter names appear as "name=..." or bare "name" separated by , ; [ ]
            for token in re.findall(r"([a-zA-Z][\w]*)(?=\s*[=,;\[\]\s])", chain):
                found["filters"].add(token)
    return found


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def classify_import(module: str) -> tuple[str, str]:
    """Return (severity, reason) — severity in {"ok", "warn", "fail"}."""
    if module in STDLIB_MODULES:
        return ("ok", "stdlib")
    if module in BASE_PYTHON_PACKAGES:
        return ("ok", "provided by projects-base")
    # Relative imports or project-local modules start lowercase single-word
    # and likely resolve within the repo
    return ("fail", "NOT provided by projects-base — add to child Dockerfile or base")


def classify_cli(tool: str) -> tuple[str, str]:
    if tool in BASE_CLI_TOOLS:
        return ("ok", "provided by projects-base")
    return ("fail", "NOT on PATH in projects-base")


def classify_ffmpeg_token(token: str, kind: str) -> tuple[str, str]:
    if token in KNOWN_MISSING_IN_UBUNTU_FFMPEG:
        return ("fail", "NOT in Ubuntu ffmpeg 4.4.2 — rebuild ffmpeg or swap codec")
    if token in BASE_FFMPEG_FEATURES:
        return ("ok", "in Ubuntu ffmpeg 4.4.2")
    return ("warn", f"unverified {kind} — check `ffmpeg -{kind}` on pod")


def short_path(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: scan_child_deps.py <child_repo_path>")
        return 2

    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 2

    py_files = [
        p for p in root.rglob("*.py")
        if ".git" not in p.parts and "runs" not in p.parts and "__pycache__" not in p.parts
    ]
    md_files = [p for p in root.rglob("*.md") if ".git" not in p.parts]

    print(f"Scanning {root}")
    print(f"  python files: {len(py_files)}")
    print(f"  markdown files: {len(md_files)}")

    fail_count = 0
    warn_count = 0

    # --- Python imports ---
    imports = scan_python_imports(py_files)
    print(f"\n== Python imports ({len(imports)} unique top-level) ==")
    for mod in sorted(imports):
        sev, reason = classify_import(mod)
        users = ", ".join(sorted({short_path(p, root) for p in imports[mod]})[:3])
        mark = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}[sev]
        print(f"  [{mark}] {mod:<20} {reason}  (used in: {users})")
        if sev == "fail":
            fail_count += 1
        elif sev == "warn":
            warn_count += 1

    # --- Subprocess tools ---
    tools = scan_subprocess_tools(py_files)
    print(f"\n== Subprocess CLI tools ({len(tools)} unique) ==")
    for tool in sorted(tools):
        sev, reason = classify_cli(tool)
        users = ", ".join(sorted({short_path(p, root) for p in tools[tool]})[:3])
        mark = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}[sev]
        print(f"  [{mark}] {tool:<20} {reason}  (used in: {users})")
        if sev == "fail":
            fail_count += 1

    # --- ffmpeg codecs/filters from both .py and .md ---
    ff = scan_ffmpeg_usage(py_files + md_files)
    print(f"\n== ffmpeg codecs referenced ({len(ff['codecs'])}) ==")
    for token in sorted(ff["codecs"]):
        sev, reason = classify_ffmpeg_token(token, "codec")
        mark = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}[sev]
        print(f"  [{mark}] {token:<20} {reason}")
        if sev == "fail":
            fail_count += 1
        elif sev == "warn":
            warn_count += 1

    print(f"\n== ffmpeg filters referenced ({len(ff['filters'])}) ==")
    for token in sorted(ff["filters"]):
        sev, reason = classify_ffmpeg_token(token, "filter")
        mark = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}[sev]
        print(f"  [{mark}] {token:<20} {reason}")
        if sev == "fail":
            fail_count += 1
        elif sev == "warn":
            warn_count += 1

    print()
    print(f"Summary: {fail_count} FAIL, {warn_count} WARN")
    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())
