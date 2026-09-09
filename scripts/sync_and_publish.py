#!/usr/bin/env python3
"""Synchronize publishable experiment artifacts and optionally push them."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
from pathlib import Path


SOURCE = Path("/home/songzhoujie/cvpr27/vstar")
DESTINATION = Path("/home/songzhoujie/cvpr27/visual-search-context-pollution-experiments")
MAX_FILE_BYTES = 15 * 1024 * 1024

CODE_SUFFIXES = {".py", ".sh"}
REPORT_SUFFIXES = {".csv", ".json", ".md", ".png", ".txt"}
EXCLUDED_DIR_NAMES = {
    "cache",
    "contact_sheets",
    "contact_sheets_v2",
    "display_images",
    "official_judge_raw",
    "raw_episodes",
    "shards",
    "trajectory_vis",
}
EXCLUDED_FILE_NAMES = {
    "crop_records.csv",
    "model_input_audit.jsonl",
    "trajectories.jsonl",
    "turn_records.csv",
}
SECRET_PATTERNS = {
    "Google-style API key": re.compile(r"\b(?:AIza|AQ\.)[A-Za-z0-9_.-]{20,}"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}"),
    "OpenAI-style API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
}


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=DESTINATION, text=True, check=check)


def excluded_path(relative_path: Path) -> bool:
    for part in relative_path.parts[:-1]:
        lowered = part.lower()
        if lowered in EXCLUDED_DIR_NAMES or lowered.startswith("crops"):
            return True
    if relative_path.name in EXCLUDED_FILE_NAMES:
        return True
    return re.fullmatch(r"v21_full_seed\d+_shard\d+\.csv", relative_path.name) is not None


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def synchronize() -> list[Path]:
    manifest = DESTINATION / "MANIFEST.txt"
    previously_managed: set[Path] = set()
    if manifest.exists():
        previously_managed = {
            Path(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        }

    copied: list[Path] = []

    for source in sorted(SOURCE.iterdir()):
        if source.is_file() and source.suffix.lower() in CODE_SUFFIXES:
            destination = DESTINATION / "code" / source.name
            copy_file(source, destination)
            copied.append(destination.relative_to(DESTINATION))

    for documentation in (SOURCE / "README_experiment.md", SOURCE / "ROUND6_MEMORY_SURGERY_README.md"):
        if documentation.exists():
            destination = DESTINATION / "docs" / documentation.name
            copy_file(documentation, destination)
            copied.append(destination.relative_to(DESTINATION))

    output_root = SOURCE / "outputs"
    for source in sorted(output_root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(output_root)
        if excluded_path(relative):
            continue
        if source.suffix.lower() not in REPORT_SUFFIXES:
            continue
        if source.stat().st_size > MAX_FILE_BYTES:
            continue
        destination = DESTINATION / "reports" / relative
        copy_file(source, destination)
        copied.append(destination.relative_to(DESTINATION))

    current = set(copied)
    for relative in sorted(previously_managed - current):
        stale = DESTINATION / relative
        if stale.is_file() and stale.resolve().is_relative_to(DESTINATION.resolve()):
            stale.unlink()

    for directory in sorted(
        (path for path in (DESTINATION / "reports").rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        if not any(directory.iterdir()):
            directory.rmdir()

    manifest.write_text(
        "# Files synchronized from /home/songzhoujie/cvpr27/vstar\n"
        + "\n".join(path.as_posix() for path in copied)
        + "\n",
        encoding="utf-8",
    )
    return copied


def scan_for_secrets(paths: list[Path]) -> None:
    violations: list[str] = []
    for relative in paths:
        path = DESTINATION / relative
        if path.suffix.lower() == ".png":
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                violations.append(f"{relative}: {label}")
    if violations:
        details = "\n".join(f"- {item}" for item in violations)
        raise RuntimeError(f"Potential secrets found; refusing to publish:\n{details}")


def commit_and_push() -> None:
    run(["git", "add", "--all"])
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=DESTINATION
    ).returncode != 0
    if not changed:
        print("No report changes to publish.")
        return
    timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    run(["git", "commit", "-m", f"Update experiment reports ({timestamp})"])
    run(["git", "push", "origin", "main"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="Commit and push synchronized files")
    args = parser.parse_args()

    copied = synchronize()
    scan_for_secrets(copied)
    print(f"Synchronized and checked {len(copied)} publishable files.")
    if args.push:
        commit_and_push()


if __name__ == "__main__":
    main()
