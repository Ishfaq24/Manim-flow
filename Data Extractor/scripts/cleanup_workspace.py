"""Safely remove generated local artifacts after smoke tests.

Cleanup is intentionally whitelist-only and dry-run by default. Source folders,
configuration, package code, and project utilities are protected even if the
project layout changes later.
"""

from __future__ import annotations

import argparse
import shutil
import stat
from datetime import datetime
from pathlib import Path


GENERATED_TARGETS = [
    ".venv-train",
    "tmp",
    ".blocked-tmp",
    "pip-cache",
    "dataset",
    "data/processed/sample",
    "repos",
    "logs",
    "data/exports",
    "data/failed",
    "data/previews",
    "data/thumbnails",
    "models/manim-smoke-lora",
    "notebooks",
    "src/extractor/__pycache__",
    "scripts/__pycache__",
]

PROTECTED_ROOTS = [
    "src",
    "configs",
    "docs",
    "examples",
    "scripts",
    "tests",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely clean generated extractor artifacts.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually remove whitelisted generated artifacts. Without this, the script is a dry run.",
    )
    parser.add_argument(
        "--quarantine",
        action="store_true",
        help="Move artifacts into .cleanup-quarantine instead of deleting them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    removed: list[str] = []
    skipped: list[str] = []
    quarantine_root = root / ".cleanup-quarantine" / datetime.now().strftime("%Y%m%d-%H%M%S")

    for relative in GENERATED_TARGETS:
        path = (root / relative).resolve()
        safety_error = safety_check(root, path, relative)
        if safety_error:
            skipped.append(f"{relative} ({safety_error})")
            continue
        if not args.yes:
            skipped.append(f"{relative} (dry-run)")
            continue
        if args.quarantine:
            destination = quarantine_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
            removed.append(f"{relative} -> {destination.relative_to(root)}")
            continue
        delete_path(path)
        if path.exists():
            skipped.append(f"{relative} (permission locked)")
        else:
            removed.append(relative)

    print("Cleanup mode:", "delete" if args.yes and not args.quarantine else "quarantine" if args.yes else "dry-run")
    print("Removed:")
    for item in removed:
        print(f"  {item}")
    print("Skipped:")
    for item in skipped:
        print(f"  {item}")


def safety_check(root: Path, path: Path, relative: str) -> str | None:
    """Return a reason to skip unsafe cleanup targets."""

    try:
        path.relative_to(root)
    except ValueError:
        return f"outside workspace: {path}"
    if not path.exists():
        return "missing"
    normalized = Path(relative)
    parts = normalized.parts
    if not parts:
        return "empty target"
    if parts[0] in PROTECTED_ROOTS and "__pycache__" not in parts:
        return "protected source/configuration path"
    return None


def delete_path(path: Path) -> None:
    """Delete one already-validated generated artifact path."""

    if path.is_dir():
        shutil.rmtree(path, onerror=reset_permissions)
    else:
        path.unlink(missing_ok=True)


def reset_permissions(function, path, excinfo) -> None:
    target = Path(path)
    try:
        target.chmod(stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        function(path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
