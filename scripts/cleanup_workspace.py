"""Remove generated local artifacts after smoke tests.

The script refuses to delete anything outside the current project root.
"""

from __future__ import annotations

import shutil
import stat
from pathlib import Path


TARGETS = [
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
    "src/extractor/config",
    "src/extractor/core",
    "src/extractor/dataset",
    "src/extractor/integrations",
    "src/extractor/metadata",
    "src/extractor/models",
    "src/extractor/prompts",
    "src/extractor/render",
    "src/extractor/utils",
    "src/extractor/workers",
    "src/extractor/__pycache__",
    "scripts/__pycache__",
]


def main() -> None:
    root = Path.cwd().resolve()
    removed: list[str] = []
    skipped: list[str] = []

    for relative in TARGETS:
        path = (root / relative).resolve()
        if not str(path).startswith(str(root)):
            raise RuntimeError(f"Refusing to delete outside workspace: {path}")
        if not path.exists():
            skipped.append(relative)
            continue
        if path.is_dir():
            shutil.rmtree(path, onexc=reset_permissions)
        else:
            path.unlink(missing_ok=True)
        if path.exists():
            skipped.append(f"{relative} (permission locked)")
        else:
            removed.append(relative)

    print("Removed:")
    for item in removed:
        print(f"  {item}")
    print("Skipped:")
    for item in skipped:
        print(f"  {item}")


def reset_permissions(function, path, excinfo) -> None:
    target = Path(path)
    try:
        target.chmod(stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        function(path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
