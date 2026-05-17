"""Repository scanner for Manim Python source files."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

from .config import ExtractorConfig
from .utils import progress

LOGGER = logging.getLogger(__name__)


class RepositoryScanner:
    """Recursively discover candidate Python files while skipping noisy trees."""

    def __init__(self, config: ExtractorConfig) -> None:
        self.config = config

    def scan(self) -> list[Path]:
        """Return sorted Python files from every configured input path."""

        discovered: set[Path] = set()
        roots = self.config.normalized_input_paths()
        for root in roots:
            if root.is_file():
                if self._is_candidate_file(root):
                    discovered.add(root)
                continue
            if not root.exists():
                LOGGER.warning("Input path does not exist: %s", root)
                continue
            for path in self._walk(root):
                discovered.add(path)
        return sorted(discovered)

    def batches(self, paths: Sequence[Path] | None = None) -> Iterator[list[Path]]:
        """Yield candidate files in configured batch sizes."""

        selected = list(paths or self.scan())
        for index in range(0, len(selected), self.config.batch_size):
            yield selected[index : index + self.config.batch_size]

    def _walk(self, root: Path) -> Iterator[Path]:
        paths = list(root.rglob("*.py"))
        for path in progress(paths, desc=f"Scanning {root.name}", unit="file"):
            if self._is_candidate_file(path):
                yield path

    def _is_candidate_file(self, path: Path) -> bool:
        if path.suffix != ".py":
            return False
        parts = set(path.parts)
        if parts.intersection(self.config.ignore_dirs):
            return False
        if path.name.startswith("."):
            return False
        return True
