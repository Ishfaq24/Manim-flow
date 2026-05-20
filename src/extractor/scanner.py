"""Repository scanner for Manim Python source files."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import ExtractorConfig
from .utils import progress

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ScanMetrics:
    """Lightweight scanner performance and pruning counters."""

    candidate_files: int = 0
    skipped_directories: int = 0
    duration_seconds: float = 0.0


class RepositoryScanner:
    """Recursively discover candidate Python files while skipping noisy trees."""

    def __init__(self, config: ExtractorConfig) -> None:
        self.config = config
        self.metrics = ScanMetrics()

    def scan(self) -> list[Path]:
        """Return sorted Python files from every configured input path."""

        started = time.perf_counter()
        self.metrics = ScanMetrics()
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
        self.metrics.candidate_files = len(discovered)
        self.metrics.duration_seconds = round(time.perf_counter() - started, 3)
        LOGGER.info(
            "Scan complete: %d candidate files, %d skipped directories, %.3fs.",
            self.metrics.candidate_files,
            self.metrics.skipped_directories,
            self.metrics.duration_seconds,
        )
        return sorted(discovered)

    def batches(self, paths: Sequence[Path] | None = None) -> Iterator[list[Path]]:
        """Yield candidate files in configured batch sizes."""

        selected = list(paths or self.scan())
        for index in range(0, len(selected), self.config.batch_size):
            yield selected[index : index + self.config.batch_size]

    def _walk(self, root: Path) -> Iterator[Path]:
        for path in progress(self._walk_pruned(root), desc=f"Scanning {root.name}", unit="file"):
            yield path

    def _walk_pruned(self, root: Path) -> Iterator[Path]:
        """Yield Python files while pruning ignored directories before descent."""

        stack = [root]
        while stack:
            directory = stack.pop()
            try:
                entries = list(directory.iterdir())
            except OSError as exc:
                LOGGER.debug("Skipping unreadable directory %s: %s", directory, exc)
                self.metrics.skipped_directories += 1
                continue
            for entry in entries:
                if entry.is_dir():
                    if self._is_ignored_dir(entry):
                        self.metrics.skipped_directories += 1
                        LOGGER.debug("Pruned ignored directory: %s", entry)
                        continue
                    stack.append(entry)
                elif self._is_candidate_file(entry):
                    yield entry

    def _is_ignored_dir(self, path: Path) -> bool:
        return path.name in self.config.ignore_dirs

    def _is_candidate_file(self, path: Path) -> bool:
        if path.suffix != ".py":
            return False
        parts = set(path.parts)
        if parts.intersection(self.config.ignore_dirs):
            return False
        if path.name.startswith("."):
            return False
        return True
