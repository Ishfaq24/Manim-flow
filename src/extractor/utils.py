"""Shared utilities for filesystem IO, hashing, logging, and batching."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


try:
    from tqdm import tqdm as progress
except ImportError:  # pragma: no cover - exercised only in minimal runtimes.
    def progress(iterable: Iterable[T], *args: Any, **kwargs: Any) -> Iterable[T]:
        """Fallback progress wrapper used when tqdm is not installed."""

        return iterable


def configure_logging(level: str = "INFO") -> None:
    """Configure consistent, compact logging for CLI and library usage."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def read_text(path: Path) -> str:
    """Read a text file while tolerating common repository encodings."""

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="replace")


def stable_hash(text: str) -> str:
    """Return a stable SHA-256 hash for deduplication."""

    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def batched(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield fixed-size batches from a sequence."""

    for index in range(0, len(items), size):
        yield items[index : index + size]


def json_default(value: Any) -> Any:
    """JSON serializer for dataclasses and paths."""

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    """Write pretty JSON with parent directories created."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    """Return unique strings while preserving first-seen order."""

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
