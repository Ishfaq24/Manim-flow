"""Build a filtered model-ready dataset from regenerated processed sources.

This script is intentionally non-destructive: it reads existing processed
split files, filters rows with known extraction problems, and writes a separate
output directory. Source-specific processed datasets remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("build_model_ready_dataset")
SPLITS = ("train", "validation", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows, skipping malformed records with a warning."""

    rows: list[dict[str, Any]] = []
    if not path.exists():
        LOGGER.warning("Missing input split: %s", path)
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping malformed JSON in %s:%d: %s", path, line_no, exc)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows with parent directories created."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_missing_symbols(row: dict[str, Any]) -> list[str]:
    diagnostics = row.get("metadata", {}).get("extraction_diagnostics", {})
    missing = diagnostics.get("missing_symbols", [])
    return [str(item) for item in missing]


def build_model_ready(
    processed_dir: Path,
    output_dir: Path,
    sources: list[str],
    train_small_rows: int,
) -> dict[str, Any]:
    """Combine clean rows from selected processed sources into one dataset."""

    stats: dict[str, Any] = {
        "sources": sources,
        "splits": {},
        "excluded": {"missing_symbols": 0, "missing_hash": 0},
        "totalRows": 0,
        "exportedRows": 0,
    }
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}

    for split in SPLITS:
        for source in sources:
            source_path = processed_dir / source / f"{split}.jsonl"
            for row in read_jsonl(source_path):
                stats["totalRows"] += 1
                metadata = row.setdefault("metadata", {})
                if not metadata.get("content_hash"):
                    stats["excluded"]["missing_hash"] += 1
                    continue
                if row_missing_symbols(row):
                    stats["excluded"]["missing_symbols"] += 1
                    continue
                metadata["model_ready_source"] = source
                split_rows[split].append(row)

        write_jsonl(output_dir / f"{split}.jsonl", split_rows[split])
        stats["splits"][split] = len(split_rows[split])
        stats["exportedRows"] += len(split_rows[split])

    train_small = split_rows["train"][:train_small_rows]
    write_jsonl(output_dir / "train_small.jsonl", train_small)
    stats["trainSmallRows"] = len(train_small)

    report = split_integrity_report(split_rows)
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (output_dir / "split_integrity_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    if report["leakage_count"]:
        raise ValueError(f"Model-ready split leakage detected: {report['leakage_count']}")
    return {"stats": stats, "split_integrity_report": report}


def split_integrity_report(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Report whether any content hash appears in more than one split."""

    hash_to_splits: dict[str, set[str]] = {}
    split_statistics: dict[str, dict[str, int]] = {}
    for split, rows in split_rows.items():
        hashes = {
            str(row.get("metadata", {}).get("content_hash"))
            for row in rows
            if row.get("metadata", {}).get("content_hash")
        }
        split_statistics[split] = {
            "examples": len(rows),
            "unique_content_hashes": len(hashes),
        }
        for content_hash in hashes:
            hash_to_splits.setdefault(content_hash, set()).add(split)

    duplicates = {
        content_hash: sorted(split_names)
        for content_hash, split_names in sorted(hash_to_splits.items())
        if len(split_names) > 1
    }
    return {
        "leakage_count": len(duplicates),
        "duplicate_hashes_across_splits": duplicates,
        "split_statistics": split_statistics,
        "split_by_content_hash": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a filtered model-ready Manim dataset.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/model-ready"))
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["3b1b-videos", "AnimationsWithManim", "manim"],
        help="Processed source directories to combine.",
    )
    parser.add_argument("--train-small-rows", type=int, default=512)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    result = build_model_ready(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        sources=args.sources,
        train_small_rows=args.train_small_rows,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
