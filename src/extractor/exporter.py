"""Dataset splitting, deduplication, exporting, and statistics."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ExtractorConfig, ExtractorStats
from .formatter import DatasetFormatter, TrainingExample
from .utils import write_json

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SplitIntegrityReport:
    """Diagnostics proving content hashes are isolated to a single split."""

    leakage_count: int
    duplicate_hashes_across_splits: dict[str, list[str]]
    split_statistics: dict[str, dict[str, int]]
    split_by_content_hash: bool


class DatasetExporter:
    """Persist training examples as split JSONL files and stats JSON."""

    def __init__(self, config: ExtractorConfig, formatter: DatasetFormatter) -> None:
        self.config = config
        self.formatter = formatter

    def deduplicate(self, examples: list[TrainingExample]) -> tuple[list[TrainingExample], int]:
        """Remove exact duplicate training rows while preserving prompt variants."""

        seen: set[tuple[str, str]] = set()
        unique: list[TrainingExample] = []
        removed = 0
        for example in examples:
            content_hash = str(example.metadata.get("content_hash", ""))
            prompt = next((message.content for message in example.messages if message.role == "user"), "")
            key = (content_hash, prompt)
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            unique.append(example)
        return unique, removed

    def split(self, examples: list[TrainingExample]) -> dict[str, list[TrainingExample]]:
        """Shuffle and split examples into train, validation, and test."""

        if self.config.split_by_content_hash:
            return self._split_by_content_hash(examples)
        LOGGER.warning(
            "Using legacy row-wise splitting; prompt augmentations may leak content across splits."
        )
        shuffled = list(examples)
        random.Random(self.config.random_seed).shuffle(shuffled)

        total = len(shuffled)
        test_count = int(total * self.config.test_ratio)
        validation_count = int(total * self.config.validation_ratio)
        test = shuffled[:test_count]
        validation = shuffled[test_count : test_count + validation_count]
        train = shuffled[test_count + validation_count :]
        train_name, validation_name, test_name = self.config.split_names
        return {train_name: train, validation_name: validation, test_name: test}

    def _split_by_content_hash(self, examples: list[TrainingExample]) -> dict[str, list[TrainingExample]]:
        """Split groups of examples so prompt augmentations stay together.

        The stable content hash identifies the underlying extracted scene. Grouping
        before shuffling prevents augmented prompts for the same assistant code from
        landing in validation or test while the base example is in train.
        """

        grouped: dict[str, list[TrainingExample]] = {}
        for index, example in enumerate(examples):
            content_hash = str(example.metadata.get("content_hash") or "")
            if not content_hash:
                content_hash = f"missing-content-hash-{index}"
                LOGGER.warning(
                    "Training example at index %d has no content_hash; isolating it as %s.",
                    index,
                    content_hash,
                )
            grouped.setdefault(content_hash, []).append(example)

        groups = sorted(grouped.items(), key=lambda item: item[0])
        random.Random(self.config.random_seed).shuffle(groups)

        total_groups = len(groups)
        test_count = int(total_groups * self.config.test_ratio)
        validation_count = int(total_groups * self.config.validation_ratio)

        train_name, validation_name, test_name = self.config.split_names
        selected = {
            test_name: groups[:test_count],
            validation_name: groups[test_count : test_count + validation_count],
            train_name: groups[test_count + validation_count :],
        }
        splits = {
            split_name: [example for _, group_examples in split_groups for example in group_examples]
            for split_name, split_groups in selected.items()
        }
        LOGGER.info(
            "Split %d content-hash groups into %s=%d, %s=%d, %s=%d groups.",
            total_groups,
            train_name,
            len(selected[train_name]),
            validation_name,
            len(selected[validation_name]),
            test_name,
            len(selected[test_name]),
        )
        return splits

    def verify_split_integrity(
        self,
        splits: dict[str, list[TrainingExample]],
    ) -> SplitIntegrityReport:
        """Return a leakage report for content hashes across dataset splits."""

        hash_to_splits: dict[str, set[str]] = {}
        split_statistics: dict[str, dict[str, int]] = {}
        for split_name, split_examples in splits.items():
            hashes = {
                str(example.metadata.get("content_hash") or "")
                for example in split_examples
                if example.metadata.get("content_hash")
            }
            split_statistics[split_name] = {
                "examples": len(split_examples),
                "unique_content_hashes": len(hashes),
            }
            for content_hash in hashes:
                hash_to_splits.setdefault(content_hash, set()).add(split_name)

        duplicate_hashes = {
            content_hash: sorted(split_names)
            for content_hash, split_names in sorted(hash_to_splits.items())
            if len(split_names) > 1
        }
        report = SplitIntegrityReport(
            leakage_count=len(duplicate_hashes),
            duplicate_hashes_across_splits=duplicate_hashes,
            split_statistics=split_statistics,
            split_by_content_hash=self.config.split_by_content_hash,
        )
        if report.leakage_count:
            LOGGER.error("Detected %d content hashes present in multiple splits.", report.leakage_count)
        else:
            LOGGER.info("Split integrity verified: no content_hash overlap across splits.")
        return report

    def export(self, examples: list[TrainingExample], stats: ExtractorStats) -> dict[str, Path]:
        """Write split files plus stats and return generated paths."""

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        splits = self.split(examples)
        integrity_report = self.verify_split_integrity(splits)
        if self.config.enforce_split_integrity and integrity_report.leakage_count:
            raise ValueError(
                "Split integrity check failed: "
                f"{integrity_report.leakage_count} content hashes appear in multiple splits."
            )
        paths: dict[str, Path] = {}
        for split_name, split_examples in splits.items():
            path = self.config.output_dir / f"{split_name}.jsonl"
            self._write_jsonl(path, split_examples)
            paths[split_name] = path

        stats.exported_examples = len(examples)
        write_json(self.config.output_dir / "stats.json", asdict(stats))
        paths["stats"] = self.config.output_dir / "stats.json"
        write_json(self.config.output_dir / "split_integrity_report.json", asdict(integrity_report))
        paths["split_integrity"] = self.config.output_dir / "split_integrity_report.json"
        return paths

    def _write_jsonl(self, path: Path, examples: list[TrainingExample]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(self.formatter.to_dict(example), ensure_ascii=False) + "\n")
