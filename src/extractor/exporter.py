"""Dataset splitting, deduplication, exporting, and statistics."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from .config import ExtractorConfig, ExtractorStats
from .formatter import DatasetFormatter, TrainingExample
from .utils import write_json


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

    def export(self, examples: list[TrainingExample], stats: ExtractorStats) -> dict[str, Path]:
        """Write split files plus stats and return generated paths."""

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        splits = self.split(examples)
        paths: dict[str, Path] = {}
        for split_name, split_examples in splits.items():
            path = self.config.output_dir / f"{split_name}.jsonl"
            self._write_jsonl(path, split_examples)
            paths[split_name] = path

        stats.exported_examples = len(examples)
        write_json(self.config.output_dir / "stats.json", asdict(stats))
        paths["stats"] = self.config.output_dir / "stats.json"
        return paths

    def _write_jsonl(self, path: Path, examples: list[TrainingExample]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(self.formatter.to_dict(example), ensure_ascii=False) + "\n")
