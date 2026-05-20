from __future__ import annotations

import unittest
from pathlib import Path

from extractor.config import ExtractorConfig
from extractor.exporter import DatasetExporter
from extractor.formatter import ChatMessage, DatasetFormatter, TrainingExample


def example(content_hash: str, prompt: str) -> TrainingExample:
    return TrainingExample(
        messages=[
            ChatMessage(role="user", content=prompt),
            ChatMessage(role="assistant", content=f"# code for {content_hash}\n"),
        ],
        metadata={"content_hash": content_hash},
    )


class DatasetExporterSplitTests(unittest.TestCase):
    def test_content_hash_groups_do_not_cross_splits(self) -> None:
        config = ExtractorConfig(
            input_paths=[Path(".")],
            output_dir=Path("unused"),
            validation_ratio=0.25,
            test_ratio=0.25,
            random_seed=7,
            split_by_content_hash=True,
        )
        exporter = DatasetExporter(config, DatasetFormatter())
        examples = [
            example(f"hash-{index}", f"prompt-{variant}")
            for index in range(12)
            for variant in range(2)
        ]

        splits = exporter.split(examples)
        report = exporter.verify_split_integrity(splits)

        self.assertEqual(report.leakage_count, 0)
        self.assertEqual(sum(len(rows) for rows in splits.values()), len(examples))
        for split_examples in splits.values():
            counts: dict[str, int] = {}
            for row in split_examples:
                content_hash = str(row.metadata["content_hash"])
                counts[content_hash] = counts.get(content_hash, 0) + 1
            self.assertTrue(all(count == 2 for count in counts.values()))


if __name__ == "__main__":
    unittest.main()
