from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_model_ready_dataset import build_model_ready


class ModelReadyDatasetTests(unittest.TestCase):
    def test_filters_rows_with_missing_symbols_and_preserves_split_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "processed" / "source-a"
            source_dir.mkdir(parents=True)
            good = (
                '{"messages":[{"role":"user","content":"p"},{"role":"assistant","content":"c"}],'
                '"metadata":{"content_hash":"hash-a","extraction_diagnostics":{"missing_symbols":[]}}}'
            )
            bad = (
                '{"messages":[{"role":"user","content":"p"},{"role":"assistant","content":"c"}],'
                '"metadata":{"content_hash":"hash-b","extraction_diagnostics":{"missing_symbols":["X"]}}}'
            )
            (source_dir / "train.jsonl").write_text(good + "\n" + bad + "\n", encoding="utf-8")
            (source_dir / "validation.jsonl").write_text("", encoding="utf-8")
            (source_dir / "test.jsonl").write_text("", encoding="utf-8")

            result = build_model_ready(
                processed_dir=root / "processed",
                output_dir=root / "model-ready",
                sources=["source-a"],
                train_small_rows=8,
            )

            self.assertEqual(result["stats"]["exportedRows"], 1)
            self.assertEqual(result["stats"]["excluded"]["missing_symbols"], 1)
            self.assertEqual(result["split_integrity_report"]["leakage_count"], 0)


if __name__ == "__main__":
    unittest.main()
