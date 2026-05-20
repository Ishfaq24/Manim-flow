from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from extractor.config import ExtractorConfig
from extractor.scanner import RepositoryScanner


def load_cleanup_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "cleanup_workspace.py"
    spec = importlib.util.spec_from_file_location("cleanup_workspace", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScannerAndCleanupSafetyTests(unittest.TestCase):
    def test_scanner_prunes_ignored_directories_before_descent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "keep").mkdir()
            (root / "keep" / "scene.py").write_text("class A: pass\n", encoding="utf-8")
            ignored = root / "node_modules" / "deep"
            ignored.mkdir(parents=True)
            (ignored / "ignored.py").write_text("class B: pass\n", encoding="utf-8")

            scanner = RepositoryScanner(ExtractorConfig(input_paths=[root]))
            files = scanner.scan()

            self.assertEqual([path.name for path in files], ["scene.py"])
            self.assertGreaterEqual(scanner.metrics.skipped_directories, 1)

    def test_cleanup_protects_source_paths(self) -> None:
        cleanup = load_cleanup_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source_path = root / "src" / "extractor" / "core"
            source_path.mkdir(parents=True)

            reason = cleanup.safety_check(root, source_path, "src/extractor/core")

            self.assertEqual(reason, "protected source/configuration path")


if __name__ == "__main__":
    unittest.main()
