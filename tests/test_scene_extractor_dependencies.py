from __future__ import annotations

import unittest
from pathlib import Path

from extractor.config import ExtractorConfig
from extractor.parser import ManimAstParser
from extractor.scene_extractor import SceneExtractor


class SceneExtractorDependencyTests(unittest.TestCase):
    def test_local_dependencies_are_included_before_scene_class(self) -> None:
        source_path = Path("sample_scene.py")
        source = """
from manim import *

SCALE = 2

def make_label():
    return Text("hello")

class HelperScene(Scene):
    def construct(self):
        pass

class MainScene(HelperScene):
    def construct(self):
        label = make_label().scale(SCALE)
        self.add(label)
"""
        parser = ManimAstParser(ExtractorConfig(input_paths=[source_path]))
        parsed = parser.parse_file_from_source(source_path, source) if hasattr(parser, "parse_file_from_source") else None
        if parsed is None:
            import ast
            from extractor.parser import ParsedFile

            tree = ast.parse(source)
            imports = parser._extract_imports(tree)
            classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
            parsed = ParsedFile(
                path=source_path,
                source=source,
                tree=tree,
                imports=imports,
                classes=classes,
                scene_classes=[info for node in classes if (info := parser._scene_info(node))],
            )

        scene = next(item for item in SceneExtractor().extract(parsed) if item.scene_name == "MainScene")

        self.assertIn("def make_label", scene.assistant_code)
        self.assertIn("SCALE = 2", scene.assistant_code)
        self.assertLess(scene.assistant_code.index("def make_label"), scene.assistant_code.index("class MainScene"))
        self.assertIn("make_label", scene.dependency_tree)


if __name__ == "__main__":
    unittest.main()
