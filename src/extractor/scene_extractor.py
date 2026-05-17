"""Scene-level extraction from parsed Manim modules."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from .parser import ParsedFile, SceneClassInfo, ast_to_source_segment, literal_node_name
from .utils import stable_hash, unique_preserve_order


@dataclass(slots=True)
class ExtractedScene:
    """Complete extracted scene payload before formatting for training."""

    scene_name: str
    source_path: Path
    class_code: str
    assistant_code: str
    construct_code: str | None
    imports_code: str
    bases: list[str]
    objects: list[str] = field(default_factory=list)
    animations: list[str] = field(default_factory=list)
    method_calls: list[str] = field(default_factory=list)
    has_construct: bool = False
    line_start: int = 0
    line_end: int = 0
    content_hash: str = ""


class SceneExtractor:
    """Extract class code, import context, construct body, and call inventory."""

    def extract(self, parsed: ParsedFile) -> list[ExtractedScene]:
        """Extract all scene classes from one parsed file."""

        if parsed.tree is None:
            return []
        imports_code = self._module_imports_source(parsed.source, parsed.tree)
        scenes: list[ExtractedScene] = []
        for scene_info in parsed.scene_classes:
            scenes.append(self._extract_scene(parsed, scene_info, imports_code))
        return scenes

    def _extract_scene(
        self,
        parsed: ParsedFile,
        scene_info: SceneClassInfo,
        imports_code: str,
    ) -> ExtractedScene:
        class_code = ast_to_source_segment(parsed.source, scene_info.node)
        construct_node = self._construct_node(scene_info.node)
        construct_code = ast_to_source_segment(parsed.source, construct_node) if construct_node else None
        assistant_code = self._build_assistant_code(imports_code, class_code)
        calls = self._call_names(scene_info.node)
        animation_blocks = self._animation_names(scene_info.node)

        return ExtractedScene(
            scene_name=scene_info.name,
            source_path=parsed.path,
            class_code=class_code,
            assistant_code=assistant_code,
            construct_code=construct_code,
            imports_code=imports_code,
            bases=scene_info.bases,
            objects=[],
            animations=animation_blocks,
            method_calls=calls,
            has_construct=scene_info.has_construct,
            line_start=scene_info.lineno,
            line_end=scene_info.end_lineno,
            content_hash=stable_hash(assistant_code),
        )

    def _module_imports_source(self, source: str, tree: ast.Module) -> str:
        chunks: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                segment = ast_to_source_segment(source, node)
                if segment:
                    chunks.append(segment)
        if not any("manim" in chunk for chunk in chunks):
            chunks.insert(0, "from manim import *")
        return "\n".join(unique_preserve_order(chunks))

    def _build_assistant_code(self, imports_code: str, class_code: str) -> str:
        parts = [part.strip() for part in (imports_code, class_code) if part and part.strip()]
        return "\n\n".join(parts).strip() + "\n"

    def _construct_node(self, node: ast.ClassDef) -> ast.FunctionDef | None:
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "construct":
                return item
        return None

    def _call_names(self, node: ast.ClassDef) -> list[str]:
        names: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = literal_node_name(child.func)
                if name:
                    names.append(name.split(".")[-1])
        return unique_preserve_order(names)

    def _animation_names(self, node: ast.ClassDef) -> list[str]:
        animations: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = literal_node_name(child.func)
                if name and name.split(".")[-1] in KNOWN_ANIMATIONS:
                    animations.append(name.split(".")[-1])
        return unique_preserve_order(animations)


KNOWN_ANIMATIONS: frozenset[str] = frozenset(
    {
        "AddTextLetterByLetter",
        "ApplyMethod",
        "ApplyWave",
        "Circumscribe",
        "Create",
        "DrawBorderThenFill",
        "FadeIn",
        "FadeOut",
        "FadeTransform",
        "Flash",
        "FocusOn",
        "GrowArrow",
        "GrowFromCenter",
        "Indicate",
        "LaggedStart",
        "MoveAlongPath",
        "ReplacementTransform",
        "Restore",
        "Rotate",
        "ShowCreation",
        "ShowIncreasingSubsets",
        "ShowPassingFlash",
        "Transform",
        "TransformFromCopy",
        "Uncreate",
        "Unwrite",
        "UpdateFromAlphaFunc",
        "Wiggle",
        "Write",
    }
)
