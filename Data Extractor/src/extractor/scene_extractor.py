"""Scene-level extraction from parsed Manim modules."""

from __future__ import annotations

import ast
import builtins
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .parser import ParsedFile, SceneClassInfo, ast_to_source_segment, literal_node_name
from .utils import stable_hash, unique_preserve_order

LOGGER = logging.getLogger(__name__)


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
    dependency_code: str = ""
    objects: list[str] = field(default_factory=list)
    animations: list[str] = field(default_factory=list)
    method_calls: list[str] = field(default_factory=list)
    missing_symbols: list[str] = field(default_factory=list)
    dependency_tree: dict[str, list[str]] = field(default_factory=dict)
    has_construct: bool = False
    line_start: int = 0
    line_end: int = 0
    content_hash: str = ""


class SceneExtractor:
    """Extract class code, import context, construct body, and call inventory."""

    def __init__(self, include_module_dependencies: bool = True) -> None:
        self.include_module_dependencies = include_module_dependencies

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
        dependency_nodes, dependency_tree, missing_symbols = (
            self._resolve_dependencies(parsed, scene_info)
            if self.include_module_dependencies
            else ([], {}, [])
        )
        dependency_code = self._dependency_source(parsed.source, dependency_nodes)
        construct_node = self._construct_node(scene_info.node)
        construct_code = ast_to_source_segment(parsed.source, construct_node) if construct_node else None
        assistant_code = self._build_assistant_code(imports_code, dependency_code, class_code)
        calls = self._call_names(scene_info.node)
        animation_blocks = self._animation_names(scene_info.node)
        if missing_symbols:
            LOGGER.debug(
                "Scene %s has unresolved local symbols: %s",
                scene_info.name,
                ", ".join(missing_symbols),
            )

        return ExtractedScene(
            scene_name=scene_info.name,
            source_path=parsed.path,
            class_code=class_code,
            assistant_code=assistant_code,
            construct_code=construct_code,
            imports_code=imports_code,
            dependency_code=dependency_code,
            bases=scene_info.bases,
            objects=[],
            animations=animation_blocks,
            method_calls=calls,
            missing_symbols=missing_symbols,
            dependency_tree=dependency_tree,
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

    def _build_assistant_code(self, imports_code: str, dependency_code: str, class_code: str) -> str:
        parts = [
            part.strip()
            for part in (imports_code, dependency_code, class_code)
            if part and part.strip()
        ]
        return "\n\n".join(parts).strip() + "\n"

    def _resolve_dependencies(
        self,
        parsed: ParsedFile,
        scene_info: SceneClassInfo,
    ) -> tuple[list[ast.AST], dict[str, list[str]], list[str]]:
        """Find module-level symbols needed by a scene class.

        This is intentionally conservative and source-order preserving. It only
        pulls in top-level functions, classes, and assignments that the scene or
        another included dependency references by name. It does not execute code
        or chase imports, which keeps extraction safe for untrusted repositories.
        """

        if parsed.tree is None:
            return [], {}, []

        symbols = self._top_level_symbols(parsed.tree, scene_info.name)
        imported_names = self._imported_names(parsed)
        has_wildcard_import = self._has_wildcard_import(parsed)
        local_names = self._defined_names(scene_info.node)
        dependency_names: set[str] = set()
        dependency_tree: dict[str, list[str]] = {}
        missing_candidates: set[str] = set()
        pending = [
            name
            for name in self._loaded_names(scene_info.node)
            if name in symbols and name not in local_names
        ]

        while pending:
            name = pending.pop(0)
            if name in dependency_names:
                continue
            dependency_names.add(name)
            node = symbols[name]
            child_names = [
                child
                for child in self._loaded_names(node)
                if child in symbols and child not in dependency_names
            ]
            dependency_tree[name] = unique_preserve_order(child_names)
            pending.extend(child_names)

        known_names = (
            set(symbols)
            | imported_names
            | local_names
            | set(dir(builtins))
            | {scene_info.name, "self", "cls", "True", "False", "None"}
        )
        for name in self._loaded_names(scene_info.node):
            if has_wildcard_import:
                continue
            if name not in known_names and not name.startswith("_"):
                missing_candidates.add(name)

        nodes = [
            node
            for node in parsed.tree.body
            if any(symbols.get(name) is node for name in dependency_names)
        ]
        return nodes, dependency_tree, sorted(missing_candidates)

    def _top_level_symbols(self, tree: ast.Module, scene_name: str) -> dict[str, ast.AST]:
        symbols: dict[str, ast.AST] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name != scene_name:
                    symbols[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for name in self._assigned_names(node):
                    symbols[name] = node
        return symbols

    def _dependency_source(self, source: str, nodes: list[ast.AST]) -> str:
        chunks = [segment for node in nodes if (segment := ast_to_source_segment(source, node))]
        return "\n\n".join(unique_preserve_order(chunks))

    def _assigned_names(self, node: ast.AST) -> list[str]:
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets.append(node.target)
        elif isinstance(node, ast.AugAssign):
            targets.append(node.target)

        names: list[str] = []
        for target in targets:
            for child in ast.walk(target):
                if isinstance(child, ast.Name):
                    names.append(child.id)
        return unique_preserve_order(names)

    def _loaded_names(self, node: ast.AST) -> list[str]:
        return unique_preserve_order(
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        )

    def _defined_names(self, node: ast.AST) -> set[str]:
        names: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                names.add(child.id)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(child.name)
                for arg in child.args.args + child.args.kwonlyargs:
                    names.add(arg.arg)
                if child.args.vararg:
                    names.add(child.args.vararg.arg)
                if child.args.kwarg:
                    names.add(child.args.kwarg.arg)
            elif isinstance(child, ast.ClassDef):
                names.add(child.name)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                names.add(child.name)
        return names

    def _imported_names(self, parsed: ParsedFile) -> set[str]:
        names: set[str] = set()
        for import_info in parsed.imports:
            for name in import_info.names:
                if name == "*":
                    continue
                names.add(import_info.alias or name.split(".")[0])
        return names

    def _has_wildcard_import(self, parsed: ParsedFile) -> bool:
        return any("*" in import_info.names for import_info in parsed.imports)

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
