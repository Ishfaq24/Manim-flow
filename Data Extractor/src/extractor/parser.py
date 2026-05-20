"""AST parser for safely identifying Manim scene classes."""

from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ExtractorConfig
from .utils import read_text


@dataclass(slots=True)
class ImportInfo:
    """A normalized import discovered in a Python module."""

    module: str | None
    names: list[str]
    alias: str | None = None


@dataclass(slots=True)
class SceneClassInfo:
    """AST-level description of a class that appears to be a Manim Scene."""

    name: str
    node: ast.ClassDef
    bases: list[str]
    has_construct: bool
    lineno: int
    end_lineno: int


@dataclass(slots=True)
class ParsedFile:
    """Parsed source file with AST and Manim-specific class candidates."""

    path: Path
    source: str
    tree: ast.Module | None
    imports: list[ImportInfo] = field(default_factory=list)
    classes: list[ast.ClassDef] = field(default_factory=list)
    scene_classes: list[SceneClassInfo] = field(default_factory=list)
    error: str | None = None


class ManimAstParser:
    """Parse source files using Python AST without executing repository code."""

    def __init__(self, config: ExtractorConfig) -> None:
        self.config = config

    def parse_file(self, path: Path) -> ParsedFile:
        """Safely parse a Python file and collect scene classes."""

        source = read_text(path)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source, filename=str(path), type_comments=True)
        except SyntaxError as exc:
            return ParsedFile(path=path, source=source, tree=None, error=str(exc))

        imports = self._extract_imports(tree)
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        scene_classes = [info for node in classes if (info := self._scene_info(node))]
        return ParsedFile(
            path=path,
            source=source,
            tree=tree,
            imports=imports,
            classes=classes,
            scene_classes=scene_classes,
        )

    def _extract_imports(self, tree: ast.Module) -> list[ImportInfo]:
        imports: list[ImportInfo] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(module=alias.name, names=[alias.name], alias=alias.asname))
            elif isinstance(node, ast.ImportFrom):
                imports.append(
                    ImportInfo(
                        module=node.module,
                        names=[alias.name for alias in node.names],
                        alias=None,
                    )
                )
        return imports

    def _scene_info(self, node: ast.ClassDef) -> SceneClassInfo | None:
        bases = [self._base_name(base) for base in node.bases]
        is_scene = any(base in self.config.scene_base_names or base.endswith("Scene") for base in bases)
        if not is_scene:
            return None
        has_construct = any(
            isinstance(item, ast.FunctionDef) and item.name == "construct" for item in node.body
        )
        return SceneClassInfo(
            name=node.name,
            node=node,
            bases=bases,
            has_construct=has_construct,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
        )

    def _base_name(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._base_name(node.value)}.{node.attr}"
        if isinstance(node, ast.Subscript):
            return self._base_name(node.value)
        if isinstance(node, ast.Call):
            return self._base_name(node.func)
        return ast.dump(node)


def literal_node_name(node: ast.AST) -> str | None:
    """Best-effort readable name for AST call targets and attributes."""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = literal_node_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return literal_node_name(node.func)
    if isinstance(node, ast.Subscript):
        return literal_node_name(node.value)
    return None


def ast_to_source_segment(source: str, node: ast.AST) -> str:
    """Return source segment for a node, falling back to ast.unparse when needed."""

    segment = ast.get_source_segment(source, node)
    if segment:
        return segment
    try:
        return ast.unparse(node)
    except Exception:
        return ""
