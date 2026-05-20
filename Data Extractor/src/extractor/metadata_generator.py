"""Metadata inference for extracted Manim scenes."""

from __future__ import annotations

from dataclasses import dataclass

from .scene_extractor import ExtractedScene
from .utils import unique_preserve_order


@dataclass(slots=True)
class SceneMetadata:
    """Training metadata attached to each extracted scene."""

    scene_name: str
    category: str
    objects: list[str]
    animations: list[str]
    difficulty: str
    source_repo: str | None
    source_path: str
    line_start: int
    line_end: int
    quality_score: float
    tags: list[str]
    complexity: dict[str, int | bool]


class MetadataGenerator:
    """Infer Manim objects, category, difficulty, tags, and quality score."""

    def __init__(self, source_repo: str | None = None) -> None:
        self.source_repo = source_repo

    def generate(self, scene: ExtractedScene) -> SceneMetadata:
        objects = self._objects(scene.method_calls)
        animations = unique_preserve_order(scene.animations)
        tags = self._tags(scene, objects, animations)
        category = self._category(scene, objects, animations, tags)
        complexity = self._complexity(scene, objects, animations)
        difficulty = self._difficulty(complexity)
        quality_score = self._quality_score(scene, objects, animations, complexity)

        return SceneMetadata(
            scene_name=scene.scene_name,
            category=category,
            objects=objects,
            animations=animations,
            difficulty=difficulty,
            source_repo=self.source_repo,
            source_path=str(scene.source_path),
            line_start=scene.line_start,
            line_end=scene.line_end,
            quality_score=quality_score,
            tags=tags,
            complexity=complexity,
        )

    def _objects(self, calls: list[str]) -> list[str]:
        return unique_preserve_order(call for call in calls if call in KNOWN_OBJECTS)

    def _tags(self, scene: ExtractedScene, objects: list[str], animations: list[str]) -> list[str]:
        text = scene.assistant_code.lower()
        tags: list[str] = []
        if {"Axes", "NumberPlane", "Graph", "FunctionGraph"}.intersection(objects) or "plot(" in text:
            tags.append("graphs")
        if {"MathTex", "Tex"}.intersection(objects):
            tags.append("math")
        if {"Circle", "Square", "Triangle", "Polygon", "Line", "Angle"}.intersection(objects):
            tags.append("geometry")
        if {"Vector", "Arrow", "Matrix"}.intersection(objects):
            tags.append("linear-algebra")
        if any(word in text for word in ("velocity", "force", "mass", "field", "wave", "pendulum")):
            tags.append("physics")
        if "ThreeDScene" in scene.bases or {"ThreeDAxes", "Surface", "Sphere"}.intersection(objects):
            tags.append("3d")
        if animations:
            tags.append("animation")
        if any(obj in objects for obj in ("Text", "MarkupText", "Paragraph")):
            tags.append("text")
        return unique_preserve_order(tags)

    def _category(
        self,
        scene: ExtractedScene,
        objects: list[str],
        animations: list[str],
        tags: list[str],
    ) -> str:
        if "graphs" in tags:
            return "graphs"
        if "3d" in tags:
            return "3d"
        if "physics" in tags:
            return "physics"
        if "geometry" in tags:
            return "geometry"
        if "math" in tags:
            return "math"
        if "text" in tags:
            return "text"
        if animations:
            return "animations"
        return "general"

    def _complexity(
        self,
        scene: ExtractedScene,
        objects: list[str],
        animations: list[str],
    ) -> dict[str, int | bool]:
        line_count = max(0, scene.line_end - scene.line_start + 1)
        return {
            "line_count": line_count,
            "object_count": len(objects),
            "animation_count": len(animations),
            "call_count": len(scene.method_calls),
            "has_construct": scene.has_construct,
        }

    def _difficulty(self, complexity: dict[str, int | bool]) -> str:
        score = (
            int(complexity["line_count"]) * 0.04
            + int(complexity["object_count"]) * 0.8
            + int(complexity["animation_count"]) * 1.0
            + int(complexity["call_count"]) * 0.12
        )
        if score < 6:
            return "beginner"
        if score < 16:
            return "intermediate"
        return "advanced"

    def _quality_score(
        self,
        scene: ExtractedScene,
        objects: list[str],
        animations: list[str],
        complexity: dict[str, int | bool],
    ) -> float:
        score = 0.25
        if scene.has_construct:
            score += 0.25
        if objects:
            score += 0.15
        if animations:
            score += 0.15
        if int(complexity["line_count"]) >= 5:
            score += 0.1
        if "from manim import" in scene.assistant_code or "import manim" in scene.assistant_code:
            score += 0.1
        return round(min(score, 1.0), 3)


KNOWN_OBJECTS: frozenset[str] = frozenset(
    {
        "Angle",
        "Annulus",
        "Arrow",
        "Axes",
        "BarChart",
        "Brace",
        "Circle",
        "ComplexPlane",
        "DecimalNumber",
        "Dot",
        "Ellipse",
        "FunctionGraph",
        "Graph",
        "ImageMobject",
        "Integer",
        "Line",
        "MathTex",
        "Matrix",
        "NumberLine",
        "NumberPlane",
        "ParametricFunction",
        "Paragraph",
        "Polygon",
        "Rectangle",
        "RegularPolygon",
        "Sector",
        "Sphere",
        "Square",
        "Surface",
        "SurroundingRectangle",
        "Table",
        "Tex",
        "Text",
        "ThreeDAxes",
        "Triangle",
        "ValueTracker",
        "Vector",
        "VGroup",
    }
)
