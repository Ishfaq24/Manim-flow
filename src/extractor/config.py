"""Configuration objects for the Manim dataset extractor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".env",
        "node_modules",
        "site-packages",
        "dist",
        "build",
        "media",
        "partial_movie_files",
    }
)

SCENE_BASE_NAMES: frozenset[str] = frozenset(
    {
        "Scene",
        "ThreeDScene",
        "MovingCameraScene",
        "GraphScene",
        "LinearTransformationScene",
        "VectorScene",
        "ZoomedScene",
        "SpecialThreeDScene",
        "InteractiveScene",
        "TeacherStudentsScene",
        "PiCreatureScene",
    }
)


@dataclass(slots=True)
class ExtractorConfig:
    """Runtime configuration for extraction and export."""

    input_paths: Sequence[Path]
    output_dir: Path = Path("dataset")
    source_repo: str | None = None
    max_workers: int = 4
    batch_size: int = 256
    validation_ratio: float = 0.05
    test_ratio: float = 0.05
    random_seed: int = 42
    min_quality_score: float = 0.35
    run_render_validation: bool = False
    render_timeout_seconds: int = 90
    manim_binary: str = "manim"
    preview_output_dir: Path | None = None
    split_names: tuple[str, str, str] = ("train", "validation", "test")
    ignore_dirs: frozenset[str] = DEFAULT_IGNORE_DIRS
    scene_base_names: frozenset[str] = SCENE_BASE_NAMES
    prompt_augmentation_count: int = 0
    include_file_metadata: bool = True
    log_level: str = "INFO"

    def normalized_input_paths(self) -> list[Path]:
        """Return absolute input paths with user home markers expanded."""

        return [path.expanduser().resolve() for path in self.input_paths]

    def validate(self) -> None:
        """Validate config values early so pipeline failures are explicit."""

        if not self.input_paths:
            raise ValueError("At least one input path is required.")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if not 0 <= self.validation_ratio < 1:
            raise ValueError("validation_ratio must be in [0, 1).")
        if not 0 <= self.test_ratio < 1:
            raise ValueError("test_ratio must be in [0, 1).")
        if self.validation_ratio + self.test_ratio >= 1:
            raise ValueError("validation_ratio + test_ratio must be < 1.")
        if self.prompt_augmentation_count < 0:
            raise ValueError("prompt_augmentation_count must be >= 0.")


@dataclass(slots=True)
class ExtractorStats:
    """Aggregate dataset statistics produced during export."""

    scanned_files: int = 0
    parsed_files: int = 0
    failed_files: int = 0
    discovered_scenes: int = 0
    valid_scenes: int = 0
    exported_examples: int = 0
    duplicates_removed: int = 0
    categories: dict[str, int] = field(default_factory=dict)
    difficulties: dict[str, int] = field(default_factory=dict)

    def bump_category(self, category: str) -> None:
        self.categories[category] = self.categories.get(category, 0) + 1

    def bump_difficulty(self, difficulty: str) -> None:
        self.difficulties[difficulty] = self.difficulties.get(difficulty, 0) + 1
