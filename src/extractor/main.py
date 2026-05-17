"""Command-line pipeline for building Manim instruction-tuning datasets."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import ExtractorConfig, ExtractorStats
from .exporter import DatasetExporter
from .formatter import DatasetFormatter, TrainingExample
from .metadata_generator import MetadataGenerator
from .parser import ManimAstParser, ParsedFile
from .prompt_generator import PromptGenerator
from .scanner import RepositoryScanner
from .scene_extractor import ExtractedScene, SceneExtractor
from .utils import configure_logging, progress
from .validator import SceneValidator

LOGGER = logging.getLogger(__name__)


class ExtractionPipeline:
    """End-to-end orchestration for scalable Manim dataset extraction."""

    def __init__(self, config: ExtractorConfig) -> None:
        config.validate()
        self.config = config
        self.scanner = RepositoryScanner(config)
        self.parser = ManimAstParser(config)
        self.scene_extractor = SceneExtractor()
        self.metadata_generator = MetadataGenerator(source_repo=config.source_repo)
        self.prompt_generator = PromptGenerator(
            augmentation_count=config.prompt_augmentation_count,
            seed=config.random_seed,
        )
        self.validator = SceneValidator(config)
        self.formatter = DatasetFormatter()
        self.exporter = DatasetExporter(config, self.formatter)
        self.stats = ExtractorStats()

    def run(self) -> dict[str, Path]:
        """Run the pipeline and export split dataset files."""

        files = self.scanner.scan()
        self.stats.scanned_files = len(files)
        LOGGER.info("Discovered %d candidate Python files.", len(files))

        parsed_files = self._parse_files(files)
        scenes = self._extract_scenes(parsed_files)
        examples = self._build_examples(scenes)
        examples, removed = self.exporter.deduplicate(examples)
        self.stats.duplicates_removed = removed

        paths = self.exporter.export(examples, self.stats)
        LOGGER.info("Exported %d examples to %s.", len(examples), self.config.output_dir)
        return paths

    def _parse_files(self, files: list[Path]) -> list[ParsedFile]:
        parsed: list[ParsedFile] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(self.parser.parse_file, path): path for path in files}
            for future in progress(as_completed(futures), total=len(futures), desc="Parsing", unit="file"):
                result = future.result()
                if result.error:
                    self.stats.failed_files += 1
                    LOGGER.debug("Failed to parse %s: %s", result.path, result.error)
                else:
                    self.stats.parsed_files += 1
                parsed.append(result)
        return parsed

    def _extract_scenes(self, parsed_files: list[ParsedFile]) -> list[ExtractedScene]:
        scenes: list[ExtractedScene] = []
        for parsed in progress(parsed_files, desc="Extracting scenes", unit="file"):
            extracted = self.scene_extractor.extract(parsed)
            self.stats.discovered_scenes += len(extracted)
            scenes.extend(extracted)
        return scenes

    def _build_examples(self, scenes: list[ExtractedScene]) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        for scene in progress(scenes, desc="Building examples", unit="scene"):
            metadata = self.metadata_generator.generate(scene)
            validation = self.validator.validate(scene)
            if not validation.valid:
                LOGGER.debug("Skipping invalid scene %s: %s", scene.scene_name, validation.errors)
                continue
            if metadata.quality_score < self.config.min_quality_score:
                LOGGER.debug(
                    "Skipping low-quality scene %s with score %.3f.",
                    scene.scene_name,
                    metadata.quality_score,
                )
                continue
            self.stats.valid_scenes += 1
            self.stats.bump_category(metadata.category)
            self.stats.bump_difficulty(metadata.difficulty)
            for prompt in self.prompt_generator.generate(metadata):
                examples.append(self.formatter.to_chat_example(scene, prompt, metadata, validation))
        return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Manim scenes into LLM training JSONL.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Repository directories or Python files to scan.")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"), help="Directory for JSONL outputs.")
    parser.add_argument("--source-repo", default=None, help="Source repository label, e.g. 3b1b/videos.")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel parser worker count.")
    parser.add_argument("--batch-size", type=int, default=256, help="Scanner batch size for callers.")
    parser.add_argument("--validation-ratio", type=float, default=0.05, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Test split ratio.")
    parser.add_argument("--min-quality-score", type=float, default=0.35, help="Minimum metadata quality score.")
    parser.add_argument("--render", action="store_true", help="Run optional Manim render validation.")
    parser.add_argument("--manim-binary", default="manim", help="Manim executable name or path.")
    parser.add_argument("--render-timeout", type=int, default=90, help="Render timeout in seconds.")
    parser.add_argument("--prompt-augmentations", type=int, default=0, help="Synthetic prompt variants per scene.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExtractorConfig:
    """Build typed config from CLI args."""

    return ExtractorConfig(
        input_paths=args.inputs,
        output_dir=args.output_dir,
        source_repo=args.source_repo,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        min_quality_score=args.min_quality_score,
        run_render_validation=args.render,
        render_timeout_seconds=args.render_timeout,
        manim_binary=args.manim_binary,
        prompt_augmentation_count=args.prompt_augmentations,
        log_level=args.log_level,
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = build_config(args)
    pipeline = ExtractionPipeline(config)
    paths = pipeline.run()
    for split, path in paths.items():
        LOGGER.info("%s: %s", split, path)


if __name__ == "__main__":
    main()
