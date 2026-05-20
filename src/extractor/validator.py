"""Validation for extracted Manim scenes."""

from __future__ import annotations

import ast
import logging
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from .config import ExtractorConfig
from .scene_extractor import ExtractedScene

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ValidationResult:
    """Validation status and diagnostics for one scene."""

    valid: bool
    status: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    render_preview_path: str | None = None


class SceneValidator:
    """Validate syntax, required scene structure, and optional renderability."""

    def __init__(self, config: ExtractorConfig) -> None:
        self.config = config

    def validate(self, scene: ExtractedScene) -> ValidationResult:
        """Run static checks and optional Manim render check."""

        errors: list[str] = []
        warning_messages: list[str] = []

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                ast.parse(scene.assistant_code)
        except SyntaxError as exc:
            errors.append(f"Syntax error: {exc}")

        if not scene.has_construct:
            errors.append("Scene class has no construct() method.")
        if not scene.scene_name:
            errors.append("Scene name is empty.")
        if self.config.strict_dependency_validation and scene.missing_symbols:
            errors.append(
                "Unresolved dependency symbols: " + ", ".join(scene.missing_symbols)
            )
        has_manim_import = any(
            token in scene.assistant_code
            for token in ("from manim import", "import manim", "manim_imports_ext")
        )
        if not has_manim_import:
            warning_messages.append("No explicit Manim import found.")

        if errors:
            return ValidationResult(valid=False, status="failed", errors=errors, warnings=warning_messages)

        if self.config.run_import_validation or self.config.run_instantiation_validation:
            runtime_result = self._validate_runtime_import(scene)
            runtime_result.warnings.extend(warning_messages)
            if not runtime_result.valid or not self.config.run_render_validation:
                return runtime_result

        if self.config.run_render_validation:
            render_result = self._render(scene)
            render_result.warnings.extend(warning_messages)
            return render_result

        return ValidationResult(valid=True, status="valid", warnings=warning_messages)

    def _validate_runtime_import(self, scene: ExtractedScene) -> ValidationResult:
        """Import extracted code in an isolated subprocess.

        Import validation is opt-in because many upstream Manim repositories need
        project-specific packages or old Manim versions. Keeping this subprocess
        isolated prevents repository code from mutating the extractor process.
        """

        with tempfile.TemporaryDirectory(prefix="manim_extract_import_") as tmp:
            temp_dir = Path(tmp)
            scene_file = temp_dir / f"{scene.scene_name}.py"
            scene_file.write_text(scene.assistant_code, encoding="utf-8")
            validation_code = (
                "import importlib.util\n"
                f"spec = importlib.util.spec_from_file_location('candidate_scene', r'{scene_file}')\n"
                "module = importlib.util.module_from_spec(spec)\n"
                "assert spec and spec.loader\n"
                "spec.loader.exec_module(module)\n"
            )
            if self.config.run_instantiation_validation:
                validation_code += (
                    f"scene_type = getattr(module, '{scene.scene_name}')\n"
                    "scene_type()\n"
                )
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", validation_code],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.config.render_timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return ValidationResult(
                    valid=False,
                    status="failed",
                    errors=["Runtime import validation timed out."],
                )
        if completed.returncode != 0:
            return ValidationResult(
                valid=False,
                status="failed",
                errors=[completed.stderr.strip() or completed.stdout.strip()],
            )
        return ValidationResult(valid=True, status="valid")

    def _render(self, scene: ExtractedScene) -> ValidationResult:
        manim_path = shutil.which(self.config.manim_binary)
        if not manim_path:
            return ValidationResult(
                valid=True,
                status="incompatible",
                warnings=["Manim binary was not found; skipped render validation."],
            )

        with tempfile.TemporaryDirectory(prefix="manim_extract_") as tmp:
            temp_dir = Path(tmp)
            scene_file = temp_dir / f"{scene.scene_name}.py"
            scene_file.write_text(scene.assistant_code, encoding="utf-8")
            command = [
                manim_path,
                "-ql",
                "--disable_caching",
                str(scene_file),
                scene.scene_name,
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.config.render_timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return ValidationResult(
                    valid=False,
                    status="failed",
                    errors=["Render validation timed out."],
                )
            if completed.returncode != 0:
                return ValidationResult(
                    valid=False,
                    status="failed",
                    errors=[completed.stderr.strip() or completed.stdout.strip()],
                )
            return ValidationResult(valid=True, status="valid")
