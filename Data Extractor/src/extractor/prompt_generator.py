"""Natural-language prompt generation for extracted scenes."""

from __future__ import annotations

import random

from .metadata_generator import SceneMetadata
from .utils import unique_preserve_order


class PromptGenerator:
    """Generate concise instruction-style prompts for LLM fine-tuning."""

    def __init__(self, augmentation_count: int = 0, seed: int = 42) -> None:
        self.augmentation_count = augmentation_count
        self.random = random.Random(seed)

    def generate(self, metadata: SceneMetadata) -> list[str]:
        """Return the primary prompt plus optional synthetic augmentations."""

        base = self._primary_prompt(metadata)
        prompts = [base]
        templates = [
            "Create a Manim scene showing {description}",
            "Write Manim code for {description}",
            "Animate {description}",
            "Build a concise Manim animation showing {description}",
        ]
        description = self._description(metadata)
        for template in self.random.sample(templates, k=min(self.augmentation_count, len(templates))):
            prompts.append(template.format(description=description))
        return unique_preserve_order(prompts)

    def _primary_prompt(self, metadata: SceneMetadata) -> str:
        description = self._description(metadata)
        if metadata.category == "graphs":
            return f"Visualize {description}"
        if metadata.category == "geometry":
            return f"Create {description}"
        if metadata.category == "text":
            return f"Show {description}"
        return f"Create a Manim animation showing {description}"

    def _description(self, metadata: SceneMetadata) -> str:
        objects = readable_join(metadata.objects[:3])
        animations = readable_join(metadata.animations[:2])

        if metadata.category == "graphs":
            subject = "a mathematical graph"
            if "Axes" in metadata.objects or "NumberPlane" in metadata.objects:
                subject = "a graph with axes"
            return self._with_animation(subject, animations)
        if metadata.category == "geometry":
            subject = f"a {objects.lower()} scene" if objects else "a geometric scene"
            return self._with_animation(subject, animations)
        if metadata.category == "math":
            subject = "mathematical expressions"
            return self._with_animation(subject, animations)
        if metadata.category == "physics":
            return self._with_animation("a physics concept", animations)
        if metadata.category == "3d":
            return self._with_animation("a 3D visualization", animations)
        if objects:
            return self._with_animation(f"{objects.lower()}", animations)
        if animations:
            return f"uses {animations}"
        scene_words = split_identifier(metadata.scene_name)
        return f"illustrates {scene_words}"

    def _with_animation(self, subject: str, animations: str) -> str:
        if animations:
            return f"{subject} using {animations}"
        return subject


def split_identifier(value: str) -> str:
    """Convert CamelCase or snake_case identifiers into readable words."""

    chars: list[str] = []
    previous = ""
    for char in value.replace("_", " "):
        if previous and char.isupper() and (previous.islower() or previous.isdigit()):
            chars.append(" ")
        chars.append(char)
        previous = char
    return " ".join("".join(chars).split()).lower()


def readable_join(values: list[str]) -> str:
    """Join labels into compact natural language."""

    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"
