"""Format extracted scenes as instruction-tuning training examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .metadata_generator import SceneMetadata
from .scene_extractor import ExtractedScene
from .validator import ValidationResult


@dataclass(slots=True)
class ChatMessage:
    """One chat-format instruction tuning message."""

    role: str
    content: str


@dataclass(slots=True)
class TrainingExample:
    """A fine-tuning-ready chat example with metadata."""

    messages: list[ChatMessage]
    metadata: dict[str, Any]


class DatasetFormatter:
    """Convert scenes into JSON, JSONL, and Hugging Face-friendly rows."""

    def to_chat_example(
        self,
        scene: ExtractedScene,
        prompt: str,
        metadata: SceneMetadata,
        validation: ValidationResult,
    ) -> TrainingExample:
        """Build one OpenAI/HF chat-style instruction example."""

        meta = asdict(metadata)
        meta["validation"] = {
            "status": validation.status,
            "warnings": validation.warnings,
            "errors": validation.errors,
        }
        meta["content_hash"] = scene.content_hash
        meta["extraction_diagnostics"] = {
            "included_dependency_symbols": list(scene.dependency_tree),
            "missing_symbols": scene.missing_symbols,
            "dependency_tree": scene.dependency_tree,
        }
        return TrainingExample(
            messages=[
                ChatMessage(role="user", content=prompt),
                ChatMessage(role="assistant", content=scene.assistant_code),
            ],
            metadata=meta,
        )

    def to_dict(self, example: TrainingExample) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return {
            "messages": [asdict(message) for message in example.messages],
            "metadata": example.metadata,
        }

    def to_hf_instruction(self, example: TrainingExample) -> dict[str, Any]:
        """Return a common Hugging Face instruction/input/output row."""

        user = next(message.content for message in example.messages if message.role == "user")
        assistant = next(message.content for message in example.messages if message.role == "assistant")
        return {
            "instruction": user,
            "input": "",
            "output": assistant,
            "messages": [asdict(message) for message in example.messages],
            "metadata": example.metadata,
        }
