"""Small LoRA training smoke test for the Manim dataset.

This is intentionally not the final production fine-tune. It verifies that:
- JSONL chat examples load correctly
- prompt/assistant text can be tokenized
- a LoRA adapter can train and save
- the resulting adapter can generate a short response
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


LOGGER = logging.getLogger("train_lora_smoke")


@dataclass(slots=True)
class SmokeConfig:
    model_name: str
    train_file: str
    validation_file: str
    output_dir: str
    max_train_samples: int = 128
    max_eval_samples: int = 32
    max_seq_length: int = 512
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    logging_steps: int = 5
    save_steps: int = 25
    eval_steps: int = 25
    seed: int = 42
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


def load_config(path: Path) -> SmokeConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SmokeConfig(**payload)


def chat_to_text(example: dict[str, Any]) -> dict[str, str]:
    messages = example["messages"]
    user = next(message["content"] for message in messages if message["role"] == "user")
    assistant = next(message["content"] for message in messages if message["role"] == "assistant")
    return {
        "text": (
            "### Instruction:\n"
            f"{user}\n\n"
            "### Response:\n"
            f"{assistant}"
        )
    }


def tokenize_dataset(dataset: Any, tokenizer: Any, max_seq_length: int) -> Any:
    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def infer_lora_targets(model: torch.nn.Module) -> list[str]:
    """Choose LoRA target modules that work across GPT-2 and many decoder models."""

    names = {
        name.split(".")[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) or module.__class__.__name__ == "Conv1D"
    }
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "c_attn", "c_proj"]
    targets = [name for name in preferred if name in names]
    if targets:
        return targets
    return sorted(names)[:4]


def make_training_args(config: SmokeConfig, use_cuda: bool) -> TrainingArguments:
    """Create TrainingArguments across Transformers versions.

    Newer prerelease Transformers builds occasionally rename or remove kwargs.
    Filtering by the live signature keeps this smoke test focused on pipeline
    validation instead of package-version trivia.
    """

    kwargs: dict[str, Any] = {
        "output_dir": config.output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "save_total_limit": 2,
        "report_to": [],
        "fp16": use_cuda,
        "no_cuda": not use_cuda,
        "use_cpu": not use_cuda,
        "seed": config.seed,
    }
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    return TrainingArguments(**filtered)


def run(config: SmokeConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(config.seed)

    train_path = Path(config.train_file)
    validation_path = Path(config.validation_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not validation_path.exists():
        raise FileNotFoundError(f"Validation file not found: {validation_path}")

    LOGGER.info("Loading dataset")
    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(validation_path)},
    )
    dataset["train"] = dataset["train"].select(range(min(config.max_train_samples, len(dataset["train"]))))
    dataset["validation"] = dataset["validation"].select(
        range(min(config.max_eval_samples, len(dataset["validation"])))
    )
    dataset = dataset.map(chat_to_text)

    LOGGER.info("Loading tokenizer and model: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    target_modules = infer_lora_targets(model)
    LOGGER.info("Using LoRA target modules: %s", target_modules)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized = {
        split: tokenize_dataset(dataset[split], tokenizer, config.max_seq_length)
        for split in ("train", "validation")
    }

    use_cuda = torch.cuda.is_available()
    training_args = make_training_args(config, use_cuda)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    LOGGER.info("Starting smoke training")
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    prompt = "### Instruction:\nCreate a sine wave graph animation\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    preview_path = Path(config.output_dir) / "generation_preview.txt"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text(generated, encoding="utf-8")
    LOGGER.info("Saved smoke adapter and generation preview to %s", config.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny LoRA smoke training job.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke_lora.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
