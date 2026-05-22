# Manim Dataset Extractor

Production-oriented tooling for converting Manim scene repositories into JSONL chat datasets for prompt-to-code fine-tuning.

## Project Layout

```text
.
|-- src/extractor/                 # AST extractor package
|-- scripts/                       # training, validation, and maintenance commands
|-- configs/                       # training and pipeline configs
|-- data/
|   |-- raw/3b1b-videos/           # cloned upstream source repo, ignored by git
|   `-- processed/3b1b-videos/     # generated train/validation/test JSONL
|-- models/                        # trained adapters/checkpoints, ignored by git
|-- docs/                          # project notes and operating guides
|-- examples/                      # tiny local sample repo
|-- tests/                         # future unit/integration tests
|-- requirements.txt               # extractor runtime dependencies
|-- requirements-train.txt         # training dependencies
`-- pyproject.toml                 # package metadata
```

## Install For Development

```powershell
python -m pip install -e .
```

Training extras:

```powershell
python -m pip install -e ".[train]"
```

## Extract A Dataset

The current 3Blue1Brown clone lives at:

```text
data/raw/3b1b-videos
```

Regenerate the processed dataset:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_extract_3b1b.ps1
```

The extractor writes:

```text
data/processed/3b1b-videos/train.jsonl
data/processed/3b1b-videos/validation.jsonl
data/processed/3b1b-videos/test.jsonl
data/processed/3b1b-videos/stats.json
data/processed/3b1b-videos/split_integrity_report.json
```

By default, train/validation/test splitting is grouped by each scene's stable
`content_hash`, so prompt augmentations for the same extracted code stay in the
same split. Use `--legacy-row-split` only when reproducing old row-wise outputs.

The extractor also includes referenced module-level helpers, parent classes, and
constants when they are needed by a scene. Use `--no-dependency-context` only if
you need the previous imports-plus-class extraction behavior.

## Validate Training Data

```powershell
python scripts\validate_training_dataset.py data\processed\3b1b-videos\train_small.jsonl
```

## Build Model-Ready Data

After regenerating the source datasets, build a filtered combined dataset:

```powershell
python scripts\build_model_ready_dataset.py
```

This writes:

```text
data/processed/model-ready/train.jsonl
data/processed/model-ready/validation.jsonl
data/processed/model-ready/test.jsonl
data/processed/model-ready/train_small.jsonl
data/processed/model-ready/split_integrity_report.json
```

Rows with unresolved extraction symbols are excluded from `model-ready`; the
original source-specific processed datasets are preserved.

## Smoke Train

This uses a tiny GPT-2 model only to prove the pipeline works. It is not for quality.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_smoke_training.ps1
```

The smoke adapter is written to:

```text
models/manim-smoke-lora
```

## Real Training Direction

For your RTX 2050, start with a small code model such as:

```text
Qwen/Qwen2.5-Coder-1.5B-Instruct
```

For serious quality, train on a larger GPU with:

```text
Qwen/Qwen2.5-Coder-7B-Instruct
```

Use the small dataset first:

```text
data/processed/3b1b-videos/train_small.jsonl
```

Then move to:

```text
data/processed/3b1b-videos/train.jsonl
```
