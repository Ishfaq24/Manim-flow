# Project Structure

This project follows a source-layout Python package plus explicit data and model artifact folders.

```text
src/extractor
```

Contains the production extractor package. Keep importable library code here.

```text
scripts
```

Contains operational commands for validation, cleanup, and training smoke tests.

```text
configs
```

Contains JSON/YAML configuration for training and extraction runs.

```text
data/raw
```

Contains external source repositories. Do not commit raw repos.

```text
data/processed
```

Contains generated datasets. Do not commit generated JSONL files unless you explicitly want a small fixture.

```text
models
```

Contains local adapters and checkpoints. Do not commit model artifacts.
