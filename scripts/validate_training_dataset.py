"""Validate JSONL chat data before model fine-tuning."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def validate(path: Path, max_rows: int | None = None) -> dict[str, object]:
    total = 0
    bad = 0
    prompt_lengths: list[int] = []
    code_lengths: list[int] = []
    categories: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if max_rows is not None and total >= max_rows:
                break
            total += 1
            try:
                row = json.loads(line)
                messages = row["messages"]
                metadata = row["metadata"]
                user = next(message["content"] for message in messages if message["role"] == "user")
                assistant = next(
                    message["content"] for message in messages if message["role"] == "assistant"
                )
                if not user.strip() or not assistant.strip():
                    raise ValueError("empty prompt or assistant code")
                prompt_lengths.append(len(user))
                code_lengths.append(len(assistant))
                categories[str(metadata.get("category", "unknown"))] += 1
                difficulties[str(metadata.get("difficulty", "unknown"))] += 1
            except Exception as exc:
                bad += 1
                print(f"Bad row {line_no}: {exc}")

    return {
        "file": str(path),
        "rows_checked": total,
        "bad_rows": bad,
        "avg_prompt_chars": round(sum(prompt_lengths) / max(len(prompt_lengths), 1), 2),
        "avg_code_chars": round(sum(code_lengths) / max(len(code_lengths), 1), 2),
        "max_code_chars": max(code_lengths or [0]),
        "categories": dict(categories),
        "difficulties": dict(difficulties),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate JSONL chat fine-tuning rows.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate(args.path, args.max_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
