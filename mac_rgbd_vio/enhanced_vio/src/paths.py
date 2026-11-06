"""Enforce the task's output isolation boundary in every command-line entrypoint."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def output_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Outputs must remain inside {ROOT}: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path
