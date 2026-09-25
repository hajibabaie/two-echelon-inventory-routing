"""Write JSON files that stay readable in a diff: one top-level key and one matrix row per line."""

import json
from pathlib import Path


def dump_rows(record: dict, path: Path) -> None:
    """Write a JSON object with one key per line and each row of a nested list on its own line."""
    lines = []
    for key, value in record.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
            rows = ",\n    ".join(json.dumps(row) for row in value)
            text = f"[\n    {rows}\n  ]"
        else:
            text = json.dumps(value)
        lines.append(f"  {json.dumps(key)}: {text}")
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write("{\n" + ",\n".join(lines) + "\n}\n")
