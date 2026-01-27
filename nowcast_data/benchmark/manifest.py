from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).with_name("manifest.json")


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(path) if path is not None else MANIFEST_PATH
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
