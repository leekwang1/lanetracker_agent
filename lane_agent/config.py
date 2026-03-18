from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def _parse_scalar(text: str) -> Any:
    s = text.strip()
    if not s:
        return ""
    low = s.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except ValueError:
        return s


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    data: Dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        data[key.strip()] = _parse_scalar(value)
    return data
