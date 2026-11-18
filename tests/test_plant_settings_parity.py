"""The Central Plant page and the backend must agree on every default.

``frontend/src/utils/plant-settings.ts`` mirrors ``DEFAULT_CENTRAL_PLANT_SETTINGS``
by hand, so a one-sided edit leaves the page showing a value the backend never
uses. That reads exactly like a setting being ignored, and it is the same symptom
as a setting read from the wrong settings block — the panel promises one number
while the plant runs on another.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from custom_components.roommind.const import DEFAULT_CENTRAL_PLANT_SETTINGS

PLANT_SETTINGS_TS = Path(__file__).resolve().parent.parent / "frontend" / "src" / "utils" / "plant-settings.ts"
_ENTRY = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*):\s*(.+?),?\s*$")


def _frontend_defaults(source: str) -> dict[str, Any]:
    """Parse the ``DEFAULT_CENTRAL_PLANT`` object literal into plain Python values."""
    block = source.split("export const DEFAULT_CENTRAL_PLANT", 1)[1].split("\n};", 1)[0]
    defaults: dict[str, Any] = {}
    for line in block.splitlines():
        match = _ENTRY.match(line)
        if match is None:
            continue
        key, raw = match.group(1), match.group(2).rstrip(",").strip()
        if len(raw) >= 2 and raw.startswith('"') and raw.endswith('"'):
            defaults[key] = raw[1:-1]
        elif raw in ("true", "false"):
            defaults[key] = raw == "true"
        else:
            defaults[key] = float(raw)
    return defaults


def _comparable(value: Any) -> Any:
    """Compare numbers as floats while keeping booleans booleans."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return float(value)
    return value


def test_the_frontend_plant_defaults_match_the_backend():
    """Every default shown on the Central Plant page is the one the backend falls back to."""
    if not PLANT_SETTINGS_TS.exists():
        pytest.skip("frontend sources are not part of this checkout")
    frontend = _frontend_defaults(PLANT_SETTINGS_TS.read_text(encoding="utf-8"))
    backend = {key: _comparable(value) for key, value in DEFAULT_CENTRAL_PLANT_SETTINGS.items()}

    assert set(frontend) == set(backend), "the two sides no longer declare the same keys"
    assert frontend == backend
