"""Shared harness for the Central Plant coordinator tests.

The harness itself lives in ``tests/conftest.py`` so the integration suite can
use it too; this module keeps the ``from .conftest import ...`` imports local.
"""

from __future__ import annotations

from tests.conftest import (  # noqa: F401 - re-exported for the test modules
    DEFAULT_THERMOSTAT_ATTRIBUTES,
    THERMOSTAT_ENTITY,
    FakeRegistry,
    FakeState,
    FakeStates,
    Harness,
    make_room,
)
