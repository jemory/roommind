"""Integration suite: the whole control loop with the real store and managers.

The shared fixtures and the coordinator harness live in ``tests/conftest.py``;
this module only re-exports the helpers the integration modules import.
"""

from __future__ import annotations

from tests.conftest import (  # noqa: F401 - re-exported for the test modules
    DEFAULT_THERMOSTAT_ATTRIBUTES,
    THERMOSTAT_ENTITY,
    Harness,
    make_room,
)
