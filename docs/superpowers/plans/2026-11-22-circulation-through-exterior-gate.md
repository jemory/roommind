# Circulation Through the Exterior Opening Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Central Plant Circulation run while the Whole-Home Exterior Opening Gate is active, with heating and cooling still blocked.

**Architecture:** One guard change. `CentralPlantManager.allowed_actions` stops returning a single forced `idle` when `exterior_paused`, and returns `{idle, circulating}` with no forced action instead — minus circulation when the wet coil is locked. Nothing else moves: the cold-start rule and the MPC already choose circulation from `legal_actions`, so they decide on their existing terms (opposing comfort errors, a wide spread between occupied rooms, or the plan's benefit against fan energy).

**Tech Stack:** Python 3.12, Home Assistant 2026.5+, pytest + pytest-asyncio, ruff, mypy.

**Spec:** `docs/adr/0006-circulation-through-the-exterior-opening-gate.md`

## Global Constraints

- Home Assistant **2026.5.0** or newer (`hacs.json`).
- `dev/check --quick` must stay green: ruff check + format, mypy, pytest with the **95 %** coverage gate, tsc, eslint, prettier, vitest, vite build.
- Heating and cooling must never become legal while the gate is active.
- Circulation must never become legal while the wet coil is locked, gate or no gate.
- The Guard chip and the "{count} room(s) paused" chip keep their existing wording; "paused" now means heating and cooling only.

## Review Focus

- **Wet coil plus an open window.** A cooling run has just ended, the coil is wet, a window opens: the blower must stay off, because circulating evaporates moisture back into the house. Tested in Task 1.
- **Uniform warm house behind an open window.** Every room above target and nothing to equalise: the plan must stay idle rather than run the blower for nothing. Tested in Task 1; this is the deliberate limit of the chosen approach.
- **A window opening while the furnace runs.** Heating must stop in the same cycle, as today — the gate still overrides an active run instead of letting it finish its minimum run. Pinned by the existing `test_an_exterior_opening_pauses_the_plant`.
- **`pause_always` with the interior door closed.** The room is cut off from the rest of the house, yet the gate still pauses the whole home; circulation must remain available for the rooms that do share air. Tested in Task 1.
- **Card wording.** The plant card can now read "Circulating air" next to "1 room(s) paused". The wording task explains what paused now covers, so it does not read as a contradiction. Task 2.

---

### Task 1: Let circulation through the gate

**Files:**
- Modify: `custom_components/roommind/managers/central_plant_manager.py:275-276`
- Test: `tests/managers/test_central_plant_manager.py` (`TestAllowedActions`, replace `test_exterior_pause_blocks_everything` at line ~233)
- Test: `tests/coordinator/test_plant_guards.py` (add after `test_an_exterior_opening_pauses_the_plant`, line ~125)

**Interfaces:**
- Consumes: `PlantGuardState(legal_actions, forced_action, guard_active, guard_reason, guard_until)`, `MODE_IDLE`, `MODE_CIRCULATING`, `GUARD_EXTERIOR_OPEN`, and the existing `wet_coil_locked: bool = False` keyword of `allowed_actions`.
- Produces: with `exterior_paused=True` and `wet_coil_locked=False`, `allowed_actions` returns `PlantGuardState({MODE_IDLE, MODE_CIRCULATING}, None, True, "exterior_open", None)`. With `wet_coil_locked=True` it returns `PlantGuardState({MODE_IDLE}, None, True, "exterior_open", None)`. `forced_action` is `None` in both.

- [ ] **Step 1: Write the failing unit tests**

Replace `test_exterior_pause_blocks_everything` in `tests/managers/test_central_plant_manager.py` with these two tests:

```python
    def test_exterior_pause_keeps_conditioning_out_but_allows_circulation(self) -> None:
        state = CentralPlantManager().allowed_actions({}, "auto", None, {}, exterior_paused=True, now_ts=1000)
        assert state == PlantGuardState({MODE_IDLE, MODE_CIRCULATING}, None, True, "exterior_open", None)

    def test_a_wet_coil_keeps_circulation_out_of_an_exterior_pause(self) -> None:
        state = CentralPlantManager().allowed_actions(
            {}, "auto", None, {}, exterior_paused=True, wet_coil_locked=True, now_ts=1000
        )
        assert state == PlantGuardState({MODE_IDLE}, None, True, "exterior_open", None)
```

- [ ] **Step 2: Write the failing coordinator tests**

Add these three tests to `tests/coordinator/test_plant_guards.py`, after `test_an_exterior_opening_pauses_the_plant`:

```python
async def test_circulation_runs_through_an_exterior_pause(harness: Harness, fake_states):
    fake_states.set("sensor.warm_temp", "27.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.cold_temp", "18.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.outdoor_temp", "20.0", {"unit_of_measurement": "°C"})
    fake_states.set("binary_sensor.window", "on")
    await harness.seed(
        {
            "warm_room": make_room(
                {
                    "temperature_sensor": "sensor.warm_temp",
                    "exterior_window_sensors": ["binary_sensor.window"],
                    "window_open_delay": 0,
                }
            ),
            "cold_room": make_room({"temperature_sensor": "sensor.cold_temp"}),
        },
        settings={"central_plant": {"thermostat_entity": THERMOSTAT_ENTITY, "exterior_open_policy": "pause_always"}},
    )

    data = await harness.cycle()

    # Two rooms pulling apart: the blower fixes that without conditioning the house.
    assert data["central_plant"]["paused"] is True
    assert data["central_plant"]["action"] == MODE_CIRCULATING


async def test_a_uniform_warm_house_stays_idle_behind_an_exterior_pause(harness: Harness, fake_states):
    fake_states.set("sensor.room_temp", "26.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.outdoor_temp", "20.0", {"unit_of_measurement": "°C"})
    fake_states.set("binary_sensor.window", "on")
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "comfort_cool": 22.0,
                    "exterior_window_sensors": ["binary_sensor.window"],
                    "window_open_delay": 0,
                }
            )
        },
        settings={"central_plant": {"thermostat_entity": THERMOSTAT_ENTITY, "exterior_open_policy": "pause_always"}},
    )

    data = await harness.cycle()

    # Nothing to equalise: circulating would only spend fan energy.
    assert data["central_plant"]["paused"] is True
    assert data["central_plant"]["action"] == MODE_IDLE


async def test_pause_always_still_allows_circulation_when_the_door_is_closed(harness: Harness, fake_states):
    fake_states.set("sensor.warm_temp", "27.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.cold_temp", "18.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.outdoor_temp", "20.0", {"unit_of_measurement": "°C"})
    fake_states.set("binary_sensor.window", "on")
    fake_states.set("binary_sensor.door", "off")
    await harness.seed(
        {
            "warm_room": make_room(
                {
                    "temperature_sensor": "sensor.warm_temp",
                    "exterior_window_sensors": ["binary_sensor.window"],
                    "interior_door_sensors": ["binary_sensor.door"],
                    "interior_door_delay": 0,
                    "window_open_delay": 0,
                }
            ),
            "cold_room": make_room({"temperature_sensor": "sensor.cold_temp"}),
        },
        settings={"central_plant": {"thermostat_entity": THERMOSTAT_ENTITY, "exterior_open_policy": "pause_always"}},
    )

    data = await harness.cycle()

    # pause_always ignores the closed door, but the rooms that do share air can still mix.
    assert data["central_plant"]["paused"] is True
    assert data["central_plant"]["action"] == MODE_CIRCULATING
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/managers/test_central_plant_manager.py::TestAllowedActions tests/coordinator/test_plant_guards.py -q`
Expected: FAIL — the two manager assertions disagree on `legal_actions`/`forced_action`, and the three coordinator tests see `action == "idle"`.

- [ ] **Step 4: Change the guard**

In `custom_components/roommind/managers/central_plant_manager.py`, replace the two-line exterior return with:

```python
        if exterior_paused:
            # Heating and cooling stay out, but the blower may still move air along
            # an open path. The wet coil still forbids circulation: evaporating
            # moisture off it raises indoor humidity.
            legal = {MODE_IDLE} if wet_coil_locked else {MODE_IDLE, MODE_CIRCULATING}
            return PlantGuardState(legal, None, True, GUARD_EXTERIOR_OPEN, None)
```

`forced_action` becomes `None`: a forced `idle` would stop the controllers from choosing circulation from a legal set, which is the whole point.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/managers/test_central_plant_manager.py tests/coordinator/test_plant_guards.py -q`
Expected: PASS, including the untouched `test_an_exterior_opening_pauses_the_plant` (one room, nothing to equalise, still idle).

- [ ] **Step 6: Commit**

```bash
git add custom_components/roommind/managers/central_plant_manager.py tests/managers/test_central_plant_manager.py tests/coordinator/test_plant_guards.py
git commit -m "feat: let circulation run while an exterior opening pauses the plant"
```

---

### Task 2: Say what the gate actually pauses

**Files:**
- Modify: `docs/features.md:148`
- Modify: `README.md:24`, `README.md:196`

**Interfaces:**
- Consumes: nothing from Task 1 at runtime; it documents the behavior Task 1 shipped.
- Produces: no code. User-facing copy only.

- [ ] **Step 1: Update the feature description**

In `docs/features.md`, replace the first sentence of the **Exterior Openings** paragraph:

> While one is open the plant pauses — heating a house with an open window wastes energy, and cooling one with an open door never reaches its target.

with:

> While one is open, heating and cooling pause — heating a house with an open window wastes energy, and cooling one with an open door never reaches its target. Circulation is still allowed: with an open path through the house the blower is what distributes outdoor air, so it runs whenever mixing the rooms would even them out.

- [ ] **Step 2: Update the README summary line**

In `README.md`, replace line 24's opening clause:

> An open window pauses the whole home; a closed **Interior Door** turns that into a local problem so the rest of the house keeps heating.

with:

> An open window pauses heating and cooling for the whole home; a closed **Interior Door** turns that into a local problem so the rest of the house keeps heating.

- [ ] **Step 3: Update the README definition**

In `README.md`, replace the first sentence of the **Exterior Opening** definition on line 196:

> An **Exterior Opening** is a window or door to the outdoors: while it is open the plant pauses, because the envelope is no longer a closed system.

with:

> An **Exterior Opening** is a window or door to the outdoors: while it is open heating and cooling pause, because the envelope is no longer a closed system. Circulation does not: the blower only moves air, and with an opening it is what carries outdoor air through the house.

- [ ] **Step 4: Verify the copy builds and nothing else broke**

Run: `./dev/check --quick`
Expected: all nine checks pass (this task touches no code, but the gate also covers the suite).

- [ ] **Step 5: Commit**

```bash
git add docs/features.md README.md docs/adr/0006-circulation-through-the-exterior-opening-gate.md
git commit -m "docs: the exterior opening gate pauses conditioning, not circulation"
```
