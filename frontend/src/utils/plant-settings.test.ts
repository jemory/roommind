import { describe, expect, test } from "vitest";
import type { CentralPlantConfig, PriceShiftModes } from "../types";
import {
  DEFAULT_CENTRAL_PLANT,
  PLANT_FIELDS,
  PLANT_LIMITS,
  readPlantFields,
  writePlantFields,
} from "./plant-settings";

describe("central plant settings", () => {
  test("saves price_shift_modes with the rest of the block", () => {
    // The price category edits this field, so dropping it from the write mapping silently
    // discarded every change to it.
    const modes: readonly PriceShiftModes[] = ["cool_only", "both"];
    for (const mode of modes) {
      const fields = readPlantFields({ ...DEFAULT_CENTRAL_PLANT, price_shift_modes: mode });
      expect(writePlantFields(fields).price_shift_modes).toBe(mode);
    }
  });

  test("round-trips the whole block", () => {
    const stored: CentralPlantConfig = {
      ...DEFAULT_CENTRAL_PLANT,
      thermostat_entity: "climate.house",
      thermostat_area_id: "living_room",
      nudge_delta_c: 2.1,
      max_nudge_headroom_c: 4.5,
      fallback_heat_c: 17,
      fallback_cool_c: 27,
      min_deadband_c: 2.5,
      circulate_fan_mode: "low",
      idle_fan_mode: "off",
      min_run_minutes: 15,
      min_off_minutes: 8,
      min_fan_minutes: 12,
      changeover_minutes: 45,
      follow_thermostat_setpoint: false,
      thermostat_override_hours: 3,
      exterior_open_policy: "pause_always",
      humidity_comfort_enabled: false,
      dehumidify_rh_threshold: 55,
      max_dehumidify_overcool_c: 1.4,
      wet_coil_lockout_minutes: 40,
      wet_coil_max_rh: 65,
      price_shift_modes: "both",
    };
    expect(writePlantFields(readPlantFields(stored))).toEqual(stored);
  });

  test("the written block has every stored key", () => {
    // A field missing from the write mapping means the page edits it and never saves it
    const written = writePlantFields(readPlantFields(DEFAULT_CENTRAL_PLANT));
    expect(Object.keys(written).sort()).toEqual(Object.keys(DEFAULT_CENTRAL_PLANT).sort());
  });

  test("reads and writes every page field", () => {
    expect(Object.keys(readPlantFields(DEFAULT_CENTRAL_PLANT)).sort()).toEqual(
      [...PLANT_FIELDS].sort(),
    );
    expect(Object.keys(writePlantFields(readPlantFields(DEFAULT_CENTRAL_PLANT)))).toHaveLength(
      PLANT_FIELDS.length,
    );
  });

  test("bounds mirror the backend schema (vol.Range in websocket_api.py)", () => {
    expect(PLANT_LIMITS).toEqual({
      fallbackHeatC: { min: 5.0, max: 25.0 },
      fallbackCoolC: { min: 18.0, max: 35.0 },
      nudgeDeltaC: { min: 0.5, max: 3.5 },
      maxNudgeHeadroomC: { min: 0.5, max: 6.0 },
      minDeadbandC: { min: 1.0, max: 5.0 },
      minRunMinutes: { min: 1, max: 60 },
      minOffMinutes: { min: 1, max: 30 },
      minFanMinutes: { min: 1, max: 60 },
      changeoverMinutes: { min: 5, max: 180 },
      thermostatOverrideHours: { min: 0.0, max: 24.0 },
      dehumidifyRhThreshold: { min: 40.0, max: 80.0 },
      maxDehumidifyOvercoolC: { min: 0.0, max: 2.5 },
      wetCoilLockoutMinutes: { min: 0, max: 120 },
      wetCoilMaxRh: { min: 35.0, max: 80.0 },
    });
  });

  test("every default sits inside its bounds", () => {
    const fields = readPlantFields(DEFAULT_CENTRAL_PLANT);
    for (const [field, limit] of Object.entries(PLANT_LIMITS)) {
      const value = fields[field];
      expect(typeof value, field).toBe("number");
      expect(value as number, field).toBeGreaterThanOrEqual(limit.min);
      expect(value as number, field).toBeLessThanOrEqual(limit.max);
    }
  });
});
