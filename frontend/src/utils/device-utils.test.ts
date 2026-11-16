import { describe, expect, test } from "vitest";
import type { HomeAssistant } from "../types";
import { fanModeLabel, thermostatFanModes } from "./device-utils";

function hassWith(fanModes?: string[]): HomeAssistant {
  return {
    states: {
      "climate.plant": {
        entity_id: "climate.plant",
        state: "heat",
        attributes: fanModes ? { fan_modes: fanModes } : {},
      },
    },
    formatEntityAttributeValue: (_st: unknown, _attr: string, value?: unknown) =>
      `t:${String(value)}`,
  } as unknown as HomeAssistant;
}

describe("thermostatFanModes", () => {
  test("uses the thermostat's own fan_modes, translated", () => {
    const options = thermostatFanModes(hassWith(["low", "high"]), "climate.plant");
    expect(options).toEqual([
      { value: "low", label: "t:low" },
      { value: "high", label: "t:high" },
    ]);
  });

  test("falls back to the shared guesses without a thermostat", () => {
    const options = thermostatFanModes(hassWith(), "climate.plant");
    expect(options.map((o) => o.value)).toEqual(["off", "low", "medium", "high", "auto"]);
  });

  test("deduplicates repeated modes", () => {
    const options = thermostatFanModes(hassWith(["low", "low", "high"]), "climate.plant");
    expect(options.map((o) => o.value)).toEqual(["low", "high"]);
  });
});

describe("fanModeLabel", () => {
  test("falls back to the raw mode without a formatter", () => {
    const hass = { states: {} } as unknown as HomeAssistant;
    expect(fanModeLabel(hass, "low")).toBe("low");
  });
});
