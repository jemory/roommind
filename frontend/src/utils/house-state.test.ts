import { describe, expect, test } from "vitest";
import type { HomeAssistant, HouseLive } from "../types";
import {
  HOUSE_MODE_OPTIONS,
  followingRoomsLabel,
  formatHouseTarget,
  houseHoldLabel,
  houseModeLabel,
  houseOverrideActive,
  housePreset,
  houseSourceLabel,
  houseTarget,
} from "./house-state";

function makeHass(temperature: "°C" | "°F" = "°C", language = "en"): HomeAssistant {
  return { language, config: { unit_system: { temperature } } } as unknown as HomeAssistant;
}

function makeHouse(overrides: Partial<HouseLive> = {}): HouseLive {
  return {
    enabled: true,
    heat_c: 21,
    cool_c: 24,
    eco_heat_c: 17,
    eco_cool_c: 27,
    climate_mode: "auto",
    schedules: [],
    schedule_selector_entity: "",
    preheat_enabled: true,
    price_shift: 0,
    override_heat: null,
    override_cool: null,
    override_until: null,
    override_type: null,
    active: true,
    band: { heat_c: 21, cool_c: 24 },
    source: "target",
    override_active: false,
    schedule_entity: "",
    schedule_name: "",
    schedule_block: null,
    next_change: null,
    following_rooms: 3,
    ...overrides,
  };
}

describe("house state", () => {
  test("the target band comes from the live band, not the stored pair", () => {
    // A schedule block moves the band; the settings pair must not win over it.
    const house = makeHouse({ heat_c: 19, cool_c: 25, band: { heat_c: 20.5, cool_c: 23 } });
    expect(houseTarget(house)).toEqual({ low: 20.5, high: 23 });
    expect(houseTarget(null)).toEqual({ low: null, high: null });
  });

  test("a hold is read from the resolved override_active flag", () => {
    expect(houseOverrideActive(makeHouse())).toBe(false);
    expect(houseOverrideActive(makeHouse({ override_active: true, override_heat: 22 }))).toBe(true);
    expect(houseOverrideActive(null)).toBe(false);
  });

  test("the preset names the override that is being held", () => {
    expect(housePreset(makeHouse())).toBe("none");
    for (const [type, preset] of [
      ["boost", "comfort"],
      ["eco", "eco"],
      ["custom", "manual"],
    ] as const) {
      expect(housePreset(makeHouse({ override_active: true, override_type: type }))).toBe(preset);
    }
  });

  test("mode labels and options cover every climate mode", () => {
    expect(HOUSE_MODE_OPTIONS.map((option) => option.value)).toEqual([
      "auto",
      "heat_only",
      "cool_only",
    ]);
    expect(houseModeLabel(makeHouse({ climate_mode: "auto" }), "en")).toBe("Auto");
    expect(houseModeLabel(makeHouse({ climate_mode: "heat_only" }), "en")).toBe("Heat Only");
    expect(houseModeLabel(makeHouse({ climate_mode: "cool_only" }), "en")).toBe("Cool Only");
    expect(houseModeLabel(null, "en")).toBe("Auto");
  });

  test("the source label names the rung in force, including the disabled layer", () => {
    expect(houseSourceLabel(makeHouse({ active: false }), "en")).toBe("House layer not set up");
    expect(houseSourceLabel(makeHouse({ source: "override" }), "en")).toBe(
      "Holding a temporary target",
    );
    expect(houseSourceLabel(makeHouse({ source: "schedule" }), "en")).toBe(
      "Following the house schedule",
    );
    expect(houseSourceLabel(makeHouse({ source: "target" }), "en")).toBe(
      "Following the house target",
    );
  });

  test("following-room counts use a singular and a plural label", () => {
    expect(followingRoomsLabel(1, "en")).toBe("1 room follows the house");
    expect(followingRoomsLabel(3, "en")).toBe("3 rooms follow the house");
  });

  test("the target formats as a range, a single side, or a placeholder", () => {
    const hass = makeHass();
    expect(formatHouseTarget(makeHouse(), hass)).toBe("21–24 °C");
    expect(formatHouseTarget(makeHouse({ band: { heat_c: 22, cool_c: 22 } }), hass)).toBe("22 °C");
    expect(formatHouseTarget(makeHouse({ band: { heat_c: 22, cool_c: null } }), hass)).toBe(
      "22 °C",
    );
    expect(formatHouseTarget(makeHouse({ band: { heat_c: null, cool_c: null } }), hass)).toBe("—");
  });

  test("the target follows the Home Assistant unit system", () => {
    expect(formatHouseTarget(makeHouse(), makeHass("°F"))).toBe("70–75 °F");
  });

  test("the hold label only exists while a hold with an end time is active", () => {
    expect(houseHoldLabel(makeHouse(), "en")).toBeNull();
    const held = makeHouse({ override_active: true, override_until: 1_800_000_000 });
    expect(houseHoldLabel(makeHouse({ ...held, override_until: null }), "en")).toBeNull();
    expect(houseHoldLabel(held, "en")).toContain("Held until");
  });
});
