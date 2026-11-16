import { describe, expect, test } from "vitest";
import en from "../locales/en.json";
import de from "../locales/de.json";
import fr from "../locales/fr.json";
import it from "../locales/it.json";
import type { HouseConfig } from "../types";
import { targetReasons } from "./target-reason";
import {
  HOUSE_DEFAULTS,
  HOUSE_PRICE_SHIFT_MAX,
  HOUSE_TEMP_BOUNDS,
  houseModeOrDefault,
  validateHouseSettings,
} from "./house-settings";

type Validatable = Pick<
  HouseConfig,
  "heat_c" | "cool_c" | "eco_heat_c" | "eco_cool_c" | "price_shift"
>;

function valid(overrides: Partial<Validatable> = {}): Validatable {
  return {
    heat_c: HOUSE_DEFAULTS.heat_c,
    cool_c: HOUSE_DEFAULTS.cool_c,
    eco_heat_c: HOUSE_DEFAULTS.eco_heat_c,
    eco_cool_c: HOUSE_DEFAULTS.eco_cool_c,
    price_shift: HOUSE_DEFAULTS.price_shift,
    ...overrides,
  };
}

describe("house settings", () => {
  test("bounds mirror the backend schema", () => {
    // const.MIN_TARGET_TEMP / MAX_TARGET_TEMP and MAX_PRICE_SHIFT_K.
    expect(HOUSE_TEMP_BOUNDS).toEqual({ min: 5, max: 35 });
    expect(HOUSE_PRICE_SHIFT_MAX).toBe(2);
    expect(HOUSE_DEFAULTS.heat_c).toBe(21);
    expect(HOUSE_DEFAULTS.cool_c).toBe(24);
    expect(HOUSE_DEFAULTS.eco_heat_c).toBe(17);
    expect(HOUSE_DEFAULTS.eco_cool_c).toBe(27);
    expect(HOUSE_DEFAULTS.enabled).toBe(true);
  });

  test("a sane block validates", () => {
    expect(validateHouseSettings(valid())).toBeNull();
    expect(validateHouseSettings(valid({ heat_c: 18, cool_c: 26, price_shift: 2 }))).toBeNull();
  });

  test("a comfort pair must keep cooling above heating", () => {
    expect(validateHouseSettings(valid({ heat_c: 24, cool_c: 24 }))).toBe("band_order");
    expect(validateHouseSettings(valid({ heat_c: 25, cool_c: 20 }))).toBe("band_order");
  });

  test("a one-sided comfort pair is valid, the order check only runs with both sides", () => {
    // A heat-only house leaves cool_c null; the eco pair still has to be ordered.
    expect(validateHouseSettings(valid({ heat_c: 24, cool_c: null }))).toBeNull();
    expect(validateHouseSettings(valid({ heat_c: null, cool_c: 20 }))).toBeNull();
  });

  test("the eco pair only has to stay ordered", () => {
    expect(validateHouseSettings(valid({ eco_heat_c: 20, eco_cool_c: 19 }))).toBe("eco_band_order");
    expect(validateHouseSettings(valid({ eco_heat_c: 20, eco_cool_c: 20 }))).toBeNull();
  });

  test("temperatures must stay inside the plausibility bounds", () => {
    expect(validateHouseSettings(valid({ heat_c: 4 }))).toBe("temp_range");
    expect(validateHouseSettings(valid({ cool_c: 36 }))).toBe("temp_range");
    expect(validateHouseSettings(valid({ eco_cool_c: 40 }))).toBe("temp_range");
  });

  test("the price shift stays between zero and the backend maximum", () => {
    expect(validateHouseSettings(valid({ price_shift: -0.5 }))).toBe("price_range");
    expect(validateHouseSettings(valid({ price_shift: 2.5 }))).toBe("price_range");
  });

  test("the order of the checks is stable", () => {
    // Both the band and the temperature are wrong: band wins so the user fixes the
    // structural error first.
    expect(validateHouseSettings(valid({ heat_c: 30, cool_c: 10 }))).toBe("band_order");
  });

  test("the house mode falls back to auto", () => {
    expect(houseModeOrDefault({ climate_mode: "heat_only" })).toBe("heat_only");
    expect(houseModeOrDefault(null)).toBe("auto");
    expect(houseModeOrDefault({})).toBe("auto");
  });

  test("house targets are recognised as an analytics reason", () => {
    const reasons = targetReasons({ target_reason: "house" } as Parameters<
      typeof targetReasons
    >[0]);
    expect(reasons).toEqual(["house"]);
  });

  test("every new key exists in all four locale files", () => {
    const keys = [
      "house.title",
      "house.disabled",
      "housecard.name",
      "housecard.description",
      "settings.house_title",
      "settings.intro.house",
      "analytics.reason.house",
    ];
    for (const [name, locale] of Object.entries({ en, de, fr, it })) {
      for (const key of keys) {
        expect(locale[key as keyof typeof en], `${name} is missing ${key}`).toBeTruthy();
      }
    }
  });

  test("the locale files stay key-identical", () => {
    const reference = Object.keys(en).sort();
    for (const [name, locale] of Object.entries({ de, fr, it })) {
      expect(Object.keys(locale).sort(), `${name} has a different key set`).toEqual(reference);
    }
  });
});
