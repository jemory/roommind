/**
 * Defaults, bounds and validation for the house layer settings.
 *
 * Mirrors the backend's `house` block (const.py DEFAULT_HOUSE_SETTINGS and the
 * voluptuous bounds in websocket_api.py) the way utils/plant-settings.ts mirrors
 * the Central Plant schema, so the panel can validate before saving and the
 * frontend and backend cannot drift.
 */
import type { ClimateMode, HouseConfig, ScheduleEntry } from "../types";

/** Plausibility bounds for house target temperatures, °C (MIN_TARGET_TEMP/MAX_TARGET_TEMP). */
export const HOUSE_TEMP_BOUNDS = { min: 5, max: 35 } as const;
/** Largest price shift the house target may move with, K (MAX_PRICE_SHIFT_K). */
export const HOUSE_PRICE_SHIFT_MAX = 2.0;

export type HouseDefaults = Omit<
  HouseConfig,
  "override_heat" | "override_cool" | "override_until" | "override_type"
>;

/** What a fresh install seeds; an upgraded install starts with `enabled: false`. */
export const HOUSE_DEFAULTS: HouseDefaults = {
  enabled: true,
  heat_c: 21,
  cool_c: 24,
  eco_heat_c: 17,
  eco_cool_c: 27,
  climate_mode: "auto",
  schedules: [] as ScheduleEntry[],
  schedule_selector_entity: "",
  preheat_enabled: true,
  price_shift: 0,
};

export type HouseValidationError = "band_order" | "eco_band_order" | "temp_range" | "price_range";

function inRange(value: number): boolean {
  return value >= HOUSE_TEMP_BOUNDS.min && value <= HOUSE_TEMP_BOUNDS.max;
}

/**
 * Validate the house layer before saving. Returns a stable error code (components
 * localize it) or null when the settings are usable.
 *
 * The comfort band must keep cooling above heating; the eco band is a setback pair
 * and only has to stay ordered; a one-sided comfort pair is valid (a house that
 * only heats or only cools), which is why the order check is skipped when a side
 * is null.
 */
export function validateHouseSettings(
  config: Pick<HouseConfig, "heat_c" | "cool_c" | "eco_heat_c" | "eco_cool_c" | "price_shift">,
): HouseValidationError | null {
  const { heat_c, cool_c, eco_heat_c, eco_cool_c, price_shift } = config;
  if (heat_c !== null && cool_c !== null && cool_c <= heat_c) return "band_order";
  if (eco_cool_c < eco_heat_c) return "eco_band_order";
  const temps = [heat_c, cool_c, eco_heat_c, eco_cool_c].filter(
    (value): value is number => value !== null,
  );
  if (temps.some((value) => !inRange(value))) return "temp_range";
  if (price_shift < 0 || price_shift > HOUSE_PRICE_SHIFT_MAX) return "price_range";
  return null;
}

/** The house climate mode a room inherits when it follows the house. */
export function houseModeOrDefault(
  config: { climate_mode?: ClimateMode } | null | undefined,
): ClimateMode {
  return config?.climate_mode ?? "auto";
}
