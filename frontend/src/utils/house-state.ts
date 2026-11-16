/**
 * Shared utilities for reading and formatting the RoomMind house layer.
 *
 * The house layer is the base target every following room inherits: one comfort
 * pair, one eco pair, an optional schedule and an optional temporary hold. These
 * helpers are pure (the live block is the only input) so the dashboard card, the
 * panel hero and the tests agree on what "in force" means.
 */
import type { ClimateMode, HomeAssistant, HouseLive } from "../types";
import { localize, type TranslationKey } from "./localize";
import { formatRemaining } from "./format";
import { formatTempLocale, tempUnit } from "./temperature";

/** Which house target side is being served, in °C; null where the side is not set. */
export interface HouseBand {
  low: number | null;
  high: number | null;
}

/** The house target in force right now (override > schedule block > house pair). */
export function houseTarget(house: HouseLive | null | undefined): HouseBand {
  const band = house?.band;
  return { low: band?.heat_c ?? null, high: band?.cool_c ?? null };
}

/** True while a temporary hold set from the house climate entity (or its card) is active. */
export function houseOverrideActive(house: HouseLive | null | undefined): boolean {
  return Boolean(house?.override_active);
}

/** The preset the house override corresponds to; "none" means the house layer is serving its schedule/target. */
export type HousePreset = "comfort" | "eco" | "manual" | "none";

export function housePreset(house: HouseLive | null | undefined): HousePreset {
  if (!houseOverrideActive(house)) return "none";
  switch (house?.override_type) {
    case "boost":
      return "comfort";
    case "eco":
      return "eco";
    case "custom":
      return "manual";
    default:
      return "manual";
  }
}

/** House climate mode choices, reusing the room mode labels. */
export const HOUSE_MODE_OPTIONS: { value: ClimateMode; key: TranslationKey; icon: string }[] = [
  { value: "auto", key: "mode.auto", icon: "mdi:autorenew" },
  { value: "heat_only", key: "mode.heat_only", icon: "mdi:fire" },
  { value: "cool_only", key: "mode.cool_only", icon: "mdi:snowflake" },
];

export function houseModeLabel(house: HouseLive | null | undefined, language: string): string {
  const mode = house?.climate_mode ?? "auto";
  const option = HOUSE_MODE_OPTIONS.find((entry) => entry.value === mode) ?? HOUSE_MODE_OPTIONS[0];
  return localize(option.key, language);
}

/** What is setting the house target right now; the disabled state on an upgraded install. */
export function houseSourceLabel(house: HouseLive | null | undefined, language: string): string {
  if (!house || !house.active) return localize("house.disabled", language);
  switch (house.source) {
    case "override":
      return localize("housecard.source.override", language);
    case "schedule":
      return localize("housecard.source.schedule", language);
    default:
      return localize("housecard.source.target", language);
  }
}

export function followingRoomsLabel(count: number, language: string): string {
  const key = count === 1 ? "house.following_count" : "house.following_count_plural";
  return localize(key, language, { count });
}

/** "70–74 °F", or a single side when the house target is one-sided. */
export function formatHouseTarget(
  house: HouseLive | null | undefined,
  hass: HomeAssistant,
): string {
  const { low, high } = houseTarget(house);
  const unit = tempUnit(hass);
  if (low === null && high === null) return "—";
  if (low !== null && high !== null) {
    if (Math.abs(high - low) < 0.05) return `${formatTempLocale(low, hass, 0, 0)} ${unit}`;
    return `${formatTempLocale(low, hass, 0, 0)}–${formatTempLocale(high, hass, 0, 0)} ${unit}`;
  }
  return `${formatTempLocale((low ?? high) as number, hass, 0, 0)} ${unit}`;
}

/** "Held until in 2 hours" while a house hold is active, else null. */
export function houseHoldLabel(
  house: HouseLive | null | undefined,
  language: string,
): string | null {
  if (!houseOverrideActive(house) || house?.override_until == null) return null;
  return localize("housecard.hold_until", language, {
    when: formatRemaining(house.override_until, language),
  });
}
