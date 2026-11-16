/**
 * Shared thermostat/entity helpers for the Central Plant.
 *
 * The Central HVAC schema owns the hardware through a single Central Thermostat, so the
 * per-room device model (TRVs and ACs) is gone; what remains here is the fan-mode naming
 * shared by the settings pages.
 */

import type { HassEntity, HomeAssistant } from "../types";

export interface FanModeOption {
  value: string;
  label: string;
}

/**
 * A fan speed as Home Assistant names it ("low" → "Niedrig"). Without a device the climate
 * domain's names apply; speeds Home Assistant does not know stay as the device reports them.
 */
export function fanModeLabel(hass: HomeAssistant, mode: string, stateObj?: HassEntity): string {
  const st = stateObj ?? { entity_id: "climate.roommind_fan_mode", state: "", attributes: {} };
  return hass.formatEntityAttributeValue?.(st, "fan_mode", mode) || mode;
}

/**
 * The fan modes a climate entity reports, translated, in the order the integration lists them.
 * Falls back to the shared `off`/`low`/`medium`/`high` guesses when the entity has no
 * `fan_modes` attribute yet (the user may not have a thermostat configured).
 */
export function thermostatFanModes(hass: HomeAssistant, entityId: string): FanModeOption[] {
  const st = entityId ? hass.states[entityId] : undefined;
  const raw = (st?.attributes?.fan_modes ?? []) as string[];
  const modes = raw.length ? raw : ["off", "low", "medium", "high", "auto"];
  const seen = new Set<string>();
  const options: FanModeOption[] = [];
  for (const mode of modes) {
    if (seen.has(mode)) continue;
    seen.add(mode);
    options.push({ value: mode, label: fanModeLabel(hass, mode, st) });
  }
  return options;
}
