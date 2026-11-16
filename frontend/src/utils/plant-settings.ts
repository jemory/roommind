/**
 * The Central Plant settings block: defaults, flat page fields, bounds for its number fields
 * and the mapping between page fields and the stored `central_plant` block. Pure data and
 * functions so the mapping is unit-tested without a DOM; the settings components only render.
 */
import type { CentralPlantConfig } from "../types";

/** Defaults of the Central Plant block; mirrors DEFAULT_CENTRAL_PLANT_SETTINGS. */
export const DEFAULT_CENTRAL_PLANT: CentralPlantConfig = {
  thermostat_entity: "",
  thermostat_area_id: "",
  nudge_delta_c: 1.2,
  max_nudge_headroom_c: 2.5,
  fallback_heat_c: 16.5,
  fallback_cool_c: 26.5,
  min_deadband_c: 1.7,
  circulate_fan_mode: "on",
  idle_fan_mode: "auto",
  min_run_minutes: 10,
  min_off_minutes: 5,
  min_fan_minutes: 10,
  changeover_minutes: 30,
  follow_thermostat_setpoint: true,
  thermostat_override_hours: 2,
  exterior_open_policy: "pause_unless_door_closed",
  humidity_comfort_enabled: true,
  dehumidify_rh_threshold: 60,
  max_dehumidify_overcool_c: 0.8,
  wet_coil_lockout_minutes: 25,
  wet_coil_max_rh: 52,
  price_shift_modes: "cool_only",
};

/** Flat page fields of the central_plant block, in the order the settings page edits them. */
export const PLANT_FIELDS = [
  "thermostatEntity",
  "thermostatAreaId",
  "nudgeDeltaC",
  "maxNudgeHeadroomC",
  "fallbackHeatC",
  "fallbackCoolC",
  "minDeadbandC",
  "circulateFanMode",
  "idleFanMode",
  "minRunMinutes",
  "minOffMinutes",
  "minFanMinutes",
  "changeoverMinutes",
  "followThermostatSetpoint",
  "thermostatOverrideHours",
  "exteriorOpenPolicy",
  "humidityComfortEnabled",
  "dehumidifyRhThreshold",
  "maxDehumidifyOvercoolC",
  "wetCoilLockoutMinutes",
  "wetCoilMaxRh",
  "priceShiftModes",
] as const;

/**
 * Bounds of the Central Plant number fields, in °C for temperatures and temperature
 * differences (rs-number-field converts them for display). Mirrors the vol.Range bounds of
 * CENTRAL_PLANT_SCHEMA in websocket_api.py: the page must never offer or send a value the
 * backend would reject.
 */
export const PLANT_LIMITS = {
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
} as const;

/** Stored `central_plant` value -> flat page fields, with the defaults filled in. */
export function readPlantFields(value: unknown): Record<string, unknown> {
  const plant = {
    ...DEFAULT_CENTRAL_PLANT,
    ...((value as CentralPlantConfig | undefined) ?? {}),
  };
  return {
    thermostatEntity: plant.thermostat_entity,
    thermostatAreaId: plant.thermostat_area_id,
    nudgeDeltaC: plant.nudge_delta_c,
    maxNudgeHeadroomC: plant.max_nudge_headroom_c,
    fallbackHeatC: plant.fallback_heat_c,
    fallbackCoolC: plant.fallback_cool_c,
    minDeadbandC: plant.min_deadband_c,
    circulateFanMode: plant.circulate_fan_mode,
    idleFanMode: plant.idle_fan_mode,
    minRunMinutes: plant.min_run_minutes,
    minOffMinutes: plant.min_off_minutes,
    minFanMinutes: plant.min_fan_minutes,
    changeoverMinutes: plant.changeover_minutes,
    followThermostatSetpoint: plant.follow_thermostat_setpoint,
    thermostatOverrideHours: plant.thermostat_override_hours,
    exteriorOpenPolicy: plant.exterior_open_policy,
    humidityComfortEnabled: plant.humidity_comfort_enabled,
    dehumidifyRhThreshold: plant.dehumidify_rh_threshold,
    maxDehumidifyOvercoolC: plant.max_dehumidify_overcool_c,
    wetCoilLockoutMinutes: plant.wet_coil_lockout_minutes,
    wetCoilMaxRh: plant.wet_coil_max_rh,
    priceShiftModes: plant.price_shift_modes,
  };
}

/** Flat page fields -> the stored `central_plant` block the backend merges. */
export function writePlantFields(v: Record<string, unknown>): Record<string, unknown> {
  return {
    thermostat_entity: v.thermostatEntity,
    thermostat_area_id: v.thermostatAreaId,
    nudge_delta_c: v.nudgeDeltaC,
    max_nudge_headroom_c: v.maxNudgeHeadroomC,
    fallback_heat_c: v.fallbackHeatC,
    fallback_cool_c: v.fallbackCoolC,
    min_deadband_c: v.minDeadbandC,
    circulate_fan_mode: v.circulateFanMode,
    idle_fan_mode: v.idleFanMode,
    min_run_minutes: v.minRunMinutes,
    min_off_minutes: v.minOffMinutes,
    min_fan_minutes: v.minFanMinutes,
    changeover_minutes: v.changeoverMinutes,
    follow_thermostat_setpoint: v.followThermostatSetpoint,
    thermostat_override_hours: v.thermostatOverrideHours,
    exterior_open_policy: v.exteriorOpenPolicy,
    humidity_comfort_enabled: v.humidityComfortEnabled,
    dehumidify_rh_threshold: v.dehumidifyRhThreshold,
    max_dehumidify_overcool_c: v.maxDehumidifyOvercoolC,
    wet_coil_lockout_minutes: v.wetCoilLockoutMinutes,
    wet_coil_max_rh: v.wetCoilMaxRh,
    price_shift_modes: v.priceShiftModes,
  };
}
