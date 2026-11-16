/**
 * Core type definitions for RoomMind frontend.
 */

export type ClimateMode = "auto" | "heat_only" | "cool_only";

/** Central Plant action; rooms report the plant action that drives them. */
export type PlantAction = "idle" | "heating" | "cooling" | "circulating";

/** Kept for the existing room widgets: a room's mode is the plant action that covers it. */
export type RoomMode = PlantAction;

/** How a room's occupancy sensors are interpreted. */
export type OccupancyMode =
  "reactive_and_trend" | "schedule_assumed_when_on" | "follow_schedule" | "guardrail_only";

/** What an open exterior opening does to the whole-home plant. */
export type ExteriorOpenPolicy = "pause_unless_door_closed" | "pause_always" | "exclude_room_only";

/** Which sides the electricity price may shift. */
export type PriceShiftModes = "cool_only" | "both";

export type OverrideType = "boost" | "eco" | "custom";

export interface NotificationTarget {
  entity_id: string;
  person_entity: string;
  notify_when: "always" | "home_only";
}

export interface ScheduleEntry {
  entity_id: string;
}

export interface CoverScheduleEntry {
  entity_id: string;
  mode?: "force" | "gate";
}

export interface RoomLiveData {
  current_temp: number | null;
  current_humidity: number | null;
  target_temp: number | null;
  heat_target: number | null;
  cool_target: number | null;
  mode: PlantAction;
  heating_power: number; // 0-100
  device_setpoint: number | null; // Central thermostat's commanded setpoint
  override_active: boolean;
  override_type: OverrideType | null;
  override_heat: number | null;
  override_cool: number | null;
  override_until: number | null;
  override_suppressed: boolean;
  active_schedule_index: number;
  window_open: boolean;
  confidence: number | null;
  mpc_active: boolean;
  presence_away: boolean;
  occupancy_vacant?: boolean;
  /** Room Priority weight and occupancy probability from the plant optimization. */
  priority_weight?: number | null;
  occupancy_probability?: number | null;
  /** Minutes of sustained occupancy expected/observed; drives the assumed-presence modes. */
  occupancy_dwell?: number | null;
  /** Effective temperature (dry-bulb corrected for humidity), null without humidity. */
  effective_temp?: number | null;
  /** Closed interior door reduces the room's air exchange with the house. */
  interior_door_open?: boolean;
  /** An exterior window or door of the room is open. */
  exterior_window_open?: boolean;
  /** The plant is circulating air into the room right now. */
  fan_circulating?: boolean;
  /** Learned blower coupling rate into this room, null before it is learned. */
  beta_fan_rate?: number | null;
  /** Air quality readings (#267), null without sensor or reading. */
  co2?: number | null;
  voc?: number | null;
  air_quality_bad?: boolean;
  air_quality_reason?: "co2" | "voc" | null;
  /** RoomMind runs the room's ventilation entity for bad air. */
  ventilation_active?: boolean;
  mold_risk_level: "ok" | "warning" | "critical";
  mold_surface_rh: number | null;
  mold_surface_temp: number | null;
  mold_prevention_active: boolean;
  /** Applied electricity price shift per side (K), 0 when none. */
  price_shift_heat?: number;
  price_shift_cool?: number;
  price_level?: "cheap" | "normal" | "expensive" | null;
  /** Stricter target a running pre-heat (pre-cool) aims at and when it starts (Unix s). */
  preheat?: { mode: "heating" | "cooling"; target: number; at: number } | null;
  mold_prevention_delta: number;
  blind_position: number | null;
  cover_auto_paused: boolean;
  cover_override_until?: number | null;
  cover_forced_reason: string;
  cover_reason?: string;
  active_cover_schedule_index: number;
  learning_paused_reason: "outdoor_unavailable" | null;
  /** Model updates so far (one per ~3 min of learning). */
  n_observations?: number;
  /** `n_observations` when "learn faster" was last used, null if never (counts the fade). */
  boost_applied_at?: number | null;
  schedule_temp_warnings: ScheduleTempWarning[];
  /** Last 6 h of room temperature, `[unix_ts, temp, target]`, oldest first (dashboard cards). */
  recent_temps?: [number, number | null, number | null][];
}

/** A schedule block temperature the backend discarded as implausible (#395). */
export interface ScheduleTempWarning {
  day: string; // lowercase English weekday, e.g. "monday"
  from: string; // "HH:MM:SS"
  field: "temperature" | "heat_temperature" | "cool_temperature";
  value: string; // the offending raw value; may not be numeric
}

/** One shared forced-air plant: thermostat, guards and fallback band. */
export interface CentralPlantConfig {
  thermostat_entity: string;
  thermostat_area_id: string;
  nudge_delta_c: number;
  max_nudge_headroom_c: number;
  fallback_heat_c: number;
  fallback_cool_c: number;
  min_deadband_c: number;
  circulate_fan_mode: string;
  idle_fan_mode: string;
  min_run_minutes: number;
  min_off_minutes: number;
  min_fan_minutes: number;
  changeover_minutes: number;
  follow_thermostat_setpoint: boolean;
  thermostat_override_hours: number;
  exterior_open_policy: ExteriorOpenPolicy;
  humidity_comfort_enabled: boolean;
  dehumidify_rh_threshold: number;
  max_dehumidify_overcool_c: number;
  wet_coil_lockout_minutes: number;
  wet_coil_max_rh: number;
  price_shift_modes: PriceShiftModes;
}

/** Live status of the Central Plant as `roommind/rooms/list` reports it. */
export interface CentralPlantLive {
  configured: boolean;
  thermostat_entity: string;
  thermostat_area_id: string;
  /** What the whole-home optimizer chose. */
  action: PlantAction;
  commanded_action: PlantAction;
  /** What the equipment reports (or the command while it has no feedback). */
  display_action?: PlantAction;
  observed_hvac_action: string | null;
  fan_circulating: boolean;
  climate_mode?: ClimateMode;
  thermostat_temp: number | null;
  house_ref_temp: number | null;
  /** The bounded setpoints the actuator holds on the wall thermostat. */
  commanded_low: number | null;
  commanded_high: number | null;
  commanded_single: number | null;
  legal_actions?: PlantAction[];
  forced_action?: PlantAction | null;
  driving_rooms: string[];
  decision_reason: string;
  guard_active: boolean;
  guard_reason: string | null;
  guard_until: number | null;
  wet_coil_charge: number;
  wet_coil_locked: boolean;
  paused?: boolean;
  blocking_rooms?: string[];
  exterior_paused_rooms: string[];
  plant_override_active: boolean;
  plant_override_heat: number | null;
  plant_override_cool: number | null;
  plant_override_until: number | null;
  mpc_active?: boolean;
  preheat?: Record<string, { mode: "heating" | "cooling"; target: number; at: number }>;
  actions?: PlantAction[];
  outdoor_temp?: number | null;
  outdoor_source?: string;
  startup_waiting?: boolean;
  all_force_off?: boolean;
  any_override?: boolean;
  price_level?: "cheap" | "normal" | "expensive" | null;
  /** The stored central-plant configuration block. */
  plant?: CentralPlantConfig;
}

export type ConflictResolution =
  "heating_priority" | "cooling_priority" | "majority" | "outdoor_temp";

/** A stored room whose Home Assistant area was deleted; it rests until the area is back (#494). */
export interface RoomWithoutArea {
  area_id: string;
  display_name: string;
}

export interface CompressorGroup {
  id: string;
  name: string;
  members: string[];
  min_run_minutes: number;
  min_off_minutes: number;
  master_entity: string;
  conflict_resolution: ConflictResolution;
  action_script: string;
  enforce_uniform_mode: boolean;
}

export interface RoomConfig {
  area_id: string;
  temperature_sensor: string;
  humidity_sensor: string;
  occupancy_sensors?: string[];
  occupancy_eco_enabled?: boolean;
  occupancy_eco_delay?: number;
  /** How the occupancy sensors are interpreted; replaces the eco-only flag. */
  occupancy_mode?: OccupancyMode;
  /** Minutes of sustained occupancy before the room counts as occupied. */
  occupancy_dwell_minutes?: number;
  co2_sensor?: string;
  /** ppm; good again below 80 % of it. */
  co2_threshold?: number;
  voc_sensor?: string;
  /** In the VOC sensor's unit; null = shown, not judged. */
  voc_threshold?: number | null;
  /** fan.*, switch.* or input_boolean.* RoomMind switches on for bad air. */
  ventilation_entity?: string;
  preheat_enabled?: boolean;
  /** K the schedule target may move with the electricity price, 0 = off. */
  price_shift?: number;
  /** Exterior windows and doors; the canonical field, mirrored as `window_sensors`. */
  exterior_window_sensors?: string[];
  /** Legacy alias of `exterior_window_sensors`, kept for older stored rooms. */
  window_sensors: string[];
  window_open_delay: number;
  window_close_delay: number;
  /** Interior doors that gate the room's coupling to the rest of the house. */
  interior_door_sensors?: string[];
  interior_door_delay?: number;
  /** Scaling of the door coupling when the interior door is closed (0..1). */
  door_closed_mix_factor?: number;
  climate_mode: ClimateMode;
  schedules: ScheduleEntry[];
  schedule_selector_entity: string;
  comfort_temp?: number;
  eco_temp?: number;
  comfort_heat: number;
  comfort_cool: number;
  eco_heat: number;
  eco_cool: number;
  override_heat?: number | null;
  override_cool?: number | null;
  override_until?: number | null;
  override_type?: OverrideType | null;
  presence_persons?: string[];
  display_name?: string;
  covers?: string[];
  /** Blinds RoomMind may only position observations for; it never drives them. */
  read_only_covers?: string[];
  covers_auto_enabled?: boolean;
  covers_deploy_threshold?: number;
  covers_min_position?: number;
  covers_override_minutes?: number;
  cover_schedules?: CoverScheduleEntry[];
  cover_schedule_selector_entity?: string;
  cover_orientations?: Record<string, number>;
  covers_outdoor_min_temp?: number | null;
  covers_night_close?: boolean;
  covers_night_position?: number;
  covers_night_close_elevation?: number;
  covers_night_close_offset_minutes?: number;
  covers_snap_deploy?: boolean;
  covers_window_hold?: boolean;
  cover_min_positions?: Record<string, number>;
  ignore_presence?: boolean;
  is_outdoor?: boolean;
  climate_control_enabled?: boolean;
  /** False keeps the room on its own comfort/eco/schedule instead of the house layer. */
  follow_house?: boolean;
  live?: RoomLiveData;
}

/** What RoomMind reads from the price source (rooms/list ``price_status``). */
export interface PriceStatus {
  entity_id: string;
  readable: boolean;
  kind: "forecast" | "history" | "switch" | null;
  price: number | null;
  level: "cheap" | "normal" | "expensive" | null;
  forecast_hours: number;
  history_hours: number;
  history_needed: number;
}

/**
 * The household base layer every following room inherits: a comfort pair, an eco
 * pair, a climate mode, an optional schedule and a price shift.
 */
export interface HouseConfig {
  enabled: boolean;
  heat_c: number | null;
  cool_c: number | null;
  eco_heat_c: number;
  eco_cool_c: number;
  climate_mode: ClimateMode;
  schedules: ScheduleEntry[];
  schedule_selector_entity: string;
  preheat_enabled: boolean;
  price_shift: number;
  override_heat: number | null;
  override_cool: number | null;
  override_until: number | null;
  override_type: OverrideType | null;
}

/**
 * The house layer as the coordinator publishes it: the resolved settings plus what
 * is in force right now. `band` is the target following rooms inherit; `source`
 * says which rung of the house layer produced it.
 */
export interface HouseLive extends HouseConfig {
  active: boolean;
  band: { heat_c: number | null; cool_c: number | null };
  source: "override" | "schedule" | "target" | null;
  override_active: boolean;
  schedule_entity: string;
  schedule_name: string;
  schedule_block: { from: string; to: string } | null;
  next_change: number | null;
  following_rooms: number;
}

export interface GlobalSettings {
  outdoor_temp_sensor: string;
  price_entity?: string;
  outdoor_humidity_sensor: string;
  outdoor_cooling_min?: number;
  outdoor_heating_max?: number;
  control_mode?: "mpc" | "bangbang";
  comfort_weight?: number;
  weather_entity?: string;
  outdoor_unavailable_notify?: boolean;
  climate_control_active?: boolean;
  learning_disabled_rooms?: string[];
  hidden_rooms?: string[];
  prediction_enabled?: boolean;
  vacation_temp?: number;
  vacation_until?: number | null;
  presence_enabled?: boolean;
  presence_persons?: string[];
  presence_away_action?: "eco" | "off";
  presence_clears_override?: boolean;
  schedule_off_action?: "eco" | "off";
  mold_detection_enabled?: boolean;
  mold_humidity_threshold?: number;
  mold_f_rsi?: number;
  mold_sustained_minutes?: number;
  mold_notification_cooldown?: number;
  mold_notifications_enabled?: boolean;
  mold_notification_targets?: NotificationTarget[];
  mold_prevention_enabled?: boolean;
  mold_prevention_intensity?: "light" | "medium" | "strong";
  mold_prevention_notify_enabled?: boolean;
  mold_prevention_notify_targets?: NotificationTarget[];
  /** The single shared forced-air plant (thermostat, guards, fallback band). */
  central_plant?: CentralPlantConfig;
  /** The household base layer; missing/inert until the occupant sets a house target. */
  house?: HouseConfig;
  room_order?: string[];
  group_by_floor?: boolean;
  boost_applied_at?: Record<string, number>;
}

// HA types for panel integration
export interface HassConnection {
  addEventListener(event: string, callback: () => void): void;
  removeEventListener(event: string, callback: () => void): void;
}

export interface HomeAssistant {
  callWS: <T>(msg: Record<string, unknown>) => Promise<T>;
  callService: (domain: string, service: string, data?: Record<string, unknown>) => Promise<void>;
  states: Record<string, HassEntity>;
  areas: Record<string, HassArea>;
  floors?: Record<string, HassFloor>;
  entities: Record<string, HassEntityRegistryEntry>;
  devices: Record<string, HassDeviceRegistryEntry>;
  language: string;
  config: { unit_system: { temperature: string }; time_zone?: string };
  user?: { is_admin: boolean; name?: string };
  connection?: HassConnection;
  /** Translated state as Home Assistant shows it (e.g. "Heizen"); missing in old releases. */
  formatEntityState?: (stateObj: HassEntity) => string;
  formatEntityAttributeValue?: (stateObj: HassEntity, attribute: string, value?: unknown) => string;
  /** HA's own light/dark choice (may differ from the OS). */
  themes?: { darkMode?: boolean };
}

export interface HassArea {
  area_id: string;
  name: string;
  picture: string | null;
  floor_id: string | null;
  icon?: string | null;
}

export interface HassFloor {
  floor_id: string;
  name: string;
  level: number | null;
}

export interface HassEntityRegistryEntry {
  entity_id: string;
  area_id: string | null;
  device_id: string | null;
  platform: string;
}

export interface HassDeviceRegistryEntry {
  id: string;
  area_id: string | null;
  name?: string | null;
  name_by_user?: string | null;
}

export interface HassEntity {
  entity_id: string;
  state: string;
  attributes: Record<string, unknown>;
  last_updated?: string;
}

export interface AnalyticsDataPoint {
  ts: number;
  room_temp: number | null;
  outdoor_temp: number | null;
  target_temp: number | null;
  /** What the room did at this sample; forecast points carry `forecast`, older ones "". */
  mode: PlantAction | "forecast" | "";
  predicted_temp: number | null;
  window_open: boolean;
  /** Share of the sample the room was heating, missing on forecast points. */
  heating_power?: number | null;
  /** Solar irradiance at this sample, missing on forecast points. */
  solar_irradiance?: number | null;
  blind_position?: number | null;
  cover_reason?: string;
  device_setpoint?: number | null;
  humidity?: number | null;
  /** Room CO2 in ppm (#267). */
  co2?: number | null;
  heat_target?: number | null;
  cool_target?: number | null;
  /** What set the target: "" schedule, else a TargetReason. */
  target_reason?: string;
  mold_boost?: number | null;
  price?: number | null;
  price_level?: string;
  /** Heat-side shift the electricity price applied (cool side moves the other way). */
  price_shift?: number | null;
  /** Humidity-corrected room temperature from the Central HVAC roll-out. */
  effective_temp?: number | null;
  /** Room Priority weight used in the joint plant plan. */
  priority_weight?: number | null;
  /** Interior door open at this sample. */
  door_open?: boolean;
  /** The plant was circulating air at this sample. */
  fan_circulating?: boolean;
  /** House reference temperature from the joint whole-home model. */
  house_temp?: number | null;
}

export interface RunOvershoot {
  runs: number;
  mean: number;
  max: number;
}

export interface AnalyticsStats {
  hours: number;
  heating_hours: number;
  cooling_hours: number;
  window_hours: number;
  heating_starts: number;
  cooling_starts: number;
  /** House-wide circulation hours and starts (Central HVAC). */
  circulating_hours?: number;
  circulate_starts?: number;
  /** How far past the target the room went after the runs, null without runs. */
  heat_overshoot?: RunOvershoot | null;
  /** Run hours by price level, null without a price source. */
  price?: {
    heating_hours: Partial<Record<"cheap" | "normal" | "expensive", number>>;
    cooling_hours: Partial<Record<"cheap" | "normal" | "expensive", number>>;
    shifted_hours: number;
  } | null;
  cool_overshoot?: RunOvershoot | null;
  comfort: {
    below_pct: number;
    within_pct: number;
    above_pct: number;
    mean_deviation: number;
    /** Mean distance while off target, null when the room never was. */
    miss_deviation: number | null;
  } | null;
}

export interface RhythmDay {
  date: string;
  heat: Array<number | null>;
  cool: Array<number | null>;
  /** Share of the hour the plant was circulating air, absent from older payloads. */
  circulate?: Array<number | null>;
  window: Array<number | null>;
  deviation: Array<number | null>;
  /** Share of the judged time too cold / too warm, same tolerance as the period figures. */
  below: Array<number | null>;
  above: Array<number | null>;
  temp: Array<number | null>;
  humidity: Array<number | null>;
  outdoor: Array<number | null>;
  /** The day's judged share on target, null without judged samples. */
  within: number | null;
}

export interface AnalyticsRhythm {
  days: RhythmDay[];
}

export interface ModelDay {
  /** Local date, YYYY-MM-DD. */
  date: string;
  confidence: number;
  mpc_active: boolean;
  /** Mean absolute error of the ~3-min forecast that day (K), null without measurements. */
  error: number | null;
  error_steps: number;
}

export interface AnalyticsData {
  detail: AnalyticsDataPoint[];
  history: AnalyticsDataPoint[];
  forecast?: AnalyticsDataPoint[];
  stats?: AnalyticsStats;
  rhythm?: AnalyticsRhythm | null;
  /** Unit of the price source (ct/kWh, EUR/kWh, ...), empty for a switch or none. */
  price_unit?: string;
  model: {
    confidence: number;
    /** One entry per local day, oldest first. */
    daily_history?: ModelDay[];
    model: {
      C: number;
      U: number;
      Q_heat: number;
      Q_cool: number;
      Q_solar: number;
      Q_occupancy: number;
      Q_fast?: number;
    };
    n_samples: number;
    n_observations: number;
    /** Unix time of the model's last learning step, null before the first or from an older store */
    last_update_at?: number | null;
    n_heating: number;
    n_cooling: number;
    applicable_modes: string[];
    mpc_active: boolean;
    sigma_e: number;
    prediction_std_idle: number;
    prediction_std_heating: number;
    has_occupancy_sensors: boolean;
    /** Sun / occupancy were in the data (variance below its start), else the gain is the start guess */
    solar_learned?: boolean;
    occupancy_learned?: boolean;
    solar_gain_now?: number;
    solar_gain_today_max?: number;
    /** Learned blower-to-room mixing rate and whether it is learned yet. */
    fan_mix_rate?: number;
    fan_mix_learned?: boolean;
    /** Learned house-to-room coupling and whether it is learned yet. */
    house_coupling?: number;
    house_coupling_learned?: boolean;
    /** 24 h occupancy probability curve (one value per hour, oldest first). */
    occupancy_trend_24h?: number[];
    /** The model one hour ahead on this range (backend model_backtest), null without a model. */
    backtest?: ModelBacktest | null;
  };
}

export interface BacktestFigures {
  /** Hours (overlapping, one per recorded point) that went into the figure. */
  n: number;
  /** Mean absolute deviation after the horizon (°C), null below the minimum of hours. */
  mae: number | null;
  /** Mean signed deviation: model minus measured, negative = room warmer than the model says. */
  bias: number | null;
}

export interface ModelBacktest {
  horizon_minutes: number;
  idle: BacktestFigures;
  /** Hours with heating or cooling. */
  active: BacktestFigures;
}

export type TimeRange = "12h" | "24h" | "2d" | "7d" | "14d" | "30d" | "90d";
