import { LitElement, html, css, nothing } from "lit";
import { customElement, property, state } from "lit/decorators.js";
import type {
  CentralPlantLive,
  HomeAssistant,
  HassArea,
  RoomConfig,
  ClimateMode,
  OccupancyMode,
  ScheduleEntry,
  CoverScheduleEntry,
  HouseLive,
} from "../types";
import "./room-detail/rs-room-hero";
import "./rs-schedule-settings";
import type { ScheduleCardPart } from "./rs-schedule-settings";
import "./room-detail/rs-target-temps";
import "./room-detail/rs-price-shift";
import "./rs-sensor-section";
import "./rs-override-section";
import "./rs-presence-section";
import "./rs-covers-section";
import "./shared/rs-edit-dialog";
import "./shared/rs-confirm-dialog";
import "./shared/rs-list-card";
import "./shared/rs-list-row";
import "./shared/rs-segmented";
import {
  coversHelp,
  editDialogHelpStyles,
  presenceHelp,
  priceHelp,
  scheduleHelp,
  sensorsHelp,
  temperaturesHelp,
} from "./room-detail/edit-dialog-help";
import { localize, type TranslationKey } from "../utils/localize";
import { HOUSE_DEFAULTS } from "../utils/house-settings";
import { houseModeLabel } from "../utils/house-state";
import { formatTempLocale } from "../utils/temperature";
import { errorText } from "../utils/errors";
import { showToast } from "../utils/toast";
import { fireSaveStatus } from "../utils/events";
import {
  DEFAULT_CO2_THRESHOLD,
  DEFAULT_DOOR_CLOSED_MIX_FACTOR,
  DEFAULT_INTERIOR_DOOR_DELAY,
  DEFAULT_OCCUPANCY_DWELL_MINUTES,
  DEFAULT_OCCUPANCY_ECO_DELAY,
} from "../utils/constants";
import { NO_OVERRIDE, type EffectiveOverride } from "../utils/override";
import { tokens } from "../styles/tokens";

type EditableSection = ScheduleCardPart | "sensors" | "presence" | "covers";

type RoomFields = Record<string, unknown>;

/** One stored room field: the state property it is read into and how it goes out again. */
interface RoomFieldSync {
  prop: string;
  read: (c: RoomConfig | null) => unknown;
  toState?: (value: unknown) => unknown;
  toPayload?: (state: unknown) => unknown;
}

const plain = (prop: string, read: (c: RoomConfig | null) => unknown): RoomFieldSync => ({
  prop,
  read,
});
const asSet = (prop: string, read: (c: RoomConfig | null) => string[]): RoomFieldSync => ({
  prop,
  read,
  toState: (value) => new Set(value as string[]),
  toPayload: (state) => [...(state as Set<string>)],
});

/**
 * Every room field the page edits. A poll adopts a field changed elsewhere (dashboard card,
 * room climate entity, a second tab) unless it was changed here too, and a save sends only the
 * fields changed here, so neither writes a stale copy back.
 */
const ROOM_FIELDS: [string, RoomFieldSync][] = [
  ["temperature_sensor", plain("_selectedTempSensor", (c) => c?.temperature_sensor ?? "")],
  ["humidity_sensor", plain("_selectedHumiditySensor", (c) => c?.humidity_sensor ?? "")],
  ["occupancy_sensors", asSet("_selectedOccupancySensors", (c) => c?.occupancy_sensors ?? [])],
  ["occupancy_mode", plain("_occupancyMode", (c) => c?.occupancy_mode ?? "reactive_and_trend")],
  [
    "occupancy_dwell_minutes",
    plain(
      "_occupancyDwellMinutes",
      (c) => c?.occupancy_dwell_minutes ?? DEFAULT_OCCUPANCY_DWELL_MINUTES,
    ),
  ],
  [
    "occupancy_eco_delay",
    plain("_occupancyEcoDelay", (c) => c?.occupancy_eco_delay ?? DEFAULT_OCCUPANCY_ECO_DELAY),
  ],
  ["co2_sensor", plain("_co2Sensor", (c) => c?.co2_sensor ?? "")],
  ["co2_threshold", plain("_co2Threshold", (c) => c?.co2_threshold ?? DEFAULT_CO2_THRESHOLD)],
  ["voc_sensor", plain("_vocSensor", (c) => c?.voc_sensor ?? "")],
  ["voc_threshold", plain("_vocThreshold", (c) => c?.voc_threshold ?? null)],
  ["ventilation_entity", plain("_ventilationEntity", (c) => c?.ventilation_entity ?? "")],
  ["preheat_enabled", plain("_preheatEnabled", (c) => c?.preheat_enabled ?? true)],
  ["price_shift", plain("_priceShift", (c) => c?.price_shift ?? 0)],
  [
    "exterior_window_sensors",
    asSet("_selectedWindowSensors", (c) => c?.exterior_window_sensors ?? c?.window_sensors ?? []),
  ],
  ["window_open_delay", plain("_windowOpenDelay", (c) => c?.window_open_delay ?? 0)],
  ["window_close_delay", plain("_windowCloseDelay", (c) => c?.window_close_delay ?? 0)],
  [
    "interior_door_sensors",
    asSet("_selectedInteriorDoorSensors", (c) => c?.interior_door_sensors ?? []),
  ],
  [
    "interior_door_delay",
    plain("_interiorDoorDelay", (c) => c?.interior_door_delay ?? DEFAULT_INTERIOR_DOOR_DELAY),
  ],
  [
    "door_closed_mix_factor",
    plain(
      "_doorClosedMixFactor",
      (c) => c?.door_closed_mix_factor ?? DEFAULT_DOOR_CLOSED_MIX_FACTOR,
    ),
  ],
  ["climate_mode", plain("_climateMode", (c) => c?.climate_mode ?? "auto")],
  ["follow_house", plain("_followHouse", (c) => c?.follow_house ?? false)],
  ["schedules", plain("_schedules", (c) => c?.schedules ?? [])],
  [
    "schedule_selector_entity",
    plain("_scheduleSelectorEntity", (c) => c?.schedule_selector_entity ?? ""),
  ],
  ["comfort_heat", plain("_comfortHeat", (c) => c?.comfort_heat ?? c?.comfort_temp ?? 21.0)],
  ["comfort_cool", plain("_comfortCool", (c) => c?.comfort_cool ?? 24.0)],
  ["eco_heat", plain("_ecoHeat", (c) => c?.eco_heat ?? c?.eco_temp ?? 17.0)],
  ["eco_cool", plain("_ecoCool", (c) => c?.eco_cool ?? 27.0)],
  [
    "presence_persons",
    {
      ...plain("_selectedPresencePersons", (c) => c?.presence_persons ?? []),
      toPayload: (state) => (state as string[]).filter((p) => p),
    },
  ],
  ["display_name", plain("_displayName", (c) => c?.display_name ?? "")],
  ["covers", asSet("_selectedCovers", (c) => c?.covers ?? [])],
  ["read_only_covers", asSet("_selectedReadOnlyCovers", (c) => c?.read_only_covers ?? [])],
  [
    "climate_control_enabled",
    plain("_climateControlEnabled", (c) => c?.climate_control_enabled ?? true),
  ],
  ["covers_auto_enabled", plain("_coversAutoEnabled", (c) => c?.covers_auto_enabled ?? false)],
  [
    "covers_deploy_threshold",
    plain("_coversDeployThreshold", (c) => c?.covers_deploy_threshold ?? 1.5),
  ],
  ["covers_min_position", plain("_coversMinPosition", (c) => c?.covers_min_position ?? 0)],
  [
    "covers_override_minutes",
    plain("_coversOverrideMinutes", (c) => c?.covers_override_minutes ?? 60),
  ],
  ["cover_schedules", plain("_coverSchedules", (c) => c?.cover_schedules ?? [])],
  [
    "cover_schedule_selector_entity",
    plain("_coverScheduleSelectorEntity", (c) => c?.cover_schedule_selector_entity ?? ""),
  ],
  ["covers_night_close", plain("_coversNightClose", (c) => c?.covers_night_close ?? false)],
  ["covers_night_position", plain("_coversNightPosition", (c) => c?.covers_night_position ?? 0)],
  ["covers_snap_deploy", plain("_coversSnapDeploy", (c) => c?.covers_snap_deploy ?? false)],
  ["covers_window_hold", plain("_coversWindowHold", (c) => c?.covers_window_hold ?? false)],
  ["cover_orientations", plain("_coverOrientations", (c) => c?.cover_orientations ?? {})],
  [
    "covers_night_close_elevation",
    plain("_coversNightCloseElevation", (c) => c?.covers_night_close_elevation ?? 0),
  ],
  [
    "covers_night_close_offset_minutes",
    plain("_coversNightCloseOffsetMinutes", (c) => c?.covers_night_close_offset_minutes ?? 0),
  ],
  [
    "covers_outdoor_min_temp",
    plain("_coversOutdoorMinTemp", (c) => c?.covers_outdoor_min_temp ?? 10),
  ],
  ["cover_min_positions", plain("_coverMinPositions", (c) => c?.cover_min_positions ?? {})],
  ["ignore_presence", plain("_ignorePresence", (c) => c?.ignore_presence ?? false)],
  ["is_outdoor", plain("_isOutdoor", (c) => c?.is_outdoor ?? false)],
];

const sameValue = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

/**
 * How long a field saved here ignores polls that still show the old value. Covers a poll read
 * before the save; the backend may also store a value differently, which is taken over after
 * this time.
 */
const UNCONFIRMED_MS = 15_000;

@customElement("rs-room-detail")
export class RsRoomDetail extends LitElement {
  @property({ attribute: false }) public area!: HassArea;
  @property({ attribute: false }) public config: RoomConfig | null = null;
  /** All rooms, so the schedule lists can say where else a helper is used. */
  @property({ attribute: false }) public rooms: Record<string, RoomConfig> = {};
  @property({ attribute: false }) public hass!: HomeAssistant;
  @property({ type: Boolean }) public presenceEnabled = false;
  /** A price source is set in the settings: shows the per-room price options. */
  @property({ type: Boolean }) public priceConfigured = false;
  @property({ attribute: false }) public presencePersons: string[] = [];
  @property({ type: Boolean }) public climateControlActive = true;
  /** Global: what rooms do while their schedule is off / everybody is away. */
  @property({ type: String }) public scheduleOffAction: "eco" | "off" = "eco";
  @property({ type: String }) public presenceAwayAction: "eco" | "off" = "eco";

  @property({ type: String }) public controlMode: "mpc" | "bangbang" = "mpc";
  @property({ attribute: false }) public outdoorTemp: number | null = null;
  @property({ type: Boolean }) public vacationActive = false;
  @property({ attribute: false }) public vacationUntil: number | null = null;
  /** Live Central Plant status, so the room hero can name a plant guard that affects it. */
  @property({ attribute: false }) public centralPlant: CentralPlantLive | null = null;
  /** Live house layer, so a following room can show what it inherits. */
  @property({ attribute: false }) public house: HouseLive | null = null;

  @state() private _followHouse = false;
  @state() private _savingFollow = false;
  @state() private _selectedTempSensor = "";
  @state() private _selectedHumiditySensor = "";
  @state() private _selectedOccupancySensors: Set<string> = new Set();
  @state() private _occupancyMode: OccupancyMode = "reactive_and_trend";
  @state() private _occupancyDwellMinutes = DEFAULT_OCCUPANCY_DWELL_MINUTES;
  @state() private _preheatEnabled = true;
  @state() private _priceShift = 0;
  @state() private _occupancyEcoDelay = DEFAULT_OCCUPANCY_ECO_DELAY;
  @state() private _co2Sensor = "";
  @state() private _co2Threshold = DEFAULT_CO2_THRESHOLD;
  @state() private _vocSensor = "";
  @state() private _vocThreshold: number | null = null;
  @state() private _ventilationEntity = "";
  @state() private _selectedWindowSensors: Set<string> = new Set();
  @state() private _windowOpenDelay = 0;
  @state() private _windowCloseDelay = 0;
  @state() private _selectedInteriorDoorSensors: Set<string> = new Set();
  @state() private _interiorDoorDelay = DEFAULT_INTERIOR_DOOR_DELAY;
  @state() private _doorClosedMixFactor = DEFAULT_DOOR_CLOSED_MIX_FACTOR;
  @state() private _climateMode: ClimateMode = "auto";
  @state() private _schedules: ScheduleEntry[] = [];
  @state() private _scheduleSelectorEntity = "";
  @state() private _comfortHeat = 21.0;
  @state() private _comfortCool = 24.0;
  @state() private _ecoHeat = 17.0;
  @state() private _ecoCool = 27.0;
  @state() private _error = "";
  @state() private _dirty = false;
  @state() private _editing: EditableSection | null = null;
  @state() private _selectedPresencePersons: string[] = [];
  @state() private _displayName = "";
  @state() private _selectedCovers: Set<string> = new Set();
  @state() private _selectedReadOnlyCovers: Set<string> = new Set();
  @state() private _coversAutoEnabled = false;
  @state() private _coversDeployThreshold = 1.5;
  @state() private _coversMinPosition = 0;
  @state() private _coversOverrideMinutes = 60;
  @state() private _coverSchedules: CoverScheduleEntry[] = [];
  @state() private _coverScheduleSelectorEntity = "";
  @state() private _coversNightClose = false;
  @state() private _coversNightPosition = 0;
  @state() private _coversSnapDeploy = false;
  @state() private _coversWindowHold = false;
  @state() private _coverOrientations: Record<string, number> = {};
  @state() private _coversNightCloseElevation = 0;
  @state() private _coversNightCloseOffsetMinutes = 0;
  @state() private _coversOutdoorMinTemp: number | null = 10;
  @state() private _coverMinPositions: Record<string, number> = {};
  @state() private _ignorePresence = false;
  @state() private _isOutdoor = false;
  @state() private _climateControlEnabled = true;
  @state() private _optimisticCoverResume = false;
  /** Override as the override control shows it (optimistic until the poll confirms). */
  @state() private _override: EffectiveOverride = NO_OVERRIDE;
  @state() private _confirmDelete = false;
  @state() private _deleting = false;
  @state() private _deleteError = "";

  private _prevAreaId: string | null = null;
  /** Room fields as last read from the config (null before the room is stored). */
  private _stored: RoomFields | null = null;
  /** Room the `_stored` fields belong to: a save that ends after a room switch must not touch them. */
  private _storedArea: string | null = null;
  /** Fields saved here that no poll has shown yet: an answer read before the save is ignored for them. */
  private _unconfirmed = new Map<string, { value: unknown; at: number }>();
  private _saveDebounce?: ReturnType<typeof setTimeout>;

  static styles = [
    tokens,
    editDialogHelpStyles,
    css`
      :host {
        display: block;
        max-width: 1240px;
        margin: 0 auto;
      }

      .layout {
        display: grid;
        grid-template-columns: minmax(0, 1fr);
        gap: var(--rs-space-4);
        align-items: start;
      }

      .col {
        display: flex;
        flex-direction: column;
        gap: var(--rs-space-4);
        min-width: 0;
      }

      @media (min-width: 960px) {
        .layout {
          grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
        }

        /* sticky only while it fits; a taller column scrolls with the page */
        .col.left {
          position: sticky;
          top: var(--rs-sticky-top, 72px);
          max-height: calc(100vh - var(--rs-sticky-top, 72px) - 16px);
          overflow-y: auto;
          overscroll-behavior: contain;
          scrollbar-width: thin;
        }
      }

      .mode {
        padding: var(--rs-space-1) var(--rs-space-4) var(--rs-space-4);
      }

      .mode-label {
        font-size: var(--rs-font-sm);
        color: var(--secondary-text-color);
        margin: var(--rs-space-2) 0 6px;
      }

      .mode-desc {
        font-size: var(--rs-font-sm);
        color: var(--secondary-text-color);
        margin-top: var(--rs-space-2);
        line-height: 1.4;
      }

      .error {
        color: var(--rs-err-text);
        font-size: 13px;
      }

      .follow-actions {
        display: flex;
        padding: 0 var(--rs-space-4) var(--rs-space-3);
      }

      .inherited {
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding: var(--rs-space-2) var(--rs-space-4) var(--rs-space-4);
        font-size: var(--rs-font-sm);
        color: var(--secondary-text-color);
      }

      .inherited b {
        color: var(--primary-text-color);
        font-weight: 500;
        font-variant-numeric: tabular-nums;
      }
    `,
  ];

  connectedCallback() {
    super.connectedCallback();
    this._initFromConfig();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    // Leaving the room within the debounce: send the change instead of dropping it
    this._flushSave();
  }

  willUpdate() {
    const currentAreaId = this.config?.area_id ?? this.area?.area_id ?? null;
    // The same element shows another room: the old room's pending edits go to the old room
    if (this._prevAreaId && currentAreaId !== this._prevAreaId) this._flushSave(this._prevAreaId);
  }

  updated(changedProps: Map<string, unknown>) {
    const currentAreaId = this.config?.area_id ?? this.area?.area_id ?? null;
    const areaChanged = currentAreaId !== this._prevAreaId;

    if (areaChanged) {
      this._initFromConfig();
      this._prevAreaId = currentAreaId;
    } else if (changedProps.has("config") && this.config) {
      if (this._stored === null && !this._dirty) this._initFromConfig();
      else this._adoptFromConfig();
    }

    if (
      changedProps.has("config") &&
      this._optimisticCoverResume &&
      !this.config?.live?.cover_auto_paused
    ) {
      this._optimisticCoverResume = false;
    }
  }

  private _initFromConfig() {
    const fields = Object.fromEntries(
      ROOM_FIELDS.map(([key, sync]) => [key, sync.read(this.config)]),
    );
    this._applyFields(fields);
    this._stored = this.config ? fields : null;
    this._storedArea = this.config?.area_id ?? this.area?.area_id ?? null;
    this._unconfirmed = new Map();
    this._dirty = false;
  }

  /** Fields changed elsewhere since the last poll and not here are taken over (see ROOM_FIELDS). */
  private _adoptFromConfig() {
    const incoming = Object.fromEntries(
      ROOM_FIELDS.map(([key, sync]) => [key, sync.read(this.config)]),
    );
    const stored: RoomFields = { ...(this._stored ?? {}) };
    const local = this._localFields();
    const adopt: RoomFields = {};
    const now = Date.now();
    for (const [key, value] of Object.entries(incoming)) {
      const sent = this._unconfirmed.get(key);
      if (sent) {
        if (!sameValue(value, sent.value) && now - sent.at < UNCONFIRMED_MS) continue;
        this._unconfirmed.delete(key);
      }
      if (!sameValue(value, stored[key]) && sameValue(local[key], stored[key])) adopt[key] = value;
      stored[key] = value;
    }
    this._applyFields(adopt);
    this._stored = stored;
  }

  private _applyFields(fields: RoomFields) {
    const self = this as unknown as Record<string, unknown>;
    for (const [key, sync] of ROOM_FIELDS) {
      if (!(key in fields)) continue;
      self[sync.prop] = sync.toState ? sync.toState(fields[key]) : fields[key];
    }
  }

  private _localFields(): RoomFields {
    const self = this as unknown as Record<string, unknown>;
    return Object.fromEntries(
      ROOM_FIELDS.map(([key, sync]) => {
        const state = self[sync.prop];
        return [key, sync.toPayload ? sync.toPayload(state) : state];
      }),
    );
  }

  private _openEdit = (section: EditableSection) => (e?: Event) => {
    e?.stopPropagation();
    this._editing = section;
  };

  private _onScheduleEditClick(e: CustomEvent<{ part?: ScheduleCardPart } | undefined>) {
    e.stopPropagation();
    this._editing = e.detail?.part ?? "schedule";
  }

  private _closeEdit = () => {
    this._editing = null;
  };

  private _onOverrideState(e: CustomEvent<EffectiveOverride>) {
    e.stopPropagation();
    this._override = e.detail;
  }

  private _onOpenAnalytics() {
    this.dispatchEvent(new CustomEvent("open-analytics", { bubbles: true, composed: true }));
  }

  private _onDeleteClick() {
    this._deleteError = "";
    this._confirmDelete = true;
  }

  private _onDeleteCancel = () => {
    if (this._deleting) return;
    this._confirmDelete = false;
  };

  private async _onDeleteConfirm() {
    this._deleting = true;
    this._deleteError = "";
    try {
      await this.hass.callWS({ type: "roommind/rooms/delete", area_id: this.area.area_id });
      this._confirmDelete = false;
      this.dispatchEvent(new CustomEvent("room-deleted", { bubbles: true, composed: true }));
    } catch (err) {
      const l = this.hass.language;
      this._deleteError = errorText(err, localize("detail.delete_error", l), l);
    } finally {
      this._deleting = false;
    }
  }

  render() {
    if (!this.area) return nothing;
    const l = this.hass.language;
    const outdoor = this._isOutdoor;

    return html`
      <div class="layout">
        <div class="col left">
          <rs-room-hero
            .hass=${this.hass}
            .area=${this.area}
            .config=${this.config}
            .isOutdoor=${outdoor}
            .overrideInfo=${this._override}
            .climateControlActive=${this.climateControlActive && this._climateControlEnabled}
            .controlMode=${this.controlMode}
            .outdoorTemp=${this.outdoorTemp}
            .vacationActive=${this.vacationActive}
            .vacationUntil=${this.vacationUntil}
            .centralPlant=${this.centralPlant}
            @display-name-changed=${this._onDisplayNameChanged}
          >
            ${
              this.config && !outdoor
                ? html`<rs-override-section
                    slot="control"
                    .hass=${this.hass}
                    .config=${this.config}
                    .climateMode=${this._climateMode}
                    .comfortHeat=${this._comfortHeat}
                    .comfortCool=${this._comfortCool}
                    .ecoHeat=${this._ecoHeat}
                    .ecoCool=${this._ecoCool}
                    .language=${l}
                    .disabled=${!(this.climateControlActive && this._climateControlEnabled)}
                    @override-state=${this._onOverrideState}
                  ></rs-override-section>`
                : nothing
            }
          </rs-room-hero>
          ${outdoor ? nothing : this._renderFollowHouseCard()}
          ${outdoor ? nothing : this._renderControlCard()}
        </div>
        <div class="col right">
          ${outdoor ? this._renderSensorsCard() : this._renderSectionCards()}
          ${this._renderRoomCard()}
          ${
            this._error && this._editing === null
              ? html`<div class="error" role="alert">${this._error}</div>`
              : nothing
          }
        </div>
      </div>
      ${this._renderEditDialog()} ${this._renderDeleteDialog()}
    `;
  }

  /**
   * Follow house vs Independent. The inherited comfort/eco/mode read-only summary and the two
   * actions use the house block the panel supplies; without it the room still renders correctly.
   */
  private _renderFollowHouseCard() {
    const l = this.hass.language;
    const following = this._followHouse;
    const inherited = this._inheritedTargets();
    return html`<rs-list-card .heading=${localize("house.title", l)}>
      <rs-list-row
        icon=${following ? "mdi:home-thermometer-outline" : "mdi:home-thermometer-off-outline"}
        .heading=${localize(following ? "house.follow_house" : "house.independent", l)}
        .secondary=${
          following
            ? localize("house.inherited_hint", l)
            : localize("detail.schedule.dialog_temps", l)
        }
      ></rs-list-row>
      <div class="follow-actions">
        ${
          following
            ? html`<ha-button .disabled=${this._savingFollow} @click=${this._onCustomizeRoom}
                >${localize("house.customize", l)}</ha-button
              >`
            : html`<ha-button .disabled=${this._savingFollow} @click=${this._onFollowHouse}
                >${localize("house.follow_house", l)}</ha-button
              >`
        }
      </div>
      ${
        following
          ? html`<div class="inherited">
              <span>${localize("house.mode", l)}: <b>${houseModeLabel(this.house, l)}</b></span>
              <span
                >${localize("schedule.column_comfort", l)}:
                <b>${this._formatInherited(inherited.comfortHeat, inherited.comfortCool)}</b></span
              >
              <span
                >${localize("schedule.column_eco", l)}:
                <b>${this._formatInherited(inherited.ecoHeat, inherited.ecoCool)}</b></span
              >
            </div>`
          : nothing
      }
    </rs-list-card>`;
  }

  /** Comfort pair comes from the in-force house band, eco from the stored eco pair. */
  private _inheritedTargets(): {
    comfortHeat: number;
    comfortCool: number;
    ecoHeat: number;
    ecoCool: number;
    climateMode: ClimateMode;
  } {
    const h = this.house;
    return {
      comfortHeat: h?.band.heat_c ?? h?.heat_c ?? HOUSE_DEFAULTS.heat_c ?? 21,
      comfortCool: h?.band.cool_c ?? h?.cool_c ?? HOUSE_DEFAULTS.cool_c ?? 24,
      ecoHeat: h?.eco_heat_c ?? HOUSE_DEFAULTS.eco_heat_c,
      ecoCool: h?.eco_cool_c ?? HOUSE_DEFAULTS.eco_cool_c,
      climateMode: h?.climate_mode ?? HOUSE_DEFAULTS.climate_mode,
    };
  }

  private _formatInherited(heat: number | null, cool: number | null): string {
    const one = (v: number | null) => (v == null ? "—" : formatTempLocale(v, this.hass, 0, 1));
    return `${one(heat)} / ${one(cool)}`;
  }

  /** Seed the room's own values from what it currently inherits, then save it as independent. */
  private async _onCustomizeRoom() {
    if (this._savingFollow) return;
    const inherited = this._inheritedTargets();
    this._comfortHeat = inherited.comfortHeat ?? this._comfortHeat;
    this._comfortCool = inherited.comfortCool ?? this._comfortCool;
    this._ecoHeat = inherited.ecoHeat;
    this._ecoCool = inherited.ecoCool;
    this._climateMode = inherited.climateMode;
    this._followHouse = false;
    await this._saveFollow({
      follow_house: false,
      comfort_heat: this._comfortHeat,
      comfort_cool: this._comfortCool,
      eco_heat: this._ecoHeat,
      eco_cool: this._ecoCool,
      climate_mode: this._climateMode,
    });
  }

  private async _onFollowHouse() {
    if (this._savingFollow) return;
    this._followHouse = true;
    await this._saveFollow({ follow_house: true });
  }

  /** Dedicated immediate save for the follow toggle; normal edits keep using _autoSave. */
  private async _saveFollow(changes: Record<string, unknown>) {
    if (this._saveDebounce) {
      clearTimeout(this._saveDebounce);
      this._saveDebounce = undefined;
    }
    this._savingFollow = true;
    fireSaveStatus(this, "saving");
    this._error = "";
    try {
      await this.hass.callWS({
        type: "roommind/rooms/save",
        area_id: this.area.area_id,
        ...changes,
      });
      if (this._storedArea === this.area.area_id) {
        this._stored = { ...(this._stored ?? {}), ...changes };
        const at = Date.now();
        for (const [key, value] of Object.entries(changes))
          this._unconfirmed.set(key, { value, at });
      }
      this._dirty = false;
      fireSaveStatus(this, "saved");
      this.dispatchEvent(new CustomEvent("room-updated", { bubbles: true, composed: true }));
    } catch (err: unknown) {
      const l = this.hass.language;
      this._error = errorText(err, localize("room.error_save_fallback", l), l);
      fireSaveStatus(this, "error");
    } finally {
      this._savingFollow = false;
    }
  }

  private _renderControlCard() {
    const l = this.hass.language;
    const autoDesc: TranslationKey = "mode.auto_desc";
    const modes: {
      value: ClimateMode;
      key: TranslationKey;
      icon: string;
      desc: TranslationKey;
    }[] = [
      { value: "auto", key: "mode.auto", icon: "mdi:autorenew", desc: autoDesc },
      {
        value: "heat_only",
        key: "mode.heat_only",
        icon: "mdi:fire",
        desc: "mode.heat_only_desc_plain",
      },
      {
        value: "cool_only",
        key: "mode.cool_only",
        icon: "mdi:snowflake",
        desc: "mode.cool_only_desc_plain",
      },
    ];
    const current = modes.find((m) => m.value === this._climateMode) ?? modes[0];
    const secondary = !this.climateControlActive
      ? localize("card.control_off_global", l)
      : localize(this._climateControlEnabled ? "detail.control.on" : "detail.control.off", l);
    return html`<rs-list-card>
      <rs-list-row
        switch
        icon="mdi:power"
        .iconColor=${this._climateControlEnabled && this.climateControlActive ? "var(--primary-color)" : ""}
        .heading=${localize("room.climate_control_toggle", l)}
        .secondary=${secondary}
        .checked=${this._climateControlEnabled}
        @toggle-changed=${this._onClimateControlToggle}
      ></rs-list-row>
      <div class="mode">
        <div class="mode-label">${localize("room.section.climate_mode", l)}</div>
        <rs-segmented
          .label=${localize("room.section.climate_mode", l)}
          .options=${modes.map((m) => ({
            value: m.value,
            label: localize(m.key, l),
            icon: m.icon,
          }))}
          .value=${this._climateMode}
          @value-changed=${this._onModeChanged}
        ></rs-segmented>
        <div class="mode-desc">${localize(current.desc, l)}</div>
      </div>
    </rs-list-card>`;
  }

  private _renderSectionCards() {
    const l = this.hass.language;
    const live = this.config?.live;
    return html`
      <rs-list-card
        .heading=${localize("room.section.schedule", l)}
        @edit-click=${this._onScheduleEditClick}
      >
        <rs-schedule-settings
          .hass=${this.hass}
          .rooms=${this.rooms}
          .roomId=${this.area.area_id}
          .schedules=${this._schedules}
          .scheduleSelectorEntity=${this._scheduleSelectorEntity}
          .activeScheduleIndex=${live?.active_schedule_index ?? -1}
          .comfortHeat=${this._comfortHeat}
          .comfortCool=${this._comfortCool}
          .ecoHeat=${this._ecoHeat}
          .ecoCool=${this._ecoCool}
          .climateMode=${this._climateMode}
          .scheduleTempWarnings=${live?.schedule_temp_warnings ?? []}
          .preheatEnabled=${this._preheatEnabled}
          .priceShift=${this._priceShift}
          .priceConfigured=${this.priceConfigured}
          .preheatSides=${this._preheatSides()}
          .scheduleOffAction=${this.scheduleOffAction}
          .presenceAwayAction=${this.presenceAwayAction}
          .presenceActive=${this._presenceActive()}
          .occupancyEco=${this._occupancyMode === "reactive_and_trend" && this._selectedOccupancySensors.size > 0}
          .editing=${false}
          @preheat-changed=${this._onPreheatChanged}
        ></rs-schedule-settings>
      </rs-list-card>

      ${this._renderSensorsCard()}
      ${
        this.presenceEnabled || this._selectedPresencePersons.length > 0 || this._ignorePresence
          ? html`<rs-list-card
              .heading=${localize("room.section.presence", l)}
              .info=${localize("presence.persons_info", l)}
              @edit-click=${this._openEdit("presence")}
            >
              <rs-presence-section
                .hass=${this.hass}
                .presenceEnabled=${this.presenceEnabled}
                .presencePersons=${this.presencePersons}
                .selectedPresencePersons=${this._selectedPresencePersons}
                .ignorePresence=${this._ignorePresence}
                .awayAction=${this.presenceAwayAction}
                .editing=${false}
                .language=${l}
              ></rs-presence-section>
            </rs-list-card>`
          : nothing
      }

      <rs-list-card
        .heading=${localize("room.section.covers", l)}
        @edit-click=${this._openEdit("covers")}
      >
        <rs-covers-section
          .hass=${this.hass}
          .area=${this.area}
          .editing=${false}
          .selectedCovers=${this._selectedCovers}
          .readOnlyCovers=${this._selectedReadOnlyCovers}
          .autoEnabled=${this._coversAutoEnabled}
          .forcedReason=${live?.cover_forced_reason ?? ""}
          .coverReason=${live?.cover_reason ?? ""}
          .autoPaused=${this._optimisticCoverResume ? false : (live?.cover_auto_paused ?? false)}
          .overrideUntil=${live?.cover_override_until ?? null}
          .coverOrientations=${this._coverOrientations}
          .coverMinPositions=${this._coverMinPositions}
          @cover-resume-auto=${this._onCoverResumeAuto}
        ></rs-covers-section>
      </rs-list-card>
    `;
  }

  private _renderSensorsCard() {
    const l = this.hass.language;
    return html`<rs-list-card
      .heading=${localize("room.section.sensors", l)}
      @edit-click=${this._openEdit("sensors")}
    >
      <rs-sensor-section
        .hass=${this.hass}
        .area=${this.area}
        .editing=${false}
        .outdoor=${this._isOutdoor}
        .temperatureSensor=${this._selectedTempSensor}
        .humiditySensor=${this._selectedHumiditySensor}
        .occupancySensors=${this._selectedOccupancySensors}
        .occupancyMode=${this._occupancyMode}
        .occupancyDwellMinutes=${this._occupancyDwellMinutes}
        .occupancyEcoDelay=${this._occupancyEcoDelay}
        .co2Sensor=${this._co2Sensor}
        .co2Threshold=${this._co2Threshold}
        .vocSensor=${this._vocSensor}
        .vocThreshold=${this._vocThreshold}
        .ventilationEntity=${this._ventilationEntity}
        .live=${this.config?.live}
        .exteriorWindowSensors=${this._selectedWindowSensors}
        .windowOpenDelay=${this._windowOpenDelay}
        .windowCloseDelay=${this._windowCloseDelay}
        .interiorDoorSensors=${this._selectedInteriorDoorSensors}
        .interiorDoorDelay=${this._interiorDoorDelay}
        .doorClosedMixFactor=${this._doorClosedMixFactor}
        .language=${l}
      ></rs-sensor-section>
    </rs-list-card>`;
  }

  /** Outdoor flag, analytics and removal: there for every room, outdoor ones included. */
  private _renderRoomCard() {
    const l = this.hass.language;
    return html`<rs-list-card>
      <rs-list-row
        switch
        icon="mdi:tree"
        .heading=${localize("room.outdoor_toggle", l)}
        .secondary=${localize("room.outdoor_hint", l)}
        .checked=${this._isOutdoor}
        @toggle-changed=${this._onOutdoorToggle}
      ></rs-list-row>
      ${
        this.config
          ? html`<rs-list-row
                clickable
                chevron
                icon="mdi:chart-line"
                .heading=${localize("detail.analytics", l)}
                .secondary=${localize(
                  this._isOutdoor ? "detail.analytics_sub_outdoor" : "detail.analytics_sub",
                  l,
                )}
                @row-click=${this._onOpenAnalytics}
              ></rs-list-row>
              <rs-list-row
                clickable
                variant="danger"
                icon="mdi:delete-outline"
                .heading=${localize("detail.delete", l)}
                .secondary=${localize("detail.delete_sub", l)}
                @row-click=${this._onDeleteClick}
              ></rs-list-row>`
          : nothing
      }
    </rs-list-card>`;
  }

  private _renderDeleteDialog() {
    if (!this._confirmDelete) return nothing;
    const l = this.hass.language;
    const name = this.config?.display_name || this.area.name;
    return html`<rs-confirm-dialog
      open
      icon="mdi:delete-outline"
      .heading=${localize("detail.delete_title", l)}
      .message=${localize("detail.delete_body", l, { name })}
      .confirmLabel=${localize("detail.delete_confirm", l)}
      .busy=${this._deleting}
      .error=${this._deleteError}
      @cancel=${this._onDeleteCancel}
      @confirm=${this._onDeleteConfirm}
    ></rs-confirm-dialog>`;
  }

  private _renderEditDialog() {
    if (this._editing === null) return nothing;
    const lang = this.hass.language;

    switch (this._editing) {
      case "schedule":
        return html`<rs-edit-dialog
          open
          icon="mdi:calendar-clock"
          .heading=${localize("detail.schedule.dialog_schedule", lang)}
          hasInfo
          .error=${this._error}
          @rs-dialog-closed=${this._closeEdit}
        >
          ${scheduleHelp(lang)}
          <rs-schedule-settings
            .hass=${this.hass}
            .schedules=${this._schedules}
            .scheduleSelectorEntity=${this._scheduleSelectorEntity}
            .activeScheduleIndex=${this.config?.live?.active_schedule_index ?? -1}
            .comfortHeat=${this._comfortHeat}
            .comfortCool=${this._comfortCool}
            .ecoHeat=${this._ecoHeat}
            .ecoCool=${this._ecoCool}
            .climateMode=${this._climateMode}
            .scheduleTempWarnings=${this.config?.live?.schedule_temp_warnings ?? []}
            .editing=${true}
            .roomName=${this._displayName || this.area.name}
            .rooms=${this.rooms}
            .roomId=${this.area.area_id}
            @schedules-changed=${this._onSchedulesChanged}
            @schedule-selector-changed=${this._onScheduleSelectorChanged}
            @open-schedule-editor=${this._onOpenScheduleEditor}
          ></rs-schedule-settings>
        </rs-edit-dialog>`;
      case "temperatures":
        return html`<rs-edit-dialog
          open
          icon="mdi:thermometer"
          .heading=${localize("detail.schedule.dialog_temps", lang)}
          hasInfo
          .error=${this._error}
          @rs-dialog-closed=${this._closeEdit}
        >
          ${temperaturesHelp(lang)}
          <rs-target-temps
            .hass=${this.hass}
            .comfortHeat=${this._comfortHeat}
            .comfortCool=${this._comfortCool}
            .ecoHeat=${this._ecoHeat}
            .ecoCool=${this._ecoCool}
            .climateMode=${this._climateMode}
            .following=${this._followHouse}
            .inherited=${this._inheritedTargets()}
            @comfort-heat-changed=${this._onComfortHeatChanged}
            @comfort-cool-changed=${this._onComfortCoolChanged}
            @eco-heat-changed=${this._onEcoHeatChanged}
            @eco-cool-changed=${this._onEcoCoolChanged}
            @customize-room=${this._onCustomizeRoomFromDialog}
          ></rs-target-temps>
        </rs-edit-dialog>`;
      case "price":
        return html`<rs-edit-dialog
          open
          icon="mdi:cash-clock"
          .heading=${localize("schedule.price_shift_label", lang)}
          hasInfo
          .error=${this._error}
          @rs-dialog-closed=${this._closeEdit}
        >
          ${priceHelp(lang)}
          <rs-price-shift
            .hass=${this.hass}
            .priceShift=${this._priceShift}
            .climateMode=${this._climateMode}
            .priceLevel=${this.config?.live?.price_level ?? null}
            .shiftHeat=${this.config?.live?.price_shift_heat ?? 0}
            .shiftCool=${this.config?.live?.price_shift_cool ?? 0}
            @price-shift-changed=${this._onPriceShiftChanged}
          ></rs-price-shift>
        </rs-edit-dialog>`;
      case "sensors":
        return html`<rs-edit-dialog
          open
          icon="mdi:thermometer"
          .heading=${localize("room.section.sensors", lang)}
          hasInfo
          .error=${this._error}
          @rs-dialog-closed=${this._closeEdit}
        >
          ${sensorsHelp(lang)}
          <rs-sensor-section
            .hass=${this.hass}
            .area=${this.area}
            .editing=${true}
            .outdoor=${this._isOutdoor}
            .temperatureSensor=${this._selectedTempSensor}
            .humiditySensor=${this._selectedHumiditySensor}
            .occupancySensors=${this._selectedOccupancySensors}
            .occupancyMode=${this._occupancyMode}
            .occupancyDwellMinutes=${this._occupancyDwellMinutes}
            .occupancyEcoDelay=${this._occupancyEcoDelay}
            .co2Sensor=${this._co2Sensor}
            .co2Threshold=${this._co2Threshold}
            .vocSensor=${this._vocSensor}
            .vocThreshold=${this._vocThreshold}
            .ventilationEntity=${this._ventilationEntity}
            .live=${this.config?.live}
            .exteriorWindowSensors=${this._selectedWindowSensors}
            .windowOpenDelay=${this._windowOpenDelay}
            .windowCloseDelay=${this._windowCloseDelay}
            .interiorDoorSensors=${this._selectedInteriorDoorSensors}
            .interiorDoorDelay=${this._interiorDoorDelay}
            .doorClosedMixFactor=${this._doorClosedMixFactor}
            .language=${this.hass.language}
            @sensor-changed=${this._onSensorChanged}
          ></rs-sensor-section>
        </rs-edit-dialog>`;
      case "presence":
        return html`<rs-edit-dialog
          open
          icon="mdi:home-account"
          .heading=${localize("room.section.presence", lang)}
          hasInfo
          .error=${this._error}
          @rs-dialog-closed=${this._closeEdit}
        >
          ${presenceHelp(lang)}
          <rs-presence-section
            .hass=${this.hass}
            .presenceEnabled=${this.presenceEnabled}
            .presencePersons=${this.presencePersons}
            .selectedPresencePersons=${this._selectedPresencePersons}
            .ignorePresence=${this._ignorePresence}
            .awayAction=${this.presenceAwayAction}
            .editing=${true}
            .language=${this.hass.language}
            @presence-persons-changed=${this._onPresencePersonsChanged}
            @ignore-presence-changed=${this._onIgnorePresenceChanged}
          ></rs-presence-section>
        </rs-edit-dialog>`;
      case "covers":
        return html`<rs-edit-dialog
          open
          icon="mdi:blinds-horizontal"
          .heading=${localize("room.section.covers", lang)}
          hasInfo
          .error=${this._error}
          @rs-dialog-closed=${this._closeEdit}
        >
          ${coversHelp(lang)}
          <rs-covers-section
            @open-schedule-editor=${this._onOpenScheduleEditor}
            .rooms=${this.rooms}
            .hass=${this.hass}
            .area=${this.area}
            .editing=${true}
            .selectedCovers=${this._selectedCovers}
            .readOnlyCovers=${this._selectedReadOnlyCovers}
            .autoEnabled=${this._coversAutoEnabled}
            .deployThreshold=${this._coversDeployThreshold}
            .minPosition=${this._coversMinPosition}
            .overrideMinutes=${this._coversOverrideMinutes}
            .coverSchedules=${this._coverSchedules}
            .coverScheduleSelectorEntity=${this._coverScheduleSelectorEntity}
            .activeCoverScheduleIndex=${this.config?.live?.active_cover_schedule_index ?? -1}
            .nightClose=${this._coversNightClose}
            .nightPosition=${this._coversNightPosition}
            .snapDeploy=${this._coversSnapDeploy}
            .forcedReason=${this.config?.live?.cover_forced_reason ?? ""}
            .coverReason=${this.config?.live?.cover_reason ?? ""}
            .windowHold=${this._coversWindowHold}
            .hasWindowSensors=${this._selectedWindowSensors.size > 0}
            .autoPaused=${
              this._optimisticCoverResume ? false : (this.config?.live?.cover_auto_paused ?? false)
            }
            .overrideUntil=${this.config?.live?.cover_override_until ?? null}
            .coverOrientations=${this._coverOrientations}
            .nightCloseElevation=${this._coversNightCloseElevation}
            .nightCloseOffsetMinutes=${this._coversNightCloseOffsetMinutes}
            .outdoorMinTemp=${this._coversOutdoorMinTemp}
            .coverMinPositions=${this._coverMinPositions}
            @covers-toggle=${this._onCoversToggle}
            @read-only-covers-toggle=${this._onReadOnlyCoversToggle}
            @setting-changed=${this._onCoverSettingChanged}
            @cover-resume-auto=${this._onCoverResumeAuto}
          ></rs-covers-section>
        </rs-edit-dialog>`;
    }
  }

  // ---- Child event handlers ----

  private _onModeChanged(e: CustomEvent<{ value: ClimateMode }>) {
    e.stopPropagation();
    this._climateMode = e.detail.value;
    this._autoSave();
  }

  /**
   * The schedule dialog opens the editor page: a pending room save (e.g. the just created
   * helper's assignment) goes out first, the page switch unmounts this element.
   */
  private async _onOpenScheduleEditor(
    e: CustomEvent<{ entityId: string; kind?: "climate" | "cover" }>,
  ) {
    e.stopPropagation();
    if (this._saveDebounce) {
      clearTimeout(this._saveDebounce);
      this._saveDebounce = undefined;
      await this._doSave();
    }
    this._editing = null;
    this.dispatchEvent(
      new CustomEvent("open-schedule-editor", {
        detail: {
          entityId: e.detail.entityId,
          areaId: this.area.area_id,
          kind: e.detail.kind ?? "climate",
        },
        bubbles: true,
        composed: true,
      }),
    );
  }

  private _onSchedulesChanged(e: CustomEvent<{ value: ScheduleEntry[] }>) {
    const before = this._schedules;
    const next = e.detail.value;
    this._schedules = next;
    this._autoSave();
    // Removing is easily undone, so it asks nothing and offers Undo instead
    const removed = before.find((s) => !next.some((n) => n.entity_id === s.entity_id));
    if (removed && next.length < before.length) {
      const l = this.hass.language;
      const name =
        (this.hass.states[removed.entity_id]?.attributes?.friendly_name as string) ||
        removed.entity_id;
      showToast(
        this,
        localize("schedule.removed", l, { name }),
        l,
        this._undoAction({ schedules: before }, () => (this._schedules = before)),
      );
    }
  }

  private _onScheduleSelectorChanged(e: CustomEvent<{ value: string }>) {
    this._scheduleSelectorEntity = e.detail.value;
    this._autoSave();
  }

  /** The temperatures dialog's "Customize this room" closes the dialog and seeds from the house. */
  private _onCustomizeRoomFromDialog() {
    this._closeEdit();
    void this._onCustomizeRoom();
  }

  private _onComfortHeatChanged(e: CustomEvent<{ value: number }>) {
    this._comfortHeat = e.detail.value;
    if (this._comfortCool < this._comfortHeat) this._comfortCool = this._comfortHeat;
    this._autoSave();
  }

  private _onComfortCoolChanged(e: CustomEvent<{ value: number }>) {
    this._comfortCool = e.detail.value;
    if (this._comfortHeat > this._comfortCool) this._comfortHeat = this._comfortCool;
    this._autoSave();
  }

  private _onEcoHeatChanged(e: CustomEvent<{ value: number }>) {
    this._ecoHeat = e.detail.value;
    if (this._ecoCool < this._ecoHeat) this._ecoCool = this._ecoHeat;
    this._autoSave();
  }

  /**
   * Undo for a toast, which outlives the room page. While the page still shows this room the
   * undo goes through the auto-save; a detached page, or one reused for another room, would
   * save its own state, so then only the changed field goes to the store.
   */
  private _undoAction(changes: Record<string, unknown>, restore: () => void): () => void {
    const areaId = this.area.area_id;
    return () => {
      if (this.isConnected && this.area.area_id === areaId) {
        restore();
        this._autoSave();
        return;
      }
      this.hass
        .callWS({ type: "roommind/rooms/save", area_id: areaId, ...changes })
        .catch((err: unknown) => {
          // eslint-disable-next-line no-console
          console.error("Undo failed:", err);
        });
    };
  }

  /**
   * Presence detection switches this room: on globally, someone in the global list (the
   * presence section says the same), not ignored.
   */
  private _presenceActive(): boolean {
    return this.presenceEnabled && !this._ignorePresence && this.presencePersons.length > 0;
  }

  /**
   * Where the pre-heat/pre-cool lead applies: MPC with a room sensor and a schedule (or price
   * shift) to start early for; the Central Plant can heat or cool as its mode allows.
   */
  private _preheatSides(): "" | "heat" | "cool" | "both" {
    if (
      (!this._schedules.length && !(this._priceShift > 0)) ||
      this.controlMode !== "mpc" ||
      !this._selectedTempSensor
    ) {
      return "";
    }
    const heat = this._climateMode !== "cool_only";
    const cool = this._climateMode !== "heat_only";
    return heat && cool ? "both" : heat ? "heat" : cool ? "cool" : "";
  }

  private _onPreheatChanged(e: CustomEvent<{ value: boolean }>) {
    this._preheatEnabled = e.detail.value;
    this._autoSave();
  }

  private _onPriceShiftChanged(e: CustomEvent<{ value: number }>) {
    this._priceShift = e.detail.value;
    this._autoSave();
  }

  private _onEcoCoolChanged(e: CustomEvent<{ value: number }>) {
    this._ecoCool = e.detail.value;
    if (this._ecoHeat > this._ecoCool) this._ecoHeat = this._ecoCool;
    this._autoSave();
  }

  private _onSensorChanged(
    e: CustomEvent<{ key: string; value: string | string[] | number | boolean | null }>,
  ) {
    const { key, value } = e.detail;
    if (key === "temperature_sensor") {
      this._selectedTempSensor = value as string;
    } else if (key === "humidity_sensor") {
      this._selectedHumiditySensor = value as string;
    } else if (key === "occupancy_sensors") {
      this._selectedOccupancySensors = new Set(value as string[]);
    } else if (key === "occupancy_mode") {
      this._occupancyMode = value as OccupancyMode;
    } else if (key === "occupancy_dwell_minutes") {
      this._occupancyDwellMinutes = value as number;
    } else if (key === "occupancy_eco_delay") {
      this._occupancyEcoDelay = value as number;
    } else if (key === "co2_sensor") {
      this._co2Sensor = value as string;
    } else if (key === "co2_threshold") {
      this._co2Threshold = value as number;
    } else if (key === "voc_sensor") {
      this._vocSensor = value as string;
    } else if (key === "voc_threshold") {
      this._vocThreshold = value as number | null;
    } else if (key === "ventilation_entity") {
      this._ventilationEntity = value as string;
    } else if (key === "exterior_window_sensors") {
      this._selectedWindowSensors = new Set(value as string[]);
    } else if (key === "window_open_delay") {
      this._windowOpenDelay = value as number;
    } else if (key === "window_close_delay") {
      this._windowCloseDelay = value as number;
    } else if (key === "interior_door_sensors") {
      this._selectedInteriorDoorSensors = new Set(value as string[]);
    } else if (key === "interior_door_delay") {
      this._interiorDoorDelay = value as number;
    } else if (key === "door_closed_mix_factor") {
      this._doorClosedMixFactor = value as number;
    }
    this._autoSave();
  }

  private _onPresencePersonsChanged(e: CustomEvent<string[]>) {
    this._selectedPresencePersons = e.detail;
    this._autoSave();
  }

  private _onIgnorePresenceChanged(e: CustomEvent<boolean>) {
    this._ignorePresence = e.detail;
    this._autoSave();
  }

  // ---- Cover event handlers ----

  private _onCoversToggle(e: CustomEvent<{ entityId: string; checked: boolean }>) {
    const { entityId, checked } = e.detail;
    const next = new Set(this._selectedCovers);
    const readOnly = new Set(this._selectedReadOnlyCovers);
    if (checked) {
      next.add(entityId);
      readOnly.delete(entityId);
    } else {
      next.delete(entityId);
      if (entityId in this._coverOrientations) {
        const nextOrientations = { ...this._coverOrientations };
        delete nextOrientations[entityId];
        this._coverOrientations = nextOrientations;
      }
      if (entityId in this._coverMinPositions) {
        const nextMinPositions = { ...this._coverMinPositions };
        delete nextMinPositions[entityId];
        this._coverMinPositions = nextMinPositions;
      }
    }
    this._selectedCovers = next;
    this._selectedReadOnlyCovers = readOnly;
    this._autoSave();
  }

  private _onReadOnlyCoversToggle(e: CustomEvent<{ entityId: string; checked: boolean }>) {
    const { entityId, checked } = e.detail;
    const next = new Set(this._selectedReadOnlyCovers);
    const motorized = new Set(this._selectedCovers);
    if (checked) {
      next.add(entityId);
      motorized.delete(entityId);
    } else {
      next.delete(entityId);
    }
    this._selectedReadOnlyCovers = next;
    this._selectedCovers = motorized;
    this._autoSave();
  }

  /** Lives here, not in the dialog: Undo still works after the dialog was closed. */
  private _onCoverSchedules(next: CoverScheduleEntry[]) {
    const before = this._coverSchedules;
    this._coverSchedules = next;
    const removed = before.find((s) => !next.some((n) => n.entity_id === s.entity_id));
    if (!removed || next.length >= before.length) return;
    const l = this.hass.language;
    const name =
      (this.hass.states[removed.entity_id]?.attributes?.friendly_name as string) ||
      removed.entity_id;
    showToast(
      this,
      localize("covers.schedule_removed", l, { name }),
      l,
      this._undoAction({ cover_schedules: before }, () => (this._coverSchedules = before)),
    );
  }

  private _onCoverSettingChanged(e: CustomEvent<{ key: string; value: unknown }>) {
    const { key, value } = e.detail;
    e.stopPropagation();
    if (key === "covers_auto_enabled") this._coversAutoEnabled = value as boolean;
    else if (key === "covers_deploy_threshold") this._coversDeployThreshold = value as number;
    else if (key === "covers_min_position") this._coversMinPosition = value as number;
    else if (key === "covers_override_minutes") this._coversOverrideMinutes = value as number;
    else if (key === "cover_schedules") this._onCoverSchedules(value as CoverScheduleEntry[]);
    else if (key === "cover_schedule_selector_entity")
      this._coverScheduleSelectorEntity = value as string;
    else if (key === "covers_night_close") this._coversNightClose = value as boolean;
    else if (key === "covers_night_position") this._coversNightPosition = value as number;
    else if (key === "covers_snap_deploy") this._coversSnapDeploy = value as boolean;
    else if (key === "covers_window_hold") this._coversWindowHold = value as boolean;
    else if (key === "cover_orientations")
      this._coverOrientations = value as Record<string, number>;
    else if (key === "covers_night_close_elevation")
      this._coversNightCloseElevation = value as number;
    else if (key === "covers_night_close_offset_minutes")
      this._coversNightCloseOffsetMinutes = value as number;
    else if (key === "covers_outdoor_min_temp") this._coversOutdoorMinTemp = value as number | null;
    else if (key === "cover_min_positions")
      this._coverMinPositions = value as Record<string, number>;
    this._autoSave();
  }

  private async _onCoverResumeAuto() {
    this._optimisticCoverResume = true;
    try {
      await this.hass.callWS({
        type: "roommind/covers/clear_override",
        area_id: this.area.area_id,
      });
    } catch (err) {
      this._optimisticCoverResume = false;
      const l = this.hass.language;
      this._error = errorText(err, localize("room.error_save_fallback", l), l);
    }
  }

  // ---- Outdoor toggle ----

  private _onClimateControlToggle(e: CustomEvent) {
    this._climateControlEnabled = e.detail;
    this._autoSave();
  }

  private _onOutdoorToggle(e: CustomEvent<boolean>) {
    this._isOutdoor = e.detail;
    this._autoSave();
  }

  // ---- Auto-save ----

  private _onDisplayNameChanged(e: CustomEvent<{ value: string }>) {
    this._displayName = e.detail.value;
    this._autoSave();
  }

  private _autoSave() {
    this._dirty = true;
    if (this._saveDebounce) clearTimeout(this._saveDebounce);
    this._saveDebounce = setTimeout(() => {
      this._saveDebounce = undefined;
      void this._doSave();
    }, 500);
  }

  private _flushSave(areaId?: string) {
    if (!this._saveDebounce) return;
    clearTimeout(this._saveDebounce);
    this._saveDebounce = undefined;
    void this._doSave(areaId);
  }

  private async _doSave(areaId: string = this.area.area_id) {
    fireSaveStatus(this, "saving");
    this._error = "";

    try {
      const local = this._localFields();
      const stored = this._stored;
      // Only what changed here: a field changed elsewhere since the last poll stays as it is
      const changes = stored
        ? Object.fromEntries(
            Object.entries(local).filter(([key, value]) => !sameValue(value, stored[key])),
          )
        : local;
      await this.hass.callWS({ type: "roommind/rooms/save", area_id: areaId, ...changes });
      // Stored now: changing a field straight back must go out again, even before the next poll
      if (this._storedArea === areaId) {
        this._stored = { ...(this._stored ?? {}), ...changes };
        const at = Date.now();
        for (const [key, value] of Object.entries(changes))
          this._unconfirmed.set(key, { value, at });
      }

      this._dirty = false;
      fireSaveStatus(this, "saved");

      this.dispatchEvent(
        new CustomEvent("room-updated", {
          bubbles: true,
          composed: true,
        }),
      );
    } catch (err: unknown) {
      const l = this.hass.language;
      this._error = errorText(err, localize("room.error_save_fallback", l), l);
      fireSaveStatus(this, "error");
    }
  }
}

declare global {
  interface HTMLElementTagNameMap {
    "rs-room-detail": RsRoomDetail;
  }
}
