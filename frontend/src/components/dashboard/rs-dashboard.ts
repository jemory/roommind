import { LitElement, html, css, nothing } from "lit";
import { customElement, property } from "lit/decorators.js";
import { repeat } from "lit/directives/repeat.js";
import type {
  CentralPlantLive,
  HomeAssistant,
  HouseLive,
  PlantAction,
  PriceStatus,
} from "../../types";
import { floorGroups, type AreaInfo } from "../../utils/dashboard";
import { isControlledZone } from "../../utils/room-state";
import {
  followingRoomsLabel,
  formatHouseTarget,
  houseHoldLabel,
  houseModeLabel,
  houseOverrideActive,
  houseSourceLabel,
} from "../../utils/house-state";
import { localize, type TranslationKey } from "../../utils/localize";
import { formatTempLocale, tempUnit } from "../../utils/temperature";
import { VACATION_SENTINEL } from "../../utils/constants";
import { tokens } from "../../styles/tokens";
import { dashboardStyles } from "./dashboard-styles";
import "./rs-room-card";
import "../shared/rs-pill";

export interface HouseSummary {
  vacationActive: boolean;
  vacationUntil: number | null;
  presenceEnabled: boolean;
  anyoneHome: boolean;
  outdoorTemp: number | null;
  priceEntity: string;
  priceStatus: PriceStatus | null;
}

const PRICE_LEVEL_KEYS: Record<"cheap" | "normal" | "expensive", TranslationKey> = {
  cheap: "settings.price_level_cheap",
  normal: "settings.price_level_normal",
  expensive: "settings.price_level_expensive",
};

const ACTION_KEYS: Record<PlantAction, TranslationKey> = {
  heating: "plant.action.heating",
  cooling: "plant.action.cooling",
  circulating: "plant.action.circulating",
  idle: "plant.action.idle",
};

const ACTION_ICONS: Record<PlantAction, string> = {
  heating: "mdi:fire",
  cooling: "mdi:snowflake",
  circulating: "mdi:fan",
  idle: "mdi:circle-small",
};

const GUARD_KEYS: Record<string, TranslationKey> = {
  min_run: "plant.guard.min_run",
  min_off: "plant.guard.min_off",
  min_fan: "plant.guard.min_fan",
  changeover: "plant.guard.changeover",
  exterior_open: "plant.guard.exterior_open",
  wet_coil: "plant.guard.wet_coil",
};

/** Rooms tab: the Central HVAC status card, house summary pills and the room cards. */
@customElement("rs-dashboard")
export class RsDashboard extends LitElement {
  @property({ attribute: false }) public hass!: HomeAssistant;
  /** All areas in display order, hidden ones included. */
  @property({ attribute: false }) public infos: AreaInfo[] = [];
  @property({ attribute: false }) public hiddenRooms: string[] = [];
  @property({ type: Boolean }) public groupByFloor = false;
  @property({ type: String }) public controlMode: "mpc" | "bangbang" = "mpc";
  @property({ type: Boolean }) public climateControlActive = true;
  @property({ attribute: false }) public summary!: HouseSummary;
  /** Live Central Plant status from rooms/list; null until the backend reports it. */
  @property({ attribute: false }) public centralPlant: CentralPlantLive | null = null;
  /** Live house layer from rooms/list; null until the backend reports it (or on older installs). */
  @property({ attribute: false }) public house: HouseLive | null = null;

  private _clearingOverride = false;
  private _resumingHouseSchedule = false;

  static styles = [
    tokens,
    dashboardStyles,
    css`
      .plant {
        margin: 0 0 12px;
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        gap: 10px;
      }

      .plant-head {
        display: flex;
        align-items: center;
        gap: 12px;
        min-width: 0;
      }

      .plant-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        flex: none;
        display: grid;
        place-items: center;
        background: var(--rs-fill);
        color: var(--secondary-text-color);
        --mdc-icon-size: 22px;
      }

      .plant-icon.heating {
        background: color-mix(in srgb, var(--rs-heat) 16%, transparent);
        color: var(--rs-heat);
      }

      .plant-icon.cooling {
        background: color-mix(in srgb, var(--rs-cool) 16%, transparent);
        color: var(--rs-cool);
      }

      .plant-icon.circulating {
        background: color-mix(in srgb, var(--rs-ok) 16%, transparent);
        color: var(--rs-ok);
      }

      .plant-text {
        flex: 1;
        min-width: 0;
      }

      .plant-title {
        font-size: 15px;
        font-weight: 500;
        line-height: 1.3;
      }

      .plant-sub {
        font-size: 12px;
        color: var(--secondary-text-color);
        line-height: 1.4;
        overflow-wrap: anywhere;
      }

      .plant-temps {
        display: flex;
        gap: 16px;
        flex: none;
        font-size: 12px;
        color: var(--secondary-text-color);
        flex-wrap: wrap;
        justify-content: flex-end;
      }

      .plant-temps span {
        display: inline-flex;
        align-items: baseline;
        gap: 4px;
        white-space: nowrap;
      }

      .plant-temps b {
        font-size: 16px;
        font-weight: 400;
        color: var(--primary-text-color);
        font-variant-numeric: tabular-nums;
      }

      .plant-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 6px 14px;
        font-size: 12px;
        color: var(--secondary-text-color);
        font-variant-numeric: tabular-nums;
        align-items: center;
      }

      .plant-meta .driving {
        min-width: 0;
        overflow-wrap: anywhere;
      }

      .plant-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        align-items: center;
      }

      .plant-chips ha-button {
        --mdc-typography-button-font-size: 12px;
        margin-left: auto;
      }

      .unconfigured .plant-title {
        font-size: 14px;
        font-weight: 500;
      }

      @media (max-width: 600px) {
        .plant-temps {
          justify-content: flex-start;
        }
      }

      /* Compact house hero: target band, mode, schedule/hold, followers, actions. */
      .house {
        margin: 0 0 12px;
        padding: 12px 16px;
        display: flex;
        flex-direction: column;
        gap: 10px;
      }

      .house-head {
        display: flex;
        align-items: center;
        gap: 12px;
        min-width: 0;
      }

      .house-icon {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        flex: none;
        display: grid;
        place-items: center;
        background: var(--rs-fill);
        color: var(--primary-color);
        --mdc-icon-size: 20px;
      }

      .house-text {
        flex: 1;
        min-width: 0;
      }

      .house-title {
        font-size: 14px;
        font-weight: 500;
        line-height: 1.3;
      }

      .house-band {
        font-size: 18px;
        font-weight: 500;
        font-variant-numeric: tabular-nums;
      }

      .house-sub {
        font-size: 12px;
        color: var(--secondary-text-color);
        line-height: 1.4;
        overflow-wrap: anywhere;
      }

      .house-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 6px 14px;
        align-items: center;
        font-size: 12px;
        color: var(--secondary-text-color);
      }

      .house-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
      }

      .house-actions ha-button {
        --mdc-typography-button-font-size: 12px;
        min-height: 32px;
      }

      .house-actions .link {
        background: none;
        border: none;
        color: var(--primary-color);
        font: inherit;
        font-size: 12px;
        cursor: pointer;
        padding: 4px 2px;
      }
    `,
  ];

  render() {
    const l = this.hass.language;
    const hero = this._renderHouseHero();
    if (this.infos.length === 0) {
      return html`${hero}
        <div class="placeholder">
          <ha-icon icon="mdi:home"></ha-icon>
          <p>${localize("panel.no_areas", l)}<br />${localize("panel.no_areas_hint", l)}</p>
        </div>`;
    }
    const hidden = new Set(this.hiddenRooms);
    const visible = this.infos.filter((i) => !hidden.has(i.area.area_id));
    if (visible.length === 0) {
      return html`${hero} ${this._renderPlant()}
        <div class="placeholder">
          <ha-icon icon="mdi:eye-off-outline"></ha-icon>
          <p>${localize("panel.all_hidden", l)}</p>
          <ha-button @click=${this._requestEdit}>${localize("panel.edit_rooms", l)}</ha-button>
        </div>`;
    }
    const groups = floorGroups(visible, this.hass, this.groupByFloor);
    return html`
      ${hero} ${this._renderPlant()} ${this._renderPills(visible)}
      ${groups.map(
        (g) => html`
          ${
            g.name
              ? html`<h2 class="floor">${g.name} <span class="count">${g.items.length}</span></h2>`
              : nothing
          }
          <div class="grid">
            ${repeat(
              g.items,
              (i) => i.area.area_id,
              (i) =>
                html`<rs-room-card
                  .hass=${this.hass}
                  .area=${i.area}
                  .config=${i.config}
                  .climateEntityCount=${i.climateEntityCount}
                  .tempSensorCount=${i.tempSensorCount}
                  .controlMode=${this.controlMode}
                  .climateControlActive=${this.climateControlActive}
                ></rs-room-card>`,
            )}
          </div>
        `,
      )}
    `;
  }

  /**
   * Compact house hero above the room grid. Renders nothing on installs that do not report the
   * house layer yet, so the dashboard stays unchanged until the backend publishes it.
   */
  private _renderHouseHero() {
    const house = this.house;
    if (!house) return nothing;
    const l = this.hass.language;
    const hold = houseHoldLabel(house, l);
    const source = houseSourceLabel(house, l);
    const detail = houseOverrideActive(house) && hold ? hold : source;
    const block = house.schedule_block;
    const blockText = block ? `${block.from}–${block.to}` : "";
    return html`<ha-card class="house">
      <div class="house-head">
        <span class="house-icon" aria-hidden="true"
          ><ha-icon icon="mdi:home-thermometer"></ha-icon
        ></span>
        <div class="house-text">
          <div class="house-title">${localize("house.title", l)}</div>
          <div class="house-band">${formatHouseTarget(house, this.hass)}</div>
        </div>
      </div>
      <div class="house-meta">
        <span>${localize("house.mode", l)}: <b>${houseModeLabel(house, l)}</b></span>
        <span>${detail}${blockText ? ` · ${blockText}` : ""}</span>
        <span>${followingRoomsLabel(house.following_rooms ?? 0, l)}</span>
      </div>
      <div class="house-actions">
        ${
          houseOverrideActive(house)
            ? html`<ha-button
                .disabled=${this._resumingHouseSchedule}
                @click=${this._resumeHouseSchedule}
                >${localize("house.resume_schedule", l)}</ha-button
              >`
            : nothing
        }
        <button type="button" class="link" @click=${this._openHouseSchedule}>
          ${localize("panel.tab.schedules", l)}
        </button>
        <button type="button" class="link" @click=${this._openHouseSettings}>
          ${localize("house.title", l)}
        </button>
      </div>
    </ha-card>`;
  }

  /** Jump to the Schedules tab, reusing the panel's existing schedule navigation event. */
  private _openHouseSchedule() {
    this.dispatchEvent(
      new CustomEvent("open-schedule-editor", {
        detail: {
          entityId: this.house?.schedule_entity ?? "",
          areaId: "",
          kind: "",
        },
        bubbles: true,
        composed: true,
      }),
    );
  }

  /** Jump to the House settings category, reusing the dashboard's settings navigation event. */
  private _openHouseSettings() {
    this.dispatchEvent(
      new CustomEvent("open-settings", {
        detail: { category: "house" },
        bubbles: true,
        composed: true,
      }),
    );
  }

  /** Clear a temporary house hold and let the panel refresh the rooms list. */
  private async _resumeHouseSchedule() {
    if (this._resumingHouseSchedule) return;
    this._resumingHouseSchedule = true;
    try {
      await this.hass.callWS({ type: "roommind/house/clear_override" });
      this.dispatchEvent(new CustomEvent("room-updated", { bubbles: true, composed: true }));
    } catch (err) {
      // eslint-disable-next-line no-console
      console.debug("[RoomMind] clear house override:", err);
    } finally {
      this._resumingHouseSchedule = false;
    }
  }

  /** Crisp Central HVAC status card: plant action, temperatures, targets, drivers and guards. */
  private _renderPlant() {
    const l = this.hass.language;
    const p = this.centralPlant;
    // Older backends do not report the plant: leave the dashboard unchanged until they do.
    if (!p) return nothing;
    // Before the first coordinator cycle the payload is the stored config block, which has no
    // `configured`/`action`: derive them so the banner does not flash "not configured".
    const configured = p.configured ?? !!p.thermostat_entity;
    const action: PlantAction = p.action ?? "idle";
    const pausedRooms = p.exterior_paused_rooms ?? [];
    if (!configured) {
      return html`<ha-card class="plant unconfigured">
        <div class="plant-head">
          <span class="plant-icon" aria-hidden="true"><ha-icon icon="mdi:hvac"></ha-icon></span>
          <div class="plant-text">
            <div class="plant-title">${localize("plant.title", l)}</div>
            <div class="plant-sub">${localize("plant.not_configured", l)}</div>
          </div>
          <ha-button @click=${this._openPlantSettings}>${localize("plant.set_up", l)}</ha-button>
        </div>
      </ha-card>`;
    }
    const unit = tempUnit(this.hass);
    const fmt = (c: number | null) => (c == null ? "--" : formatTempLocale(c, this.hass, 0, 1));
    const guardKey = p.guard_reason ? GUARD_KEYS[p.guard_reason] : undefined;
    return html`<ha-card class="plant">
      <div class="plant-head">
        <span class="plant-icon ${action}" aria-hidden="true"
          ><ha-icon .icon=${ACTION_ICONS[action]}></ha-icon
        ></span>
        <div class="plant-text">
          <div class="plant-title">${localize(ACTION_KEYS[action], l)}</div>
          <div class="plant-sub">${p.decision_reason || localize("plant.subtitle", l)}</div>
        </div>
        <div class="plant-temps">
          <span>${localize("plant.house_ref", l)} <b>${fmt(p.house_ref_temp)}</b>${unit}</span>
          <span>${localize("plant.thermostat", l)} <b>${fmt(p.thermostat_temp)}</b>${unit}</span>
        </div>
      </div>

      <div class="plant-meta">
        ${this._renderTargets(p, fmt, unit)}
        <span class="driving">
          ${localize("plant.driving_rooms", l)}:
          <b
            >${
              (p.driving_rooms ?? []).length
                ? p.driving_rooms.map((id) => this.hass.areas?.[id]?.name ?? id).join(", ")
                : localize("plant.no_drivers", l)
            }</b
          >
        </span>
      </div>

      ${this._renderChips(p, guardKey, pausedRooms)}
    </ha-card>`;
  }

  private _renderTargets(p: CentralPlantLive, fmt: (c: number | null) => string, unit: string) {
    const l = this.hass.language;
    const band =
      p.commanded_low != null && p.commanded_high != null && p.commanded_low !== p.commanded_high;
    if (band) {
      return html`<span class="target"
        >${localize("plant.fallback_band", l)}
        <b>${fmt(p.commanded_low)}–${fmt(p.commanded_high)} ${unit}</b></span
      >`;
    }
    if (p.commanded_single != null) {
      return html`<span class="target"
        >${localize("plant.nudge", l)} <b>${fmt(p.commanded_single)} ${unit}</b></span
      >`;
    }
    return html`<span class="target"
      >${localize("plant.fallback_band", l)}
      <b>${fmt(p.commanded_low)}–${fmt(p.commanded_high)} ${unit}</b></span
    >`;
  }

  private _renderChips(
    p: CentralPlantLive,
    guardKey: TranslationKey | undefined,
    pausedRooms: string[],
  ) {
    const l = this.hass.language;
    // A standing outdoor limit, so it stays out of the Guard chip: "cooling is off
    // because it is 51 °F outside" is a state of the weather, not a fault.
    const blockedDirections = p.outdoor_blocked ?? [];
    const blockedChip =
      blockedDirections.length && p.outdoor_temp != null
        ? html`<rs-pill icon="mdi:thermometer-off"
            >${localize("plant.chip.outdoor_blocked", l, {
              action: blockedDirections.map((a) => localize(ACTION_KEYS[a], l)).join(", "),
              temp: `${formatTempLocale(p.outdoor_temp, this.hass, 0, 1)} ${tempUnit(this.hass)}`,
            })}</rs-pill
          >`
        : nothing;
    const chips = [
      p.wet_coil_locked
        ? html`<rs-pill icon="mdi:water-alert" tone="warn"
            >${localize("plant.chip.wet_coil", l)}</rs-pill
          >`
        : nothing,
      p.guard_active
        ? html`<rs-pill icon="mdi:shield-lock-outline" tone="warn"
            >${localize("plant.chip.guard", l)}${
              guardKey ? ` · ${localize(guardKey, l)}` : ""
            }</rs-pill
          >`
        : nothing,
      blockedChip,
      pausedRooms.length
        ? html`<rs-pill icon="mdi:window-open-variant" tone="warn"
            >${localize("plant.chip.exterior", l, { count: pausedRooms.length })}</rs-pill
          >`
        : nothing,
      p.plant_override_active
        ? html`<rs-pill icon="mdi:hand-back-right-outline" tone="ok"
            >${localize("plant.chip.override", l)}${
              p.plant_override_heat != null || p.plant_override_cool != null
                ? ` · ${this._overrideText(p)}`
                : ""
            }</rs-pill
          >`
        : nothing,
    ].filter((c) => c !== nothing);
    const clear =
      p.plant_override_active || p.plant_override_heat != null || p.plant_override_cool != null
        ? html`<ha-button .disabled=${this._clearingOverride} @click=${this._clearOverride}
            >${localize("plant.clear_override", l)}</ha-button
          >`
        : nothing;
    if (!chips.length && clear === nothing) return nothing;
    return html`<div class="plant-chips">${chips}${clear}</div>`;
  }

  private _overrideText(p: CentralPlantLive): string {
    const unit = tempUnit(this.hass);
    const fmt = (c: number) => `${formatTempLocale(c, this.hass, 0, 1)} ${unit}`;
    if (p.plant_override_heat != null && p.plant_override_cool != null) {
      return `${fmt(p.plant_override_heat)}–${fmt(p.plant_override_cool)}`;
    }
    const single = p.plant_override_heat ?? p.plant_override_cool;
    return single == null ? "" : fmt(single);
  }

  private _renderPills(visible: AreaInfo[]) {
    const l = this.hass.language;
    const s = this.summary;
    // Rooms RoomMind controls: outdoor areas and rooms without a temperature sensor only measure
    const configured = visible.filter((i) => isControlledZone(i.config));
    const heating = visible.filter((i) => i.config?.live?.mode === "heating").length;
    const cooling = visible.filter((i) => i.config?.live?.mode === "cooling").length;
    const circulating = visible.filter(
      (i) => i.config?.live?.mode === "circulating" || i.config?.live?.fan_circulating,
    ).length;
    const mold = visible.filter(
      (i) =>
        i.config?.live?.mold_risk_level === "warning" ||
        i.config?.live?.mold_risk_level === "critical",
    ).length;
    const level = s.priceEntity ? s.priceStatus?.level : null;
    const vacationUntil =
      s.vacationActive && s.vacationUntil && s.vacationUntil < VACATION_SENTINEL
        ? new Intl.DateTimeFormat(l, { day: "numeric", month: "short" }).format(
            new Date(s.vacationUntil * 1000),
          )
        : null;

    const pills = [
      configured.length
        ? html`<rs-pill icon="mdi:home-thermometer-outline"
            >${localize("panel.pill.rooms", l)} <b>${configured.length}</b></rs-pill
          >`
        : nothing,
      configured.length
        ? html`<rs-pill icon="mdi:fire" .tone=${heating ? "heat" : "default"}
            >${localize("panel.pill.heating", l)} <b>${heating}</b></rs-pill
          >`
        : nothing,
      cooling
        ? html`<rs-pill icon="mdi:snowflake" tone="cool"
            >${localize("panel.pill.cooling", l)} <b>${cooling}</b></rs-pill
          >`
        : nothing,
      circulating
        ? html`<rs-pill icon="mdi:fan"
            >${localize("panel.pill.circulating", l)} <b>${circulating}</b></rs-pill
          >`
        : nothing,
      s.outdoorTemp != null
        ? html`<rs-pill icon="mdi:thermometer"
            >${localize("panel.pill.outdoor", l)}
            <b
              >${formatTempLocale(s.outdoorTemp, this.hass)}&nbsp;${tempUnit(this.hass)}</b
            ></rs-pill
          >`
        : nothing,
      level
        ? html`<rs-pill
            icon="mdi:cash-clock"
            .tone=${level === "cheap" ? "ok" : level === "expensive" ? "warn" : "default"}
            >${localize("panel.pill.price", l)}
            <b>${localize(PRICE_LEVEL_KEYS[level], l)}</b></rs-pill
          >`
        : nothing,
      s.vacationActive
        ? html`<rs-pill icon="mdi:airplane" tone="ok"
            >${
              vacationUntil
                ? localize("panel.pill.vacation_until", l, { date: vacationUntil })
                : localize("panel.stat.vacation", l)
            }</rs-pill
          >`
        : nothing,
      s.presenceEnabled && !s.anyoneHome
        ? html`<rs-pill icon="mdi:home-export-outline">${localize("panel.pill.away", l)}</rs-pill>`
        : nothing,
      mold
        ? html`<rs-pill icon="mdi:water-alert" tone="warn"
            ><b>${mold}</b> ${localize("panel.pill.mold", l)}</rs-pill
          >`
        : nothing,
    ].filter((p) => p !== nothing);
    if (!pills.length) return nothing;
    return html`<div class="pills" role="list" aria-label=${localize("panel.pill.label", l)}>
      ${pills.map((p) => html`<span role="listitem">${p}</span>`)}
    </div>`;
  }

  private _requestEdit() {
    this.dispatchEvent(new CustomEvent("edit-requested", { bubbles: true, composed: true }));
  }

  private _openPlantSettings() {
    this.dispatchEvent(
      new CustomEvent("open-settings", {
        detail: { category: "plant" },
        bubbles: true,
        composed: true,
      }),
    );
  }

  private async _clearOverride() {
    if (this._clearingOverride) return;
    this._clearingOverride = true;
    try {
      await this.hass.callWS({ type: "roommind/plant/clear_override" });
      this.dispatchEvent(new CustomEvent("room-updated", { bubbles: true, composed: true }));
    } catch (err) {
      // eslint-disable-next-line no-console
      console.debug("[RoomMind] clear plant override:", err);
    } finally {
      this._clearingOverride = false;
    }
  }
}

declare global {
  interface HTMLElementTagNameMap {
    "rs-dashboard": RsDashboard;
  }
}
