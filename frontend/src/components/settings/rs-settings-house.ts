/**
 * rs-settings-house – House layer settings: one comfort and eco pair, the mode
 * following rooms inherit, the house schedule, preheat and price shift, plus the
 * list of rooms that follow the house (each can be made independent).
 *
 * Presentational: edits are fired as `setting-changed` `{ key: "house", value }`
 * for the orchestrator to validate and save as a partial `{ house: {...} }`.
 * An "Independent" action fires `room-independent` `{ areaId }`.
 */
import { html, css, nothing } from "lit";
import { customElement, property } from "lit/decorators.js";
import type { ClimateMode, HomeAssistant, HouseConfig, HouseLive, RoomConfig } from "../../types";
import { localize } from "../../utils/localize";
import { HOUSE_PRICE_SHIFT_MAX, HOUSE_TEMP_BOUNDS } from "../../utils/house-settings";
import { HOUSE_MODE_OPTIONS, followingRoomsLabel } from "../../utils/house-state";
import { RsSettingsBase } from "./rs-settings-base";
import "../shared/rs-number-field";
import "../shared/rs-segmented";
import "../shared/rs-icon-button";

@customElement("rs-settings-house")
export class RsSettingsHouse extends RsSettingsBase {
  @property({ attribute: false }) public hass!: HomeAssistant;
  /** The stored house block (all fields the page edits). */
  @property({ attribute: false }) public house!: HouseConfig;
  /** Live house block from rooms/list, for the authoritative following count. */
  @property({ attribute: false }) public houseLive?: HouseLive | null;
  /** All rooms, used for the following-rooms list. */
  @property({ attribute: false }) public rooms: Record<string, RoomConfig> = {};
  /** Validation error from the orchestrator's last save attempt, if any. */
  @property({ type: String }) public error = "";
  @property({ type: Boolean }) public busy = false;

  render() {
    const l = this.hass.language;
    const h = this.house;
    const followers = this._followingRooms();
    const count = this.houseLive?.following_rooms ?? followers.length;

    return html`
      <div class="settings-section first">
        <rs-list-row
          switch
          wrap
          .heading=${localize("house.enabled", l)}
          .secondary=${localize("house.enable_warning", l, { count })}
          .checked=${h.enabled}
          @toggle-changed=${(e: CustomEvent<boolean>) => this._set({ enabled: e.detail })}
        ></rs-list-row>
        <p class="following-count">${followingRoomsLabel(count, l)}</p>
        ${this.error ? html`<p class="rs-alert" role="alert">${this.error}</p>` : nothing}
      </div>

      <div class="settings-section">
        <h3 class="section-title">${localize("house.comfort", l)}</h3>
        <div class="threshold-grid">
          <rs-number-field
            .hass=${this.hass}
            kind="temperature"
            .label=${localize("schedule_editor.heat_below", l)}
            .min=${HOUSE_TEMP_BOUNDS.min}
            .max=${HOUSE_TEMP_BOUNDS.max}
            .value=${h.heat_c}
            allow-empty
            @value-changed=${(e: CustomEvent<{ value: number | null }>) =>
              this._set({ heat_c: e.detail.value })}
          ></rs-number-field>
          <rs-number-field
            .hass=${this.hass}
            kind="temperature"
            .label=${localize("schedule_editor.cool_above", l)}
            .min=${HOUSE_TEMP_BOUNDS.min}
            .max=${HOUSE_TEMP_BOUNDS.max}
            .value=${h.cool_c}
            allow-empty
            @value-changed=${(e: CustomEvent<{ value: number | null }>) =>
              this._set({ cool_c: e.detail.value })}
          ></rs-number-field>
        </div>
      </div>

      <div class="settings-section">
        <h3 class="section-title">${localize("house.eco", l)}</h3>
        <div class="threshold-grid">
          <rs-number-field
            .hass=${this.hass}
            kind="temperature"
            .label=${localize("schedule_editor.heat_below", l)}
            .min=${HOUSE_TEMP_BOUNDS.min}
            .max=${HOUSE_TEMP_BOUNDS.max}
            .value=${h.eco_heat_c}
            @value-changed=${(e: CustomEvent<{ value: number }>) =>
              this._set({ eco_heat_c: e.detail.value })}
          ></rs-number-field>
          <rs-number-field
            .hass=${this.hass}
            kind="temperature"
            .label=${localize("schedule_editor.cool_above", l)}
            .min=${HOUSE_TEMP_BOUNDS.min}
            .max=${HOUSE_TEMP_BOUNDS.max}
            .value=${h.eco_cool_c}
            @value-changed=${(e: CustomEvent<{ value: number }>) =>
              this._set({ eco_cool_c: e.detail.value })}
          ></rs-number-field>
        </div>
      </div>

      <div class="settings-section">
        <span class="field-label">${localize("house.mode", l)}</span>
        <rs-segmented
          wrap
          .label=${localize("house.mode", l)}
          .options=${HOUSE_MODE_OPTIONS.map((option) => ({
            value: option.value,
            label: localize(option.key, l),
            icon: option.icon,
          }))}
          .value=${h.climate_mode}
          @value-changed=${(e: CustomEvent<{ value: string }>) =>
            this._set({ climate_mode: e.detail.value as ClimateMode })}
        ></rs-segmented>
      </div>

      <div class="settings-section">
        <h3 class="section-title">${localize("house.schedule", l)}</h3>
        ${this._renderSchedules()}
        <ha-entity-picker
          .hass=${this.hass}
          .includeDomains=${["schedule"]}
          .label=${localize("detail.schedule.add", l)}
          @value-changed=${(e: CustomEvent<{ value?: string }>) => {
            const value = e.detail?.value;
            if (value && !h.schedules.some((s) => s.entity_id === value)) {
              this._set({ schedules: [...h.schedules, { entity_id: value }] });
            }
            const picker = e.target as HTMLElement & { value: string };
            picker.value = "";
          }}
        ></ha-entity-picker>
        <ha-entity-picker
          .hass=${this.hass}
          .includeDomains=${["select", "input_select"]}
          .label=${localize("schedule.selector_label", l)}
          .value=${h.schedule_selector_entity}
          @value-changed=${(e: CustomEvent<{ value?: string }>) =>
            this._set({ schedule_selector_entity: e.detail?.value ?? "" })}
        ></ha-entity-picker>
      </div>

      <div class="settings-section">
        <rs-list-row
          switch
          wrap
          .heading=${localize("house.preheat", l)}
          .checked=${h.preheat_enabled}
          @toggle-changed=${(e: CustomEvent<boolean>) => this._set({ preheat_enabled: e.detail })}
        ></rs-list-row>
      </div>

      <div class="settings-section">
        <span class="field-label">${localize("house.price_shift", l)}</span>
        <div class="slider-row">
          <input
            type="range"
            aria-label=${localize("house.price_shift", l)}
            min="0"
            max=${HOUSE_PRICE_SHIFT_MAX}
            step="0.1"
            .value=${String(h.price_shift)}
            @change=${(e: Event) => {
              const value = parseFloat((e.target as HTMLInputElement).value);
              if (!isNaN(value)) this._set({ price_shift: value });
            }}
          />
          <span class="slider-value">${h.price_shift.toFixed(1)} K</span>
        </div>
      </div>

      <div class="settings-section">
        <h3 class="section-title">${localize("house.follow_house", l)}</h3>
        ${this._renderFollowing()}
      </div>
    `;
  }

  private _renderSchedules() {
    const l = this.hass.language;
    const schedules = this.house.schedules;
    if (!schedules.length) {
      return html`<p class="note">${localize("house.schedule_none", l)}</p>`;
    }
    return html`<div class="schedule-list">
      ${schedules.map(
        (schedule, index) =>
          html`<div class="schedule-row">
            <ha-icon icon="mdi:calendar-clock" aria-hidden="true"></ha-icon>
            <span class="schedule-name">
              ${this.hass.states[schedule.entity_id]?.attributes?.friendly_name ?? schedule.entity_id}
            </span>
            <rs-icon-button
              size="sm"
              icon="mdi:close"
              .label=${localize("common.remove_named", l, { name: schedule.entity_id })}
              @click=${() =>
                this._set({
                  schedules: schedules.filter((_, i) => i !== index),
                })}
            ></rs-icon-button>
          </div>`,
      )}
    </div>`;
  }

  private _renderFollowing() {
    const l = this.hass.language;
    const followers = this._followingRooms();
    if (!followers.length) return html`<p class="note">${localize("house.schedule_none", l)}</p>`;
    return html`<div class="person-list">
      ${followers.map((room) => {
        const name = room.display_name || this.hass.areas?.[room.area_id]?.name || room.area_id;
        return html`<div class="person-row">
          <ha-icon icon="mdi:home" aria-hidden="true"></ha-icon>
          <span class="person-name">${name}</span>
          <ha-button
            appearance="outlined"
            .disabled=${this.busy}
            @click=${() =>
              this.dispatchEvent(
                new CustomEvent("room-independent", {
                  detail: { areaId: room.area_id },
                  bubbles: true,
                  composed: true,
                }),
              )}
            >${localize("house.independent", l)}</ha-button
          >
        </div>`;
      })}
    </div>`;
  }

  private _followingRooms(): RoomConfig[] {
    return Object.values(this.rooms).filter(
      (room) => room.follow_house !== false && !room.is_outdoor,
    );
  }

  private _set(changes: Partial<HouseConfig>) {
    this._fire("house", { ...this.house, ...changes });
  }

  static styles = [
    RsSettingsBase.settingsBaseStyles,
    css`
      .following-count {
        margin: var(--rs-space-2) 0 0;
        font-size: 13px;
        color: var(--secondary-text-color);
      }
      .rs-alert {
        margin: var(--rs-space-3) 0 0;
        padding: var(--rs-space-2) var(--rs-space-3);
        border-radius: var(--rs-radius-sm);
        background: color-mix(in srgb, var(--rs-err) 12%, transparent);
        color: var(--rs-err-text);
        font-size: 13px;
      }
      .schedule-list,
      .person-list {
        display: flex;
        flex-direction: column;
        gap: 2px;
        margin-bottom: var(--rs-space-3);
      }
      .schedule-row,
      .person-row {
        display: flex;
        align-items: center;
        gap: var(--rs-space-3);
        padding: 4px 4px 4px 12px;
        border-radius: var(--rs-radius-sm);
        background: var(--rs-fill);
        --mdc-icon-size: 18px;
      }
      .schedule-row ha-icon,
      .person-row ha-icon {
        color: var(--secondary-text-color);
      }
      .schedule-name,
      .person-name {
        flex: 1;
        min-width: 0;
        font-size: var(--rs-font-md);
        font-weight: 500;
        overflow-wrap: anywhere;
      }
      .slider-row {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .slider-row input[type="range"] {
        flex: 1;
        accent-color: var(--primary-color);
      }
      .slider-value {
        min-width: 3.5em;
        text-align: right;
        font-size: 13px;
        color: var(--secondary-text-color);
      }
    `,
  ];
}

declare global {
  interface HTMLElementTagNameMap {
    "rs-settings-house": RsSettingsHouse;
  }
}
