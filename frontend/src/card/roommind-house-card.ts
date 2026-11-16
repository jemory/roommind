/**
 * Dashboard card for the RoomMind house layer: the whole-home target, mode and
 * schedule that following rooms inherit. Same look and controls as the room card.
 *
 * This file is part of the card bundle (see ../card.ts): it must not import any
 * panel component, or the two bundles would define the same custom elements.
 */
import { LitElement, html, css, nothing, type PropertyValues } from "lit";
import { property, state } from "lit/decorators.js";
import type { ClimateMode, HomeAssistant, HouseLive } from "../types";
import { localize, uiLanguage, type TranslationKey } from "../utils/localize";
import { errorText } from "../utils/errors";
import {
  formatTempLocale,
  tempUnit,
  toCelsius,
  toDisplay,
  usesFahrenheit,
} from "../utils/temperature";
import { MAX_TARGET_TEMP_C, MIN_TARGET_TEMP_C } from "../utils/temperature";
import { HOUSE_DEFAULTS } from "../utils/house-settings";
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
  type HousePreset,
} from "../utils/house-state";
import { modeStyles } from "../styles/shared-mode-styles";
import { tokens } from "../styles/tokens";
import { refreshRooms, setFeedHass, subscribeRooms, type RoomsFeed } from "./rooms-feed";

export interface HouseCardConfig {
  type: string;
  compact?: boolean;
}

/** The stepper pair shown for the house target; °C internally, like the room card. */
interface HouseTemps {
  heat: number;
  cool: number;
}

const TEMPS_SAVE_DELAY_MS = 800;

const PRESETS: { value: Exclude<HousePreset, "none">; key: TranslationKey; icon: string }[] = [
  { value: "comfort", key: "override.comfort", icon: "mdi:sofa" },
  { value: "eco", key: "override.eco", icon: "mdi:leaf" },
  { value: "manual", key: "override.custom", icon: "mdi:thermometer" },
];

export class RoomMindHouseCard extends LitElement {
  @property({ attribute: false }) public hass!: HomeAssistant;

  @state() private _config?: HouseCardConfig;
  @state() private _feed: RoomsFeed | null = null;
  @state() private _feedError = "";
  @state() private _tempsDraft: HouseTemps | null = null;
  @state() private _modeDraft: ClimateMode | null = null;
  @state() private _error = "";

  private _unsubscribe?: () => void;
  private _tempsTimer?: ReturnType<typeof setTimeout>;
  private _tempsEdit = 0;

  public static getConfigForm() {
    return {
      schema: [{ name: "compact", selector: { boolean: {} } }],
      computeLabel: (_schema: { name: string }) =>
        localize("roomcard.editor.compact", uiLanguage()),
    };
  }

  public static getStubConfig(): HouseCardConfig {
    return { type: "custom:roommind-house-card" };
  }

  public setConfig(config: HouseCardConfig): void {
    if (!config) throw new Error("Invalid configuration");
    if (this._config && config !== this._config) {
      this._flushTemps();
      this._resetDrafts();
    }
    this._config = config;
  }

  public getCardSize(): number {
    return this._config?.compact ? 4 : 7;
  }

  public getGridOptions() {
    return this._config?.compact ? { columns: 6, min_columns: 6 } : { columns: 12, min_columns: 9 };
  }

  connectedCallback(): void {
    super.connectedCallback();
    if (this.hass) setFeedHass(this.hass);
    this._unsubscribe = subscribeRooms((feed, error) => {
      this._feed = feed;
      this._feedError = error;
    });
  }

  disconnectedCallback(): void {
    super.disconnectedCallback();
    this._unsubscribe?.();
    this._unsubscribe = undefined;
    this._flushTemps();
  }

  protected willUpdate(changed: PropertyValues<this>): void {
    if (changed.has("hass") && this.hass) setFeedHass(this.hass);
  }

  private get _lang(): string {
    return this.hass?.language ?? "en";
  }

  private get _house(): HouseLive | null {
    return this._feed?.house ?? null;
  }

  private _temps(house: HouseLive | null): HouseTemps {
    if (this._tempsDraft) return this._tempsDraft;
    const band = houseTarget(house);
    return {
      heat: band.low ?? house?.heat_c ?? HOUSE_DEFAULTS.heat_c ?? 21,
      cool: band.high ?? house?.cool_c ?? HOUSE_DEFAULTS.cool_c ?? 24,
    };
  }

  private _mode(house: HouseLive | null): ClimateMode {
    return this._modeDraft ?? house?.climate_mode ?? "auto";
  }

  private _resetDrafts(): void {
    this._tempsDraft = null;
    this._modeDraft = null;
    this._error = "";
  }

  // ---- Rendering ----

  render() {
    if (!this._config || !this.hass) return nothing;
    if (!this._feed) {
      return this._renderMessage(
        this._feedError || localize("roomcard.loading", this._lang),
        this._feedError ? "error" : "",
      );
    }
    const house = this._house;
    if (!house || !house.enabled || !house.active) {
      return this._renderMessage(houseSourceLabel(house, this._lang));
    }
    return html`
      <ha-card class=${this._config.compact ? "compact" : ""}>
        <div class="accent mode-other"></div>
        <div class="content">
          ${this._renderHeader(house)} ${this._renderStatus(house)}
          <div class="controls">${this._renderPresets(house)} ${this._renderMode(house)}</div>
          ${this._renderTemps(house)}
          <div class="error" role="status" aria-live="polite">${this._error}</div>
        </div>
      </ha-card>
    `;
  }

  private _renderMessage(text: string, cls = "") {
    return html`
      <ha-card class=${this._config?.compact ? "compact" : ""}>
        <div class="content">
          <div class="header">
            <span class="name">${localize("housecard.name", this._lang)}</span>
          </div>
          <div class="hint ${cls}">${text}</div>
        </div>
      </ha-card>
    `;
  }

  private _renderHeader(house: HouseLive) {
    const source = houseSourceLabel(house, this._lang);
    return html`
      <div class="header">
        <span class="name">${localize("housecard.name", this._lang)}</span>
        <span class="mode-pill mode-other" title=${source}>
          <span class="mode-dot"></span>
          ${source}
        </span>
      </div>
    `;
  }

  private _renderStatus(house: HouseLive) {
    const l = this._lang;
    const ref = this._feed?.houseRefTemp;
    const hold = houseHoldLabel(house, l);
    const block = house.schedule_block;
    const schedule = block ? `${block.from}–${block.to}` : "";
    return html`
      <div class="temps">
        <span class="current">
          ${ref != null ? formatTempLocale(ref, this.hass) : "--"}<span class="unit"
            >${tempUnit(this.hass)}</span
          >
        </span>
        <div class="target-block">
          <span class="target">
            ${localize("card.target", l)}
            <strong>${formatHouseTarget(house, this.hass)}</strong>
          </span>
          ${block ? html`<span class="schedule">${localize("house.schedule", l)} ${schedule}</span>` : nothing}
        </div>
      </div>
      ${
        houseOverrideActive(house)
          ? html`<div class="override-status ${housePreset(house)}">
              <ha-icon icon="mdi:timer-outline" aria-hidden="true"></ha-icon>
              <span>${hold ?? localize("house.hold_until", l)}</span>
            </div>`
          : nothing
      }
      <div class="hint following">${followingRoomsLabel(house.following_rooms, l)}</div>
    `;
  }

  /** Comfort / eco / manual / none, written as a house hold or cleared back to the schedule. */
  private _renderPresets(house: HouseLive) {
    const l = this._lang;
    const selected = housePreset(house);
    return html`
      <section class="preset-section">
        <h4 id="house-preset-label">${localize("detail.override.label", l)}</h4>
        <div class="segments" role="group" aria-labelledby="house-preset-label">
          ${PRESETS.map(
            (p) => html`
              <button
                class="segment ${p.value} ${selected === p.value ? "active" : ""}"
                aria-pressed=${selected === p.value}
                @click=${() => void this._setPreset(house, p.value)}
              >
                <ha-icon icon=${p.icon} aria-hidden="true"></ha-icon>
                <span>${localize(p.key, l)}</span>
              </button>
            `,
          )}
          <button
            class="segment none ${selected === "none" ? "active" : ""}"
            aria-pressed=${selected === "none"}
            @click=${() => void this._setPreset(house, "none")}
          >
            <ha-icon icon="mdi:calendar-clock" aria-hidden="true"></ha-icon>
            <span>${localize("house.resume_schedule", l)}</span>
          </button>
        </div>
      </section>
    `;
  }

  private _renderMode(house: HouseLive) {
    const l = this._lang;
    const current = this._mode(house);
    return html`
      <section class="mode-section">
        <div class="section-head">
          <h4 title=${houseModeLabel(house, l)}>${localize("house.mode", l)}</h4>
        </div>
        <div class="segments">
          ${HOUSE_MODE_OPTIONS.map(
            (m) => html`
              <button
                class="segment ${current === m.value ? "active" : ""}"
                aria-pressed=${current === m.value}
                aria-label=${localize(m.key, l)}
                title=${localize(m.key, l)}
                @click=${() => current !== m.value && void this._saveMode(m.value)}
              >
                <ha-icon icon=${m.icon}></ha-icon>
                <span>${localize(m.key, l)}</span>
              </button>
            `,
          )}
        </div>
      </section>
    `;
  }

  private _renderTemps(house: HouseLive) {
    const l = this._lang;
    const mode = this._mode(house);
    const t = this._temps(house);
    const heat = localize("mode.heating", l);
    const cool = localize("mode.cooling", l);
    return html`
      <section class="temps-section">
        <h4>${localize("roomcard.temperatures", l)}</h4>
        <div class="temp-grid ${mode === "auto" ? "auto" : "single"}">
          ${
            mode === "auto"
              ? html`
                  <span class="corner"></span>
                  <span class="col-head"><ha-icon icon="mdi:fire"></ha-icon>${heat}</span>
                  <span class="col-head"><ha-icon icon="mdi:snowflake"></ha-icon>${cool}</span>
                `
              : nothing
          }
          <span class="row-label">${localize("house.comfort", l)}</span>
          ${
            mode !== "cool_only"
              ? this._stepper(
                  `${localize("house.comfort", l)} ${heat}`,
                  t.heat,
                  (v) => this._editTemps(house, "heat", v),
                  mode === "auto" ? "mdi:fire" : undefined,
                )
              : nothing
          }
          ${
            mode !== "heat_only"
              ? this._stepper(
                  `${localize("house.comfort", l)} ${cool}`,
                  t.cool,
                  (v) => this._editTemps(house, "cool", v),
                  mode === "auto" ? "mdi:snowflake" : undefined,
                )
              : nothing
          }
        </div>
      </section>
    `;
  }

  private _stepper(
    label: string,
    valueC: number,
    onChange: (v: number) => void,
    sideIcon?: string,
  ) {
    const l = this._lang;
    return html`
      <div class="stepper" role="group" aria-label=${label}>
        <button
          class="step"
          aria-label=${localize("roomcard.lower", l, { name: label })}
          aria-disabled=${valueC <= MIN_TARGET_TEMP_C ? "true" : "false"}
          @click=${() => valueC > MIN_TARGET_TEMP_C && onChange(this._step(valueC, -1))}
        >
          <ha-icon icon="mdi:minus"></ha-icon>
        </button>
        <span class="value" aria-live="polite"
          >${
            sideIcon
              ? html`<ha-icon class="side-icon" icon=${sideIcon} aria-hidden="true"></ha-icon>`
              : nothing
          }${formatTempLocale(valueC, this.hass, 0, 1)}
          ${tempUnit(this.hass)}</span
        >
        <button
          class="step"
          aria-label=${localize("roomcard.raise", l, { name: label })}
          aria-disabled=${valueC >= MAX_TARGET_TEMP_C ? "true" : "false"}
          @click=${() => valueC < MAX_TARGET_TEMP_C && onChange(this._step(valueC, 1))}
        >
          <ha-icon icon="mdi:plus"></ha-icon>
        </button>
      </div>
    `;
  }

  /** One display step (0.5 °C / 1 °F), snapped to the step grid like the room card. */
  private _step(valueC: number, dir: 1 | -1): number {
    const step = usesFahrenheit(this.hass) ? 1 : 0.5;
    const grid = toDisplay(valueC, this.hass) / step;
    const index = dir > 0 ? Math.floor(grid + 1e-6) + 1 : Math.ceil(grid - 1e-6) - 1;
    const next = toCelsius(index * step, this.hass);
    return Math.min(MAX_TARGET_TEMP_C, Math.max(MIN_TARGET_TEMP_C, next));
  }

  // ---- Actions ----

  private _editTemps(_house: HouseLive, key: keyof HouseTemps, value: number): void {
    const next = { ...this._temps(this._house), [key]: value };
    if (next.cool <= next.heat) {
      if (key === "heat") next.cool = value;
      else next.heat = value;
    }
    this._tempsDraft = next;
    this._tempsEdit += 1;
    if (this._tempsTimer) clearTimeout(this._tempsTimer);
    this._tempsTimer = setTimeout(() => void this._saveTemps(), TEMPS_SAVE_DELAY_MS);
  }

  private _flushTemps(): void {
    if (!this._tempsTimer) return;
    clearTimeout(this._tempsTimer);
    void this._saveTemps();
  }

  private async _saveTemps(): Promise<void> {
    this._tempsTimer = undefined;
    const t = this._tempsDraft;
    if (!t) return;
    const edit = this._tempsEdit;
    this._error = "";
    try {
      await this.hass.callWS({
        type: "roommind/house/save",
        heat_c: t.heat,
        cool_c: t.cool,
      });
      await refreshRooms(true);
    } catch (err) {
      this._error = errorText(err, localize("roomcard.error_save", this._lang), this._lang);
    }
    if (edit === this._tempsEdit) this._tempsDraft = null;
  }

  private async _saveMode(mode: ClimateMode): Promise<void> {
    this._error = "";
    this._modeDraft = mode;
    try {
      await this.hass.callWS({ type: "roommind/house/save", climate_mode: mode });
      await refreshRooms(true);
    } catch (err) {
      this._error = errorText(err, localize("roomcard.error_save", this._lang), this._lang);
    }
    this._modeDraft = null;
  }

  /** Presets write the matching override pair; "none" (and "Resume schedule") clears it. */
  private async _setPreset(
    house: HouseLive,
    preset: Exclude<HousePreset, "none"> | "none",
  ): Promise<void> {
    this._error = "";
    if (preset === "none") {
      try {
        await this.hass.callWS({ type: "roommind/house/clear_override" });
        await refreshRooms(true);
      } catch (err) {
        this._error = errorText(err, localize("override.error_clear", this._lang), this._lang);
      }
      return;
    }
    const mode = this._mode(house);
    const band = houseTarget(house);
    const heat = band.low ?? house.heat_c ?? HOUSE_DEFAULTS.heat_c ?? 21;
    const cool = band.high ?? house.cool_c ?? HOUSE_DEFAULTS.cool_c ?? 24;
    let overrideHeat: number | null;
    let overrideCool: number | null;
    if (preset === "comfort") {
      overrideHeat = heat;
      overrideCool = cool;
    } else if (preset === "eco") {
      overrideHeat = house.eco_heat_c;
      overrideCool = house.eco_cool_c;
    } else {
      // Manual keeps the currently held pair, or the comfort pair when none is held.
      overrideHeat = house.override_heat ?? heat;
      overrideCool = house.override_cool ?? cool;
    }
    if (mode === "heat_only") overrideCool = null;
    if (mode === "cool_only") overrideHeat = null;
    try {
      await this.hass.callWS({
        type: "roommind/house/save",
        override_heat: overrideHeat,
        override_cool: overrideCool,
        override_type: preset === "comfort" ? "boost" : preset === "eco" ? "eco" : "custom",
      });
      await refreshRooms(true);
    } catch (err) {
      this._error = errorText(err, localize("override.error_set", this._lang), this._lang);
    }
  }

  static styles = [
    tokens,
    modeStyles,
    css`
      :host {
        display: block;
      }

      ha-card {
        position: relative;
        overflow: hidden;
        height: 100%;
        box-sizing: border-box;
        container-type: inline-size;
      }

      .accent {
        position: absolute;
        inset: 0 auto 0 0;
        width: 4px;
        background: var(--disabled-text-color, #bdbdbd);
      }

      .content {
        padding: 16px 16px 12px 20px;
      }

      .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        min-width: 0;
      }

      .name {
        font-size: 16px;
        font-weight: 500;
        color: var(--primary-text-color);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        min-width: 0;
        min-height: 44px;
        line-height: 44px;
      }

      .mode-pill {
        flex-shrink: 0;
        gap: 5px;
        font-size: 12px;
        padding: 3px 10px;
        border-radius: 12px;
      }

      .mode-dot {
        width: 7px;
        height: 7px;
      }

      .temps {
        display: flex;
        align-items: flex-end;
        gap: 12px;
        flex-wrap: wrap;
      }

      .current {
        font-size: 40px;
        font-weight: 300;
        line-height: 1;
        color: var(--primary-text-color);
        font-variant-numeric: tabular-nums;
      }

      .unit {
        font-size: 18px;
        color: var(--secondary-text-color);
        margin-left: 2px;
      }

      .target-block {
        display: flex;
        flex-direction: column;
        gap: 2px;
        font-size: 13px;
        color: var(--secondary-text-color);
        padding-bottom: 3px;
      }

      .target strong {
        font-weight: 500;
        color: var(--primary-text-color);
        font-variant-numeric: tabular-nums;
      }

      .schedule {
        font-variant-numeric: tabular-nums;
      }

      .override-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-top: 12px;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 13px;
        font-variant-numeric: tabular-nums;
        color: var(--rs-warn-text);
        background: color-mix(in srgb, var(--rs-warn) 12%, transparent);
      }

      .override-status.eco {
        color: var(--rs-ok-text);
        background: color-mix(in srgb, var(--rs-ok) 12%, transparent);
      }

      .override-status.manual,
      .override-status.custom {
        color: var(--rs-cool-text);
        background: color-mix(in srgb, var(--rs-cool) 12%, transparent);
      }

      .override-status ha-icon {
        --mdc-icon-size: 16px;
      }

      .hint {
        margin-top: 10px;
        font-size: 13px;
        color: var(--secondary-text-color);
      }

      .hint.error,
      .error {
        color: var(--rs-err-text);
        font-size: 13px;
      }

      .error:not(:empty) {
        margin-top: 10px;
      }

      section {
        margin-top: 16px;
        padding-top: 12px;
        border-top: 1px solid var(--divider-color, rgba(0, 0, 0, 0.12));
      }

      h4 {
        margin: 0 0 8px;
        font-size: 13px;
        font-weight: 500;
        color: var(--secondary-text-color);
      }

      .section-head {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 4px;
      }

      .section-head h4 {
        margin: 0;
      }

      button {
        font: inherit;
        color: inherit;
        touch-action: manipulation;
      }

      button:focus-visible {
        outline: 2px solid var(--primary-color);
        outline-offset: 2px;
      }

      .segments {
        display: grid;
        grid-auto-flow: column;
        grid-auto-columns: minmax(0, 1fr);
        gap: 8px;
      }

      .segment {
        min-height: 56px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 2px;
        padding: 6px 4px;
        border: 1px solid var(--divider-color, rgba(0, 0, 0, 0.12));
        border-radius: 10px;
        background: transparent;
        color: var(--primary-text-color);
        font-size: 12px;
        line-height: 1.2;
        text-align: center;
        cursor: pointer;
        min-width: 0;
        overflow-wrap: break-word;
        hyphens: auto;
        transition:
          background-color 0.15s,
          border-color 0.15s;
      }

      .segment ha-icon {
        --mdc-icon-size: 18px;
        flex-shrink: 0;
      }

      .segment:hover:not(:disabled) {
        background: rgba(var(--rgb-primary-text-color, 0, 0, 0), 0.04);
      }

      .segment.active {
        border-color: var(--primary-color);
        background: rgba(var(--rgb-primary-color, 3, 169, 244), 0.1);
        color: var(--primary-color);
      }

      .temp-grid {
        display: grid;
        align-items: center;
        gap: 8px 12px;
      }

      .temp-grid.single {
        grid-template-columns: 1fr auto;
      }

      .temp-grid.auto {
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      }

      .temp-grid.auto .corner {
        display: none;
      }

      .temp-grid.auto .row-label {
        grid-column: 1 / -1;
        margin-bottom: -4px;
      }

      @container (max-width: 339px) {
        .temp-grid.auto {
          grid-template-columns: minmax(0, 1fr);
        }

        .temp-grid.auto .col-head {
          display: none;
        }

        .temp-grid.auto .side-icon {
          display: inline-block;
        }
      }

      @container (min-width: 440px) {
        .temp-grid.auto {
          grid-template-columns: auto 1fr 1fr;
        }

        .temp-grid.auto .corner {
          display: block;
        }

        .temp-grid.auto .row-label {
          grid-column: auto;
          margin-bottom: 0;
        }
      }

      .col-head {
        display: flex;
        align-items: center;
        gap: 4px;
        font-size: 12px;
        color: var(--secondary-text-color);
      }

      .col-head ha-icon {
        --mdc-icon-size: 14px;
      }

      .row-label {
        font-size: 14px;
        color: var(--primary-text-color);
      }

      .stepper {
        min-width: 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid var(--divider-color, rgba(0, 0, 0, 0.12));
        border-radius: 22px;
      }

      .step {
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: none;
        border-radius: 50%;
        background: transparent;
        color: var(--primary-text-color);
        cursor: pointer;
      }

      .step ha-icon {
        --mdc-icon-size: 18px;
      }

      .step:hover:not([aria-disabled="true"]) {
        background: rgba(var(--rgb-primary-text-color, 0, 0, 0), 0.06);
      }

      .step[aria-disabled="true"] {
        opacity: 0.35;
        cursor: default;
      }

      .side-icon {
        display: none;
        --mdc-icon-size: 14px;
        margin-right: 2px;
        color: var(--secondary-text-color);
      }

      .value {
        flex: 1;
        min-width: 48px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        white-space: nowrap;
        text-align: center;
        font-size: 15px;
        font-variant-numeric: tabular-nums;
        color: var(--primary-text-color);
      }

      /* Compact: smaller status block and icon-only segments. */
      .compact .content {
        padding: 10px 12px 8px 16px;
      }

      .compact .name {
        font-size: 15px;
        line-height: 1.25;
        white-space: normal;
        overflow-wrap: break-word;
        flex-shrink: 0;
        max-width: 100%;
      }

      .compact .current {
        font-size: 28px;
      }

      .compact .unit {
        font-size: 14px;
      }

      .compact .override-status {
        margin-top: 6px;
      }

      .compact h4,
      .compact .segment span {
        position: absolute;
        width: 1px;
        height: 1px;
        overflow: hidden;
        clip-path: inset(50%);
        white-space: nowrap;
      }

      .compact .controls {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0 16px;
      }

      .compact .controls section {
        flex: 0 1 auto;
        margin-top: 8px;
        padding-top: 0;
        border-top: none;
      }

      .compact .segments {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }

      .compact .segment {
        flex: 0 0 44px;
        min-height: 44px;
        padding: 0;
      }

      .compact .temps-section {
        margin-top: 8px;
        padding-top: 8px;
      }

      @media (prefers-reduced-motion: reduce) {
        .segment {
          transition: none;
        }
      }
    `,
  ];
}

const TAG = "roommind-house-card";

// HA swaps window.customElements for a scoped-registry polyfill while it starts; extra
// modules load in parallel with it. Defining after <home-assistant> lands and guarding
// with get() (for a resource loaded twice) keeps the card discoverable. Same as the room card.
void customElements.whenDefined("home-assistant").then(() => {
  if (!window.customElements.get(TAG)) window.customElements.define(TAG, RoomMindHouseCard);
});

declare global {
  interface HTMLElementTagNameMap {
    "roommind-house-card": RoomMindHouseCard;
  }
}
