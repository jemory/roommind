import { LitElement, html, css, nothing } from "lit";
import { customElement, property } from "lit/decorators.js";
import type { HomeAssistant, ClimateMode } from "../../types";
import { localize, type TranslationKey } from "../../utils/localize";
import { formatTempLocale } from "../../utils/temperature";
import { HOUSE_MODE_OPTIONS } from "../../utils/house-state";
import { tokens } from "../../styles/tokens";
import { tempRowStyles } from "../../styles/temp-row-styles";
import { tempRow } from "../shared/temp-row";
import "../shared/rs-info-icon";

/** The house-inherited values a following room displays read-only. */
export interface InheritedTargets {
  comfortHeat: number | null;
  comfortCool: number | null;
  ecoHeat: number | null;
  ecoCool: number | null;
  climateMode: ClimateMode;
}

/**
 * Comfort and eco targets of a room (content of the temperatures dialog): a level each, with
 * the heat and/or cool side the climate mode uses. Fires `comfort-heat-changed`,
 * `comfort-cool-changed`, `eco-heat-changed`, `eco-cool-changed` with `{ value }` in °C.
 *
 * When `following` is true the values come from the house layer: they are read-only, an
 * inherited hint is shown and `customize-room` asks the parent to make the room independent
 * (seeded from those values).
 */
@customElement("rs-target-temps")
export class RsTargetTemps extends LitElement {
  @property({ attribute: false }) public hass!: HomeAssistant;
  @property({ type: Number }) public comfortHeat = 21.0;
  @property({ type: Number }) public comfortCool = 24.0;
  @property({ type: Number }) public ecoHeat = 17.0;
  @property({ type: Number }) public ecoCool = 27.0;
  @property({ type: String }) public climateMode: ClimateMode = "auto";
  /** True while the room follows the house: values below are read-only. */
  @property({ type: Boolean }) public following = false;
  /** House values to show (and seed from) while following. */
  @property({ attribute: false }) public inherited: InheritedTargets | null = null;

  static styles = [
    tokens,
    tempRowStyles,
    css`
      :host {
        display: block;
      }

      .level + .level {
        margin-top: var(--rs-space-5);
      }

      .level-title {
        display: flex;
        align-items: center;
        gap: var(--rs-space-1);
        margin: 0;
        font-size: var(--rs-font-md);
        font-weight: 500;
      }

      .inherited-hint {
        margin: 0 0 var(--rs-space-4);
        padding: var(--rs-space-2) var(--rs-space-3);
        border-radius: var(--rs-radius-sm, 6px);
        background: var(--rs-fill);
        color: var(--secondary-text-color);
        font-size: var(--rs-font-sm);
        line-height: 1.5;
      }

      .readout {
        display: flex;
        justify-content: space-between;
        gap: var(--rs-space-3);
        padding: 6px 0;
        font-size: var(--rs-font-sm);
        color: var(--secondary-text-color);
        border-bottom: 1px solid var(--divider-color);
      }

      .readout b {
        color: var(--primary-text-color);
        font-weight: 500;
        font-variant-numeric: tabular-nums;
      }

      .customize {
        margin-top: var(--rs-space-5);
      }
    `,
  ];

  render() {
    if (this.following) return this._renderInherited();
    const l = this.hass.language;
    const heat = this.climateMode !== "cool_only";
    const cool = this.climateMode !== "heat_only";
    const level = (
      title: string,
      info: string,
      heatC: number,
      coolC: number,
      onHeat: (v: number) => void,
      onCool: (v: number) => void,
    ) => {
      const row = (
        side: "heat" | "cool",
        key: TranslationKey,
        celsius: number,
        cb: (v: number) => void,
      ) => {
        const label = localize(key, l);
        return tempRow({
          hass: this.hass,
          side,
          label,
          ariaLabel: `${title}, ${label}`,
          celsius,
          onChange: cb,
        });
      };
      return html`<section class="level">
        <h4 class="level-title">${title}<rs-info-icon .text=${info}></rs-info-icon></h4>
        ${heat ? row("heat", "schedule_editor.heat_below", heatC, onHeat) : nothing}
        ${cool ? row("cool", "schedule_editor.cool_above", coolC, onCool) : nothing}
      </section>`;
    };
    return html`
      ${level(
        localize("schedule.column_comfort", l),
        localize("schedule.comfort_hint", l),
        this.comfortHeat,
        this.comfortCool,
        this._onComfortHeat,
        this._onComfortCool,
      )}
      ${level(
        localize("schedule.column_eco", l),
        localize("schedule.eco_hint", l),
        this.ecoHeat,
        this.ecoCool,
        this._onEcoHeat,
        this._onEcoCool,
      )}
    `;
  }

  /** Read-only house values with the inherited hint and a customize action. */
  private _renderInherited() {
    const l = this.hass.language;
    const v = this.inherited;
    const mode = v?.climateMode ?? this.climateMode;
    const modeLabelKey =
      HOUSE_MODE_OPTIONS.find((entry) => entry.value === mode)?.key ?? HOUSE_MODE_OPTIONS[0].key;
    const showHeat = mode !== "cool_only";
    const showCool = mode !== "heat_only";
    const unit = (c: number | null) => (c == null ? "—" : formatTempLocale(c, this.hass, 0, 1));
    return html`
      <p class="inherited-hint">${localize("house.inherited_hint", l)}</p>
      <section class="level">
        <h4 class="level-title">${localize("schedule.column_comfort", l)}</h4>
        ${
          showHeat
            ? html`<div class="readout">
                <span>${localize("schedule_editor.heat_below", l)}</span>
                <b>${unit(v?.comfortHeat ?? null)}</b>
              </div>`
            : nothing
        }
        ${
          showCool
            ? html`<div class="readout">
                <span>${localize("schedule_editor.cool_above", l)}</span>
                <b>${unit(v?.comfortCool ?? null)}</b>
              </div>`
            : nothing
        }
      </section>
      <section class="level">
        <h4 class="level-title">${localize("schedule.column_eco", l)}</h4>
        ${
          showHeat
            ? html`<div class="readout">
                <span>${localize("schedule_editor.heat_below", l)}</span>
                <b>${unit(v?.ecoHeat ?? null)}</b>
              </div>`
            : nothing
        }
        ${
          showCool
            ? html`<div class="readout">
                <span>${localize("schedule_editor.cool_above", l)}</span>
                <b>${unit(v?.ecoCool ?? null)}</b>
              </div>`
            : nothing
        }
      </section>
      <section class="level">
        <div class="readout">
          <span>${localize("house.mode", l)}</span>
          <b>${localize(modeLabelKey, l)}</b>
        </div>
      </section>
      <div class="customize">
        <ha-button @click=${this._onCustomize}> ${localize("house.customize", l)} </ha-button>
      </div>
    `;
  }

  private _onCustomize() {
    this.dispatchEvent(new CustomEvent("customize-room", { bubbles: true, composed: true }));
  }

  private _fire(name: string, value: number) {
    this.dispatchEvent(new CustomEvent(name, { detail: { value }, bubbles: true, composed: true }));
  }

  private _onComfortHeat = (val: number) => {
    this._fire("comfort-heat-changed", val);
    if (this.comfortCool < val) this._fire("comfort-cool-changed", val);
  };

  private _onComfortCool = (val: number) => {
    this._fire("comfort-cool-changed", val);
    if (this.comfortHeat > val) this._fire("comfort-heat-changed", val);
  };

  private _onEcoHeat = (val: number) => {
    this._fire("eco-heat-changed", val);
    if (this.ecoCool < val) this._fire("eco-cool-changed", val);
  };

  private _onEcoCool = (val: number) => {
    this._fire("eco-cool-changed", val);
    if (this.ecoHeat > val) this._fire("eco-heat-changed", val);
  };
}

declare global {
  interface HTMLElementTagNameMap {
    "rs-target-temps": RsTargetTemps;
  }
}
