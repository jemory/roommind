import { LitElement, html, css, nothing } from "lit";
import { customElement, property } from "lit/decorators.js";
import type { HomeAssistant } from "../../types";
import { localize, type TranslationKey } from "../../utils/localize";
import { tokens } from "../../styles/tokens";
import "./../shared/rs-list-row";

/**
 * Solar orientation of the room's exterior glazing. Opt-in refinement of the thermal model:
 * with an azimuth set the model uses the sun actually striking that facade instead of a
 * horizontal-plane estimate; with none set today's behaviour is kept.
 */
@customElement("rs-solar-section")
export class RsSolarSection extends LitElement {
  @property({ attribute: false }) public hass!: HomeAssistant;
  @property({ attribute: false }) public solarAzimuths: number[] = [];
  /** Blind orientations, offered as a one-click prefill (never saved on its own). */
  @property({ attribute: false }) public coverOrientations: Record<string, number> = {};
  @property() public language = "en";

  /** Same compass convention as the covers section (8 directions, deg, 0 = N). */
  private static readonly _DIRECTIONS: Array<{
    shortLabel: TranslationKey;
    longLabel: TranslationKey;
    deg: number;
  }> = [
    { shortLabel: "covers.orientation_N", longLabel: "covers.orientation_N_full", deg: 0 },
    { shortLabel: "covers.orientation_NE", longLabel: "covers.orientation_NE_full", deg: 45 },
    { shortLabel: "covers.orientation_E", longLabel: "covers.orientation_E_full", deg: 90 },
    { shortLabel: "covers.orientation_SE", longLabel: "covers.orientation_SE_full", deg: 135 },
    { shortLabel: "covers.orientation_S", longLabel: "covers.orientation_S_full", deg: 180 },
    { shortLabel: "covers.orientation_SW", longLabel: "covers.orientation_SW_full", deg: 225 },
    { shortLabel: "covers.orientation_W", longLabel: "covers.orientation_W_full", deg: 270 },
    { shortLabel: "covers.orientation_NW", longLabel: "covers.orientation_NW_full", deg: 315 },
  ];

  static styles = [
    tokens,
    css`
      :host {
        display: block;
      }

      .hint {
        margin: var(--rs-space-2) var(--rs-space-4) var(--rs-space-1);
        font-size: var(--rs-font-sm);
        color: var(--secondary-text-color);
        line-height: 1.5;
      }

      .prefill {
        margin: var(--rs-space-1) var(--rs-space-4) var(--rs-space-2);
      }
    `,
  ];

  render() {
    const l = this.language || this.hass?.language || "en";
    const selected = new Set(this.solarAzimuths);
    const blindAzimuths = this._blindAzimuths();
    return html`
      <p class="hint">${localize("solar.hint", l)}</p>
      ${
        blindAzimuths.length
          ? html`<div class="prefill">
              <ha-button @click=${() => this._copyFromBlinds(blindAzimuths)}
                >${localize("solar.copy_from_blinds", l)}</ha-button
              >
            </div>`
          : nothing
      }
      ${RsSolarSection._DIRECTIONS.map(
        (d) =>
          html`<rs-list-row
            switch
            .checked=${selected.has(d.deg)}
            .heading=${localize(d.longLabel, l)}
            .secondary=${localize(d.shortLabel, l)}
            @toggle-changed=${(e: CustomEvent<boolean>) => this._toggle(d.deg, e.detail)}
          ></rs-list-row>`,
      )}
    `;
  }

  /** Unique azimuths of the configured covers, sorted for a stable prefill. */
  private _blindAzimuths(): number[] {
    return [...new Set(Object.values(this.coverOrientations))].sort((a, b) => a - b);
  }

  private _copyFromBlinds(azimuths: number[]) {
    // Only prefill the local field; the parent's normal save path (500 ms debounce) persists it.
    this._emit([...azimuths]);
  }

  private _toggle(deg: number, checked: boolean) {
    const next = new Set(this.solarAzimuths);
    if (checked) next.add(deg);
    else next.delete(deg);
    this._emit([...next].sort((a, b) => a - b));
  }

  private _emit(value: number[]) {
    this.dispatchEvent(
      new CustomEvent("thermal-setting-changed", {
        detail: { key: "solar_azimuths", value },
        bubbles: true,
        composed: true,
      }),
    );
  }
}

declare global {
  interface HTMLElementTagNameMap {
    "rs-solar-section": RsSolarSection;
  }
}
