import { LitElement, html, css } from "lit";
import { customElement, property } from "lit/decorators.js";
import type { HomeAssistant, RoomConfig } from "../../types";
import { localize } from "../../utils/localize";
import { tokens } from "../../styles/tokens";
import "./../shared/rs-list-row";

type ConnectionKey = "neighbour_rooms" | "open_rooms";

/**
 * Room connections feeding the thermal model. Both lists are opt-in: a doorway is a normal air
 * path, an open-concept connection dominates the room's air exchange, and leaving both empty
 * keeps today's whole-house behaviour.
 */
@customElement("rs-connections-section")
export class RsConnectionsSection extends LitElement {
  @property({ attribute: false }) public hass!: HomeAssistant;
  @property() public areaId = "";
  /** All rooms in the house, keyed by area_id. */
  @property({ attribute: false }) public rooms: Record<string, RoomConfig> = {};
  @property({ attribute: false }) public neighbourRooms: Set<string> = new Set();
  @property({ attribute: false }) public openRooms: Set<string> = new Set();
  @property() public language = "en";

  static styles = [
    tokens,
    css`
      :host {
        display: block;
      }

      .group + .group {
        margin-top: var(--rs-space-3);
      }

      .group-title {
        padding: var(--rs-space-2) var(--rs-space-4) 0;
        font-size: var(--rs-font-sm);
        font-weight: 500;
        color: var(--primary-text-color);
      }

      .hint {
        margin: 0 var(--rs-space-4) var(--rs-space-1);
        font-size: var(--rs-font-sm);
        color: var(--secondary-text-color);
        line-height: 1.5;
      }

      .empty {
        margin: var(--rs-space-3) var(--rs-space-4);
        font-size: var(--rs-font-sm);
        color: var(--secondary-text-color);
      }
    `,
  ];

  render() {
    const l = this.language || this.hass?.language || "en";
    const others = this._otherRooms();
    if (!others.length) {
      return html`<p class="empty">${localize("connections.no_other_rooms", l)}</p>`;
    }
    return html`
      <div class="group">
        <div class="group-title">${localize("connections.neighbour_title", l)}</div>
        <p class="hint">${localize("connections.neighbour_hint", l)}</p>
        ${others.map((room) => this._roomRow(room, "neighbour_rooms", this.neighbourRooms))}
      </div>
      <div class="group">
        <div class="group-title">${localize("connections.open_title", l)}</div>
        <p class="hint">${localize("connections.open_hint", l)}</p>
        ${others.map((room) => this._roomRow(room, "open_rooms", this.openRooms))}
      </div>
    `;
  }

  /** Other rooms in the house, from the list the panel already has (no new websocket call). */
  private _otherRooms(): Array<{ id: string; name: string }> {
    return Object.entries(this.rooms)
      .filter(([id]) => id !== this.areaId)
      .map(([id, config]) => ({
        id,
        name: config.display_name || id,
      }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }

  private _roomRow(room: { id: string; name: string }, key: ConnectionKey, current: Set<string>) {
    const checked = current.has(room.id);
    return html`<rs-list-row
      switch
      .checked=${checked}
      .heading=${room.name}
      @toggle-changed=${(e: CustomEvent<boolean>) => this._toggle(key, room.id, current, e.detail)}
    ></rs-list-row>`;
  }

  private _toggle(key: ConnectionKey, id: string, current: Set<string>, checked: boolean) {
    const next = new Set(current);
    if (checked) next.add(id);
    else next.delete(id);
    this.dispatchEvent(
      new CustomEvent("thermal-setting-changed", {
        detail: { key, value: [...next] },
        bubbles: true,
        composed: true,
      }),
    );
  }
}

declare global {
  interface HTMLElementTagNameMap {
    "rs-connections-section": RsConnectionsSection;
  }
}
