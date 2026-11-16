/**
 * rs-settings-plant – The single shared Central Plant: thermostat, setpoint nudging, fallback
 * band, fan modes, equipment guards, exterior-opening policy and humidity/wet-coil comfort.
 */
import { html, css, nothing } from "lit";
import { RsSettingsBase } from "./rs-settings-base";
import { customElement, property } from "lit/decorators.js";
import type { ExteriorOpenPolicy, HomeAssistant } from "../../types";
import { localize } from "../../utils/localize";
import { thermostatFanModes } from "../../utils/device-utils";
import { PLANT_LIMITS } from "../../utils/plant-settings";
import "../shared/rs-number-field";
import "../shared/rs-segmented";

@customElement("rs-settings-plant")
export class RsSettingsPlant extends RsSettingsBase {
  @property({ attribute: false }) public hass!: HomeAssistant;
  @property({ type: String }) public thermostatEntity = "";
  @property({ type: String }) public thermostatAreaId = "";
  @property({ type: Number }) public nudgeDeltaC = 1.0;
  @property({ type: Number }) public maxNudgeHeadroomC = 3.0;
  @property({ type: Number }) public fallbackHeatC = 16.0;
  @property({ type: Number }) public fallbackCoolC = 28.0;
  @property({ type: Number }) public minDeadbandC = 1.0;
  @property({ type: String }) public circulateFanMode = "on";
  @property({ type: String }) public idleFanMode = "auto";
  @property({ type: Number }) public minRunMinutes = 10;
  @property({ type: Number }) public minOffMinutes = 5;
  @property({ type: Number }) public minFanMinutes = 10;
  @property({ type: Number }) public changeoverMinutes = 30;
  @property({ type: Boolean }) public followThermostatSetpoint = true;
  @property({ type: Number }) public thermostatOverrideHours = 4;
  @property({ type: String }) public exteriorOpenPolicy: ExteriorOpenPolicy =
    "pause_unless_door_closed";
  @property({ type: Boolean }) public humidityComfortEnabled = true;
  @property({ type: Number }) public dehumidifyRhThreshold = 60;
  @property({ type: Number }) public maxDehumidifyOvercoolC = 0.8;
  @property({ type: Number }) public wetCoilLockoutMinutes = 25;
  @property({ type: Number }) public wetCoilMaxRh = 52;

  render() {
    const l = this.hass.language;
    const configured = !!this.thermostatEntity;
    const fanOptions = thermostatFanModes(this.hass, this.thermostatEntity);
    return html`
      <div class="settings-section first">
        <rs-field
          .label=${localize("plant.thermostat_entity", l)}
          .info=${localize("plant.thermostat_entity_info", l)}
        >
          <ha-entity-picker
            .hass=${this.hass}
            .value=${this.thermostatEntity}
            .includeDomains=${["climate"]}
            .label=${localize("plant.thermostat_entity", l)}
            allow-custom-entity
            @value-changed=${(e: CustomEvent) => {
              const v = (e.detail?.value as string) ?? "";
              if (v !== this.thermostatEntity) this._fire("thermostatEntity", v);
            }}
          ></ha-entity-picker>
        </rs-field>
        <p class="field-hint">${localize("plant.thermostat_entity_hint", l)}</p>
      </div>

      <div class="settings-section">
        <rs-field
          .label=${localize("plant.thermostat_area", l)}
          .info=${localize("plant.thermostat_area_info", l)}
        >
          <rs-segmented
            wrap
            .label=${localize("plant.thermostat_area", l)}
            .options=${[
              { value: "", label: localize("plant.thermostat_area_none", l) },
              ...Object.values(this.hass.areas ?? {}).map((a) => ({
                value: a.area_id,
                label: a.name,
              })),
            ]}
            .value=${this.thermostatAreaId}
            @value-changed=${(e: CustomEvent<{ value: string }>) =>
              this._fire("thermostatAreaId", e.detail.value)}
          ></rs-segmented>
        </rs-field>
      </div>

      ${
        configured
          ? html`
              <div class="settings-section">
                <span class="section-title">${localize("plant.section.fallback", l)}</span>
                <div class="threshold-grid">
                  <rs-number-field
                    .hass=${this.hass}
                    kind="temperature"
                    .label=${localize("plant.fallback_heat", l)}
                    .info=${localize("plant.fallback_heat_info", l)}
                    .min=${PLANT_LIMITS.fallbackHeatC.min}
                    .max=${PLANT_LIMITS.fallbackHeatC.max}
                    .step=${0.5}
                    .value=${this.fallbackHeatC}
                    @value-changed=${(e: CustomEvent<{ value: number }>) =>
                      this._fire("fallbackHeatC", e.detail.value)}
                  ></rs-number-field>
                  <rs-number-field
                    .hass=${this.hass}
                    kind="temperature"
                    .label=${localize("plant.fallback_cool", l)}
                    .info=${localize("plant.fallback_cool_info", l)}
                    .min=${PLANT_LIMITS.fallbackCoolC.min}
                    .max=${PLANT_LIMITS.fallbackCoolC.max}
                    .step=${0.5}
                    .value=${this.fallbackCoolC}
                    @value-changed=${(e: CustomEvent<{ value: number }>) =>
                      this._fire("fallbackCoolC", e.detail.value)}
                  ></rs-number-field>
                </div>
              </div>

              <div class="settings-section">
                <span class="section-title">${localize("plant.section.nudge", l)}</span>
                <div class="threshold-grid">
                  <rs-number-field
                    .hass=${this.hass}
                    kind="temperature-delta"
                    .label=${localize("plant.nudge_delta", l)}
                    .info=${localize("plant.nudge_delta_info", l)}
                    .min=${PLANT_LIMITS.nudgeDeltaC.min}
                    .max=${PLANT_LIMITS.nudgeDeltaC.max}
                    .step=${0.1}
                    .value=${this.nudgeDeltaC}
                    @value-changed=${(e: CustomEvent<{ value: number }>) =>
                      this._fire("nudgeDeltaC", e.detail.value)}
                  ></rs-number-field>
                  <rs-number-field
                    .hass=${this.hass}
                    kind="temperature-delta"
                    .label=${localize("plant.max_nudge_headroom", l)}
                    .info=${localize("plant.max_nudge_headroom_info", l)}
                    .min=${PLANT_LIMITS.maxNudgeHeadroomC.min}
                    .max=${PLANT_LIMITS.maxNudgeHeadroomC.max}
                    .step=${0.5}
                    .value=${this.maxNudgeHeadroomC}
                    @value-changed=${(e: CustomEvent<{ value: number }>) =>
                      this._fire("maxNudgeHeadroomC", e.detail.value)}
                  ></rs-number-field>
                </div>
                <rs-number-field
                  .hass=${this.hass}
                  kind="temperature-delta"
                  .label=${localize("plant.min_deadband", l)}
                  .info=${localize("plant.min_deadband_info", l)}
                  .min=${PLANT_LIMITS.minDeadbandC.min}
                  .max=${PLANT_LIMITS.minDeadbandC.max}
                  .step=${0.5}
                  .value=${this.minDeadbandC}
                  @value-changed=${(e: CustomEvent<{ value: number }>) =>
                    this._fire("minDeadbandC", e.detail.value)}
                ></rs-number-field>
              </div>

              <div class="settings-section">
                <span class="section-title">${localize("plant.section.fans", l)}</span>
                <rs-field
                  .label=${localize("plant.circulate_fan_mode", l)}
                  .info=${localize("plant.circulate_fan_mode_info", l)}
                >
                  <rs-segmented
                    wrap
                    .label=${localize("plant.circulate_fan_mode", l)}
                    .options=${fanOptions}
                    .value=${this.circulateFanMode}
                    @value-changed=${(e: CustomEvent<{ value: string }>) =>
                      this._fire("circulateFanMode", e.detail.value)}
                  ></rs-segmented>
                </rs-field>
                <rs-field
                  .label=${localize("plant.idle_fan_mode", l)}
                  .info=${localize("plant.idle_fan_mode_info", l)}
                >
                  <rs-segmented
                    wrap
                    .label=${localize("plant.idle_fan_mode", l)}
                    .options=${[
                      { value: "auto", label: localize("plant.fan.auto", l) },
                      { value: "off", label: localize("plant.fan.off", l) },
                    ]}
                    .value=${this.idleFanMode}
                    @value-changed=${(e: CustomEvent<{ value: string }>) =>
                      this._fire("idleFanMode", e.detail.value)}
                  ></rs-segmented>
                </rs-field>
                <p class="field-hint">${localize("plant.fan_modes_hint", l)}</p>
              </div>

              <div class="settings-section">
                <span class="section-title">${localize("plant.section.guards", l)}</span>
                <div class="threshold-grid">
                  <rs-number-field
                    .label=${localize("plant.min_run_minutes", l)}
                    .info=${localize("plant.min_run_minutes_info", l)}
                    suffix="min"
                    .min=${PLANT_LIMITS.minRunMinutes.min}
                    .max=${PLANT_LIMITS.minRunMinutes.max}
                    .value=${this.minRunMinutes}
                    @value-changed=${(e: CustomEvent<{ value: number }>) =>
                      this._fire("minRunMinutes", e.detail.value)}
                  ></rs-number-field>
                  <rs-number-field
                    .label=${localize("plant.min_off_minutes", l)}
                    .info=${localize("plant.min_off_minutes_info", l)}
                    suffix="min"
                    .min=${PLANT_LIMITS.minOffMinutes.min}
                    .max=${PLANT_LIMITS.minOffMinutes.max}
                    .value=${this.minOffMinutes}
                    @value-changed=${(e: CustomEvent<{ value: number }>) =>
                      this._fire("minOffMinutes", e.detail.value)}
                  ></rs-number-field>
                  <rs-number-field
                    .label=${localize("plant.min_fan_minutes", l)}
                    .info=${localize("plant.min_fan_minutes_info", l)}
                    suffix="min"
                    .min=${PLANT_LIMITS.minFanMinutes.min}
                    .max=${PLANT_LIMITS.minFanMinutes.max}
                    .value=${this.minFanMinutes}
                    @value-changed=${(e: CustomEvent<{ value: number }>) =>
                      this._fire("minFanMinutes", e.detail.value)}
                  ></rs-number-field>
                  <rs-number-field
                    .label=${localize("plant.changeover_minutes", l)}
                    .info=${localize("plant.changeover_minutes_info", l)}
                    suffix="min"
                    .min=${PLANT_LIMITS.changeoverMinutes.min}
                    .max=${PLANT_LIMITS.changeoverMinutes.max}
                    .value=${this.changeoverMinutes}
                    @value-changed=${(e: CustomEvent<{ value: number }>) =>
                      this._fire("changeoverMinutes", e.detail.value)}
                  ></rs-number-field>
                </div>
              </div>

              <div class="settings-section">
                <span class="section-title">${localize("plant.section.policy", l)}</span>
                <rs-field
                  .label=${localize("plant.exterior_open_policy", l)}
                  .info=${localize("plant.exterior_open_policy_info", l)}
                >
                  <rs-segmented
                    wrap
                    .label=${localize("plant.exterior_open_policy", l)}
                    .options=${[
                      {
                        value: "pause_unless_door_closed",
                        label: localize("plant.exterior_policy.pause_unless_door_closed", l),
                      },
                      {
                        value: "pause_always",
                        label: localize("plant.exterior_policy.pause_always", l),
                      },
                      {
                        value: "exclude_room_only",
                        label: localize("plant.exterior_policy.exclude_room_only", l),
                      },
                    ]}
                    .value=${this.exteriorOpenPolicy}
                    @value-changed=${(e: CustomEvent<{ value: string }>) =>
                      this._fire("exteriorOpenPolicy", e.detail.value)}
                  ></rs-segmented>
                </rs-field>
              </div>

              <div class="settings-section">
                <rs-list-row
                  switch
                  wrap
                  .heading=${localize("plant.follow_thermostat_setpoint", l)}
                  .secondary=${localize("plant.follow_thermostat_setpoint_hint", l)}
                  .checked=${this.followThermostatSetpoint}
                  @toggle-changed=${(e: CustomEvent<boolean>) =>
                    this._fire("followThermostatSetpoint", e.detail)}
                ></rs-list-row>
                ${
                  this.followThermostatSetpoint
                    ? html`<rs-number-field
                        .label=${localize("plant.thermostat_override_hours", l)}
                        .info=${localize("plant.thermostat_override_hours_info", l)}
                        suffix="h"
                        .min=${PLANT_LIMITS.thermostatOverrideHours.min}
                        .max=${PLANT_LIMITS.thermostatOverrideHours.max}
                        .step=${0.5}
                        .value=${this.thermostatOverrideHours}
                        @value-changed=${(e: CustomEvent<{ value: number }>) =>
                          this._fire("thermostatOverrideHours", e.detail.value)}
                      ></rs-number-field>`
                    : nothing
                }
              </div>

              <div class="settings-section">
                <span class="section-title">${localize("plant.section.humidity", l)}</span>
                <rs-list-row
                  switch
                  wrap
                  .heading=${localize("plant.humidity_comfort_enabled", l)}
                  .secondary=${localize("plant.humidity_comfort_enabled_hint", l)}
                  .checked=${this.humidityComfortEnabled}
                  @toggle-changed=${(e: CustomEvent<boolean>) =>
                    this._fire("humidityComfortEnabled", e.detail)}
                ></rs-list-row>
                ${
                  this.humidityComfortEnabled
                    ? html`
                        <div class="threshold-grid">
                          <rs-number-field
                            .label=${localize("plant.dehumidify_rh_threshold", l)}
                            .info=${localize("plant.dehumidify_rh_threshold_info", l)}
                            suffix="%"
                            .min=${PLANT_LIMITS.dehumidifyRhThreshold.min}
                            .max=${PLANT_LIMITS.dehumidifyRhThreshold.max}
                            .step=${1}
                            .value=${this.dehumidifyRhThreshold}
                            @value-changed=${(e: CustomEvent<{ value: number }>) =>
                              this._fire("dehumidifyRhThreshold", e.detail.value)}
                          ></rs-number-field>
                          <rs-number-field
                            .hass=${this.hass}
                            kind="temperature-delta"
                            .label=${localize("plant.max_dehumidify_overcool", l)}
                            .info=${localize("plant.max_dehumidify_overcool_info", l)}
                            .min=${PLANT_LIMITS.maxDehumidifyOvercoolC.min}
                            .max=${PLANT_LIMITS.maxDehumidifyOvercoolC.max}
                            .step=${0.1}
                            .value=${this.maxDehumidifyOvercoolC}
                            @value-changed=${(e: CustomEvent<{ value: number }>) =>
                              this._fire("maxDehumidifyOvercoolC", e.detail.value)}
                          ></rs-number-field>
                          <rs-number-field
                            .label=${localize("plant.wet_coil_lockout_minutes", l)}
                            .info=${localize("plant.wet_coil_lockout_minutes_info", l)}
                            suffix="min"
                            .min=${PLANT_LIMITS.wetCoilLockoutMinutes.min}
                            .max=${PLANT_LIMITS.wetCoilLockoutMinutes.max}
                            .value=${this.wetCoilLockoutMinutes}
                            @value-changed=${(e: CustomEvent<{ value: number }>) =>
                              this._fire("wetCoilLockoutMinutes", e.detail.value)}
                          ></rs-number-field>
                          <rs-number-field
                            .label=${localize("plant.wet_coil_max_rh", l)}
                            .info=${localize("plant.wet_coil_max_rh_info", l)}
                            suffix="%"
                            .min=${PLANT_LIMITS.wetCoilMaxRh.min}
                            .max=${PLANT_LIMITS.wetCoilMaxRh.max}
                            .step=${1}
                            .value=${this.wetCoilMaxRh}
                            @value-changed=${(e: CustomEvent<{ value: number }>) =>
                              this._fire("wetCoilMaxRh", e.detail.value)}
                          ></rs-number-field>
                        </div>
                      `
                    : nothing
                }
              </div>
            `
          : nothing
      }
    `;
  }

  static styles = [
    RsSettingsBase.settingsBaseStyles,
    css`
      .field-hint {
        display: block;
        margin-top: var(--rs-space-2);
        color: var(--secondary-text-color);
        font-size: var(--rs-font-sm);
        line-height: 1.4;
      }

      rs-segmented {
        margin-top: var(--rs-space-1);
      }
    `,
  ];
}

declare global {
  interface HTMLElementTagNameMap {
    "rs-settings-plant": RsSettingsPlant;
  }
}
