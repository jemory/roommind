import { describe, expect, test } from "vitest";
import type { RoomConfig, RoomLiveData } from "../types";
import {
  hintText,
  isControlledZone,
  roomHints,
  roomKind,
  roomStateInfo,
  type RoomHintContext,
} from "./room-state";

const ctx: RoomHintContext = {
  climateControlActive: true,
  controlMode: "mpc",
  language: "de",
  displayDelta: (d) => d,
  formatTemp: (c) => c.toFixed(1).replace(".", ","),
  unit: "°C",
  now: new Date(2026, 10, 12, 5, 0).getTime(),
};

function room(live: Partial<RoomLiveData>): RoomConfig {
  return {
    temperature_sensor: "sensor.t",
    live: { mode: "idle", mpc_active: true, mold_risk_level: "ok", ...live } as RoomLiveData,
  } as RoomConfig;
}

const sixAm = new Date(2026, 10, 12, 6, 0).getTime() / 1000;

describe("roomHints pre-heat", () => {
  test("names the target and the start while the room heats ahead", () => {
    const hints = roomHints(
      room({ mode: "heating", preheat: { mode: "heating", target: 21.5, at: sixAm } }),
      "controlled",
      ctx,
    );
    const hint = hints.find((h) => h.id === "preheat");
    expect(hint).toBeDefined();
    expect(hintText(hint!, "de")).toBe("Heizt vor auf 21,5 °C ab 06:00");
  });

  test("pre-cooling has its own text", () => {
    const hints = roomHints(
      room({ mode: "cooling", preheat: { mode: "cooling", target: 24, at: sixAm } }),
      "controlled",
      ctx,
    );
    expect(
      hintText(
        hints.find((h) => h.id === "preheat")!,
        "en",
      ),
    ).toBe("Cooling ahead for 24,0 °C from 06:00");
  });

  test("no hint without a pre-heat or once the room stopped", () => {
    expect(roomHints(room({ mode: "heating" }), "controlled", ctx).map((h) => h.id)).not.toContain(
      "preheat",
    );
    const stale = room({ mode: "idle", preheat: { mode: "heating", target: 21.5, at: sixAm } });
    expect(roomHints(stale, "controlled", ctx).map((h) => h.id)).not.toContain("preheat");
  });
});

describe("roomHints air quality (#267)", () => {
  test("names the CO2 reading and comes right after the mold risk", () => {
    const hints = roomHints(
      room({
        mold_risk_level: "warning",
        window_open: true,
        co2: 1349.6,
        air_quality_bad: true,
        air_quality_reason: "co2",
      }),
      "controlled",
      ctx,
    );
    expect(hints.map((h) => h.id).slice(0, 3)).toEqual(["mold_risk", "air_quality", "window_open"]);
    expect(hintText(hints[1], "de")).toBe("CO₂ 1.350 ppm, bitte lüften");
  });

  test("says when RoomMind ventilates, and VOC has no figure", () => {
    const venting = roomHints(
      room({
        co2: 1100,
        air_quality_bad: true,
        air_quality_reason: "co2",
        ventilation_active: true,
      }),
      "controlled",
      ctx,
    );
    expect(hintText(venting[0], "de")).toBe("CO₂ 1.100 ppm, Lüftung läuft");
    const voc = roomHints(
      room({ voc: 300, air_quality_bad: true, air_quality_reason: "voc" }),
      "sensor",
      ctx,
    );
    expect(voc.map((h) => h.id)).toEqual(["air_quality"]);
    expect(hintText(voc[0], "de")).toBe("Luft belastet, bitte lüften");
  });

  test("no hint while the air is good", () => {
    const hints = roomHints(room({ co2: 900, air_quality_bad: false }), "controlled", ctx);
    expect(hints.map((h) => h.id)).not.toContain("air_quality");
  });
});

describe("Central HVAC room kind", () => {
  test("a room with a temperature sensor is a controlled Thermal Zone without devices", () => {
    const config = {
      area_id: "living",
      temperature_sensor: "sensor.t",
      window_sensors: [],
      live: { mode: "idle", mpc_active: true, mold_risk_level: "ok" },
    } as unknown as RoomConfig;
    expect(isControlledZone(config)).toBe(true);
    expect(roomKind(config, 0)).toBe("controlled");
  });

  test("an outdoor area is never a controlled zone", () => {
    const config = {
      area_id: "garden",
      temperature_sensor: "sensor.out",
      is_outdoor: true,
      window_sensors: [],
      live: { mode: "idle", mpc_active: false, mold_risk_level: "ok" },
    } as unknown as RoomConfig;
    expect(isControlledZone(config)).toBe(false);
    expect(roomKind(config, 0)).toBe("outdoor");
  });

  test("a room without a temperature sensor is not controlled", () => {
    const config = {
      area_id: "hall",
      temperature_sensor: "",
      window_sensors: [],
    } as unknown as RoomConfig;
    expect(isControlledZone(config)).toBe(false);
    expect(roomKind(config, 0)).toBe("empty");
  });
});

describe("roomStateInfo circulating", () => {
  test("reports circulating air with the fan icon", () => {
    const config = {
      area_id: "living",
      temperature_sensor: "sensor.t",
      window_sensors: [],
      live: { mode: "circulating", mpc_active: true, mold_risk_level: "ok" },
    } as unknown as RoomConfig;
    const st = roomStateInfo(config, true);
    expect(st.tone).toBe("circulating");
    expect(st.icon).toBe("mdi:fan");
    expect(st.key).toBe("card.state.circulating");
  });

  test("a fan-only sample without the circulating mode still shows circulation", () => {
    const config = {
      area_id: "living",
      temperature_sensor: "sensor.t",
      window_sensors: [],
      live: { mode: "idle", fan_circulating: true, mpc_active: true, mold_risk_level: "ok" },
    } as unknown as RoomConfig;
    expect(roomStateInfo(config, true).tone).toBe("circulating");
  });
});

describe("roomHints Central HVAC openings and effective temperature", () => {
  test("an open exterior opening is a warning", () => {
    const hints = roomHints(room({ exterior_window_open: true }), "controlled", ctx);
    const hint = hints.find((h) => h.id === "exterior_window_open");
    expect(hint).toBeDefined();
    expect(hint!.severity).toBe("warn");
  });

  test("a closed interior door is an info hint", () => {
    const hints = roomHints(room({ interior_door_open: false }), "controlled", ctx);
    expect(hints.map((h) => h.id)).toContain("interior_door_closed");
  });

  test("an effective temperature far from the reading is named", () => {
    const hints = roomHints(room({ current_temp: 22, effective_temp: 23.1 }), "controlled", ctx);
    const hint = hints.find((h) => h.id === "effective_temp");
    expect(hint).toBeDefined();
    expect(hintText(hint!, "de")).toContain("23,1");
  });

  test("a small humidity correction stays quiet", () => {
    const hints = roomHints(room({ current_temp: 22, effective_temp: 22.1 }), "controlled", ctx);
    expect(hints.map((h) => h.id)).not.toContain("effective_temp");
  });
});
