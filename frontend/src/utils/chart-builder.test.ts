import { describe, expect, test } from "vitest";
import type { AnalyticsDataPoint, HomeAssistant } from "../types";
import { buildChartSeries, type ChartBuildContext } from "./chart-builder";

/**
 * ECharts throws "id duplicates: <id>" out of setOption and draws nothing at all, so a
 * repeated series id blanks the whole analytics chart while the axes and the legend still
 * render. The per-panel loop used to push a second `circulating_events` on top of the one
 * the temperature plot already had.
 */

const hass = {
  config: { unit_system: { temperature: "°C" } },
  language: "en",
} as unknown as HomeAssistant;

const BASE_TS = 1_794_981_600; // seconds

function point(
  offsetSeconds: number,
  overrides: Partial<AnalyticsDataPoint> = {},
): AnalyticsDataPoint {
  return {
    ts: BASE_TS + offsetSeconds,
    room_temp: 21.5,
    outdoor_temp: 9.6,
    target_temp: 20.5,
    mode: "idle",
    predicted_temp: null,
    window_open: false,
    humidity: 44,
    fan_circulating: false,
    ...overrides,
  } as AnalyticsDataPoint;
}

function context(overrides: Partial<ChartBuildContext> = {}): ChartBuildContext {
  return {
    hass,
    language: "en",
    chartAnchor: 1_795_021_277_000,
    rangeStart: 1_794_981_600_000,
    rangeEnd: 1_795_021_277_000,
    ...overrides,
  };
}

function ids(series: Array<Record<string, unknown>>): string[] {
  return series.map((s) => s.id).filter((id): id is string => typeof id === "string");
}

describe("buildChartSeries", () => {
  test("every series id is unique, so ECharts can apply the option", () => {
    // One sample of each run type, plus outdoor and humidity so the sub-panels exist too:
    // the duplicate used to appear only once a sub-panel was shown.
    const points = [
      point(0, { mode: "heating" }),
      point(600, { mode: "cooling", window_open: true }),
      point(1200, { mode: "circulating", fan_circulating: true }),
      point(1800),
    ];

    const series = buildChartSeries(points, context());
    const seriesIds = ids(series);

    expect(seriesIds.length).toBeGreaterThan(0);
    expect(new Set(seriesIds).size).toBe(seriesIds.length);
  });

  test("circulation is drawn once, on the temperature plot", () => {
    const points = [point(0, { mode: "circulating" }), point(600, { mode: "circulating" })];

    const series = buildChartSeries(points, context());
    const circulating = series.filter((s) => s.id === "circulating_events");

    expect(circulating).toHaveLength(1);
    expect(circulating[0].panel).toBeUndefined();
  });
});
