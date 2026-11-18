/**
 * Analytics data export utilities (CSV).
 */
import type { AnalyticsData } from "../types";
import type { RoomConfig, HomeAssistant } from "../types";

const COLUMNS = [
  "room_temp",
  "outdoor_temp",
  "target_temp",
  "mode",
  "predicted_temp",
  "window_open",
  "heating_power",
  "solar_irradiance",
  "blind_position",
  "cover_reason",
  "device_setpoint",
  "humidity",
  "heat_target",
  "cool_target",
  "target_reason",
  "mold_boost",
  "price",
  "price_level",
  "price_shift",
] as const;

/** A CSV field: empty for null, quoted when it holds a comma, quote or line break. */
function field(value: unknown): string {
  if (value == null) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function buildCsvString(data: AnalyticsData): string | null {
  const points = [...data.history, ...data.detail];
  if (points.length === 0) return null;

  const header = ["timestamp", "datetime", ...COLUMNS].join(",");
  const rows = points.map((p) =>
    [p.ts, new Date(p.ts * 1000).toISOString(), ...COLUMNS.map((c) => p[c])].map(field).join(","),
  );

  return [header, ...rows].join("\n");
}

/**
 * Offer `content` as a file download. Returns false only when the browser threw — a
 * sandboxed iframe without `allow-downloads` blocks the click without throwing, so the
 * caller still has to say something when nothing appears.
 *
 * The anchor is attached to the document before clicking: a detached one is ignored by
 * some browsers, and the object URL is revoked on a later task because revoking it
 * synchronously can cancel a download that has not started yet.
 */
export function downloadString(content: string, filename: string, mimeType: string): boolean {
  try {
    const blob = new Blob([content], { type: `${mimeType};charset=utf-8` });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30_000);
    return true;
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn("[RoomMind] download failed:", err);
    return false;
  }
}

export function buildExportFilename(
  hass: HomeAssistant,
  rooms: Record<string, RoomConfig>,
  selectedRoom: string,
  rangeStart: number,
  rangeEnd: number,
  suffix: string,
  ext: string,
): string {
  const area = hass?.areas?.[selectedRoom];
  const roomConfig = rooms[selectedRoom];
  const name = (roomConfig?.display_name || area?.name || selectedRoom)
    .replace(/\s+/g, "_")
    .toLowerCase();
  if (suffix) {
    return `roommind_${suffix}_${name}.${ext}`;
  }
  const from = new Date(rangeStart).toISOString().slice(0, 10);
  const to = new Date(rangeEnd).toISOString().slice(0, 10);
  return `roommind_${name}_${from}_${to}.${ext}`;
}
