/**
 * Clipboard utilities with a fallback for non-HTTPS and embedded contexts.
 *
 * The result is awaited on purpose. `navigator.clipboard.writeText` rejects when the page
 * runs in an iframe without the `clipboard-write` permission — which is how the panel is
 * embedded in some shells — and the previous version returned `true` before that rejection
 * arrived, so a blocked copy looked exactly like a successful one.
 */

export async function copyToClipboard(text: string): Promise<boolean> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Blocked (insecure context or a restrictive iframe): fall through to the textarea.
    }
  }
  return copyToClipboardFallback(text);
}

function copyToClipboardFallback(text: string): boolean {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch (err) {
    // eslint-disable-next-line no-console
    console.debug("[RoomMind] clipboard fallback:", err);
  }
  document.body.removeChild(ta);
  return ok;
}
