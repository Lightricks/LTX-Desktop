import type { VideoGenerationModelSpecsResponse } from "./video-generation-model-specs.ts";

/**
 * One-line empty-grid explainer for local generation. Callers wire it as a branch
 * after loading and fetch-error, gated on local mode — an empty cloud grid is not
 * a local hardware problem. Copy must not mention the API (same rule as
 * `local-generation-unsupported.ts`).
 */

function hasAnyLocalCell(specs: VideoGenerationModelSpecsResponse): boolean {
  return specs.local_models.some(
    (item) =>
      Object.keys(item.spec.supported_resolutions_durations ?? {}).length > 0 ||
      Object.keys(item.spec.a2v_supported_resolutions_durations ?? {}).length > 0,
  );
}

export function localUnavailableMessage(
  specs: VideoGenerationModelSpecsResponse | null,
  // OS platform (`window.electronAPI.platform`): a pool-refused Mac must
  // hear pool/allowance wording, never "graphics card".
  platform: string | null,
): string | null {
  // A missing response belongs to the loading/error branches, and a non-empty
  // grid needs no explanation even if a reason field is present. Brave keeps
  // its bets in the grid, so the server's map is the whole answer.
  if (specs === null || hasAnyLocalCell(specs)) return null;
  const reason = specs.local_unavailable_reason ?? null;
  switch (reason) {
    case "ram_shortfall": {
      const required = specs.local_unavailable_required_ram_gb ?? null;
      const detected = specs.local_unavailable_detected_ram_gb ?? null;
      if (required !== null && detected !== null) {
        return `Local generation needs ${required} GB of system RAM. This machine has ${detected} GB.`;
      }
      if (required !== null) {
        return `Local generation needs ${required} GB of system RAM.`;
      }
      return "This machine can't run any of the available generation settings.";
    }
    case "stream_over":
      if (platform === "darwin") {
        return "Memory available for video on this Mac can't run any of the available settings.";
      }
      return "This machine's graphics card can't run any of the available settings.";
    case "unsupported":
      return "This machine can't run local generation.";
    default:
      // Null, absent, or a future reason the wire type doesn't know yet: the
      // generic line. Deliberately no assertNever — the reason union widens
      // with the backend vocabulary, and a display helper must degrade,
      // never throw from render.
      return "This machine can't run any of the available generation settings.";
  }
}
