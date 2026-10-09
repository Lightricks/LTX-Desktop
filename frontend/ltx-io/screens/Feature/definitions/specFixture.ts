import type { VideoGenerationModelSpecsResponse } from "../../../../lib/video-generation-model-specs";

type LocalSpec = VideoGenerationModelSpecsResponse["local_models"][number]["spec"];
type ResolutionCell = LocalSpec["supported_resolutions_durations"][string];

const LOCAL_ASPECTS = ["21:9", "16:9", "3:2", "4:3", "1:1", "4:5", "9:16"] as const;
const LOCAL_FPS = ["24", "25", "48", "50"] as const;

function cell(
  durations: ResolutionCell["fps_to_durations"][string],
  aspectRatios: ResolutionCell["aspect_ratios"],
): ResolutionCell {
  return {
    fps_to_durations: Object.fromEntries(LOCAL_FPS.map((fps) => [fps, [...durations]])),
    aspect_ratios: aspectRatios,
  };
}

function localSpec(displayName: string): LocalSpec {
  return {
    display_name: displayName,
    supported_resolutions_durations: {
      "540p": cell([2, 3, 4, 5, 6, 8, 10, 20], [...LOCAL_ASPECTS]),
      "720p": cell([2, 3, 4, 5, 6, 8, 10, 20], [...LOCAL_ASPECTS]),
      "1080p": cell([2, 3, 4, 5, 10], [...LOCAL_ASPECTS]),
    },
    a2v_supported_resolutions_durations: {
      "270p": cell([5, 6, 8, 10, 20], [...LOCAL_ASPECTS]),
      "360p": cell([5, 6, 8, 10, 20], [...LOCAL_ASPECTS]),
      "540p": cell([5, 6, 8, 10, 20], [...LOCAL_ASPECTS]),
      "720p": cell([5, 6, 8, 10, 20], [...LOCAL_ASPECTS]),
      "1080p": cell([5, 10], [...LOCAL_ASPECTS]),
    },
  };
}

const LTX_25_FAST_SPEC = localSpec("LTX 2.5 Fast");
const LTX_23_FAST_SPEC = localSpec("LTX 2.3 Fast");

/** Mirrors the backend LOCAL catalog shape for tests. */
export function specFixture(): VideoGenerationModelSpecsResponse {
  return {
    api_models: [],
    local_models: [
      {
        pipeline: "fast",
        spec: LTX_25_FAST_SPEC,
      },
    ],
    downloaded_local_models: [
      {
        model: "ltx-2.5-fast",
        pipeline: "fast",
        spec: LTX_25_FAST_SPEC,
      },
    ],
    active_offering: "ltx-2.5-fast",
    low_performance_machine: false,
  };
}

export function specFixtureBothOfferings(
  activeDisplayName: "LTX 2.5 Fast" | "LTX 2.3 Fast" = "LTX 2.3 Fast",
): VideoGenerationModelSpecsResponse {
  const activeSpec =
    activeDisplayName === "LTX 2.3 Fast" ? LTX_23_FAST_SPEC : LTX_25_FAST_SPEC;
  return {
    api_models: [],
    local_models: [{ pipeline: "fast", spec: activeSpec }],
    downloaded_local_models: [
      {
        model: "ltx-2.5-fast",
        pipeline: "fast",
        spec: LTX_25_FAST_SPEC,
      },
      {
        model: "ltx-2.3-fast",
        pipeline: "fast",
        spec: LTX_23_FAST_SPEC,
      },
    ],
    active_offering:
      activeDisplayName === "LTX 2.3 Fast" ? "ltx-2.3-fast" : "ltx-2.5-fast",
    low_performance_machine: false,
  };
}
