import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type {
  VideoGenerationDuration,
  VideoGenerationModelSpecsResponse,
} from "../../../../lib/video-generation-model-specs.ts";
import { resolveFeatureFormValues, resolveFeatureSeedValues } from "../types.ts";
import { specFixture, specFixtureBothOfferings } from "./specFixture.ts";
import {
  AUDIO_TO_VIDEO_DEFAULTS,
  AUDIO_TO_VIDEO_EXAMPLE_PROMPT,
  audioToVideoDefinition,
  fromGeneration,
  toCreateBody,
  validateAudioToVideo,
  type AudioToVideoContext,
  type AudioToVideoValues,
} from "./audioToVideo.ts";

const audioAsset = { assetId: "audio-1" };
const startAsset = { assetId: "start-1" };
const seed = {
  audio: { assetId: "seed-audio" },
  startFrame: { assetId: "seed-image" },
};

const LOCAL_ASPECTS = ["21:9", "16:9", "3:2", "4:3", "1:1", "4:5", "9:16"] as const;

function a2vCell(durations: VideoGenerationDuration[]) {
  return {
    fps_to_durations: { "24": durations },
    aspect_ratios: [...LOCAL_ASPECTS],
  };
}

/** Local Fast with an A2V matrix at 540p/24 — the only cell A2V ever asks for. */
function a2vSpecs(
  durations: VideoGenerationDuration[] = [5, 6, 8, 10, 20],
): VideoGenerationModelSpecsResponse {
  const spec = {
    display_name: "LTX 2.5 Fast",
    supported_resolutions_durations: {
      "540p": a2vCell([5, 6, 8, 10, 20]),
    },
    a2v_supported_resolutions_durations: {
      "540p": a2vCell(durations),
    },
  };
  return {
    api_models: [],
    local_models: [
      {
        pipeline: "fast",
        spec,
      },
    ],
    downloaded_local_models: [
      {
        model: "ltx-2.5-fast" as const,
        pipeline: "fast",
        spec,
      },
    ],
  };
}

function context(
  overrides: Partial<AudioToVideoContext> = {},
): AudioToVideoContext {
  return {
    specs: a2vSpecs(),
    seed,
    audioDurationSeconds: 7.2,
    audioDurationPending: false,
    ...overrides,
  };
}

function values(
  overrides: Partial<AudioToVideoValues> = {},
): AudioToVideoValues {
  return {
    ...AUDIO_TO_VIDEO_DEFAULTS,
    audio: audioAsset,
    startFrame: startAsset,
    ...overrides,
  };
}

function aspectOptions(
  formValues: AudioToVideoValues,
  formContext: AudioToVideoContext = context(),
) {
  const field = audioToVideoDefinition
    .form(formValues, formContext)
    .fields.find((candidate) => candidate.dataKey === "aspectRatio");
  return field?.kind === "options" ? field.options : undefined;
}

describe("audioToVideo definition", () => {
  it("defaults to Fast, Auto aspect, and empty media slots", () => {
    assert.deepEqual(AUDIO_TO_VIDEO_DEFAULTS, {
      startFrame: null,
      prompt: AUDIO_TO_VIDEO_EXAMPLE_PROMPT,
      audio: null,
      model: "ltx-2.5-fast",
      resolution: "540p",
      aspectRatio: "auto",
    });
    assert.deepEqual(audioToVideoDefinition.defaults, AUDIO_TO_VIDEO_DEFAULTS);
  });

  it("renders start frame, prompt, audio, model, resolution, then aspect — no duration/fps", () => {
    const fields = audioToVideoDefinition.form(values(), context()).fields;
    assert.deepEqual(
      fields.map((field) => ({
        kind: field.kind,
        id: field.id,
        dataKey: field.dataKey,
      })),
      [
        { kind: "image-asset", id: "startFrame", dataKey: "startFrame" },
        { kind: "textarea", id: "prompt", dataKey: "prompt" },
        { kind: "audio-asset", id: "audio", dataKey: "audio" },
        { kind: "options", id: "model", dataKey: "model" },
        { kind: "options", id: "resolution", dataKey: "resolution" },
        { kind: "options", id: "aspectRatio", dataKey: "aspectRatio" },
      ],
    );
    const modelField = fields.find((field) => field.dataKey === "model");
    assert.deepEqual(
      modelField?.kind === "options"
        ? modelField.options.map((option) => option.value)
        : undefined,
      ["ltx-2.5-fast"],
    );
  });

  it("form shows Auto aspect only when startFrame is set", () => {
    const specsContext = context({ specs: specFixture() });
    const ratios = [
      { value: "16:9", label: "16:9" },
      { value: "9:16", label: "9:16" },
      {
        value: "21:9",
        label: "21:9",
        warning:
          "The model was trained on 16:9 and 9:16, other ratios may produce lower quality results.",
      },
      {
        value: "3:2",
        label: "3:2",
        warning:
          "The model was trained on 16:9 and 9:16, other ratios may produce lower quality results.",
      },
      {
        value: "4:3",
        label: "4:3",
        warning:
          "The model was trained on 16:9 and 9:16, other ratios may produce lower quality results.",
      },
      {
        value: "1:1",
        label: "1:1",
        warning:
          "The model was trained on 16:9 and 9:16, other ratios may produce lower quality results.",
      },
      {
        value: "4:5",
        label: "4:5",
        warning:
          "The model was trained on 16:9 and 9:16, other ratios may produce lower quality results.",
      },
    ];
    assert.deepEqual(aspectOptions(values({ startFrame: startAsset }), specsContext), [
      { value: "auto", label: "Auto" },
      ...ratios,
    ]);
    assert.deepEqual(aspectOptions(values({ startFrame: null }), specsContext), ratios);
  });

  it("drops a stale Auto aspect when the start image is removed", () => {
    const cleared = audioToVideoDefinition.applyChange(
      values({ aspectRatio: "auto" }),
      {
        kind: "image-asset",
        fieldId: "startFrame",
        dataKey: "startFrame",
        value: null,
      },
      context(),
    );
    assert.equal(
      audioToVideoDefinition.normalize(cleared, context()).aspectRatio,
      "16:9",
    );
    assert.equal(
      audioToVideoDefinition.normalize(
        values({ aspectRatio: "auto" }),
        context(),
      ).aspectRatio,
      "auto",
    );
  });

  it("validate requires audio", () => {
    assert.deepEqual(validateAudioToVideo(values({ audio: null }), context()), [
      { id: "audio-required", fieldId: "audio", message: "Audio is required." },
    ]);
    assert.deepEqual(validateAudioToVideo(values(), context()), []);
  });

  it("validate requires prompt or start image", () => {
    assert.deepEqual(
      validateAudioToVideo(
        values({ prompt: "   ", startFrame: null }),
        context(),
      ),
      [
        {
          id: "prompt-or-start-frame-required",
          fieldId: "prompt",
          message: "Add a prompt or a start image.",
        },
      ],
    );
    assert.deepEqual(
      validateAudioToVideo(
        values({ prompt: "", startFrame: startAsset }),
        context(),
      ),
      [],
    );
    assert.deepEqual(
      validateAudioToVideo(
        values({ prompt: "sing", startFrame: null }),
        context(),
      ),
      [],
    );
  });

  it("considers in-envelope 10.563s audio valid without snapping to 20s", () => {
    assert.deepEqual(
      validateAudioToVideo(
        values(),
        context({ audioDurationSeconds: 10.563 }),
      ),
      [],
    );
    assert.equal(
      audioToVideoDefinition.isReady?.(
        values(),
        context({ audioDurationSeconds: 10.563 }),
      ),
      true,
    );
    const body = toCreateBody(
      values(),
      context({ audioDurationSeconds: 10.563 }),
    );
    assert.equal("duration" in body.params, false);
    assert.equal("numFrames" in body.params, false);
  });

  it("explains an over-cap audio instead of leaving Generate dead", () => {
    const issues = validateAudioToVideo(
      values(),
      context({ audioDurationSeconds: 21 }),
    );
    assert.deepEqual(issues, [
      {
        id: "audio-longer-than-cap",
        fieldId: "audio",
        message: "This audio is longer than 20s at 540p. Choose a lower resolution or replace it with a shorter clip.",
        alwaysRevealed: true,
      },
    ]);
    assert.equal(
      audioToVideoDefinition.isReady?.(
        values(),
        context({ audioDurationSeconds: 21 }),
      ),
      false,
    );
  });

  it("explains audio shorter than the 2s minimum", () => {
    const issues = validateAudioToVideo(
      values(),
      context({ audioDurationSeconds: 1.0 }),
    );
    assert.deepEqual(issues, [
      {
        id: "audio-shorter-than-minimum",
        fieldId: "audio",
        message: "This audio is shorter than 2s. Replace it to continue.",
        alwaysRevealed: true,
      },
    ]);
    assert.equal(
      audioToVideoDefinition.isReady?.(
        values(),
        context({ audioDurationSeconds: 1.0 }),
      ),
      false,
    );
  });

  it("explains audio with no readable length", () => {
    assert.deepEqual(
      validateAudioToVideo(values(), context({ audioDurationSeconds: null })),
      [
        {
          id: "audio-duration-unknown",
          fieldId: "audio",
          message:
            "We couldn't read this audio's length. Replace it to continue.",
          alwaysRevealed: true,
        },
      ],
    );
  });

  it("stays quiet while the audio asset is still loading", () => {
    assert.deepEqual(
      validateAudioToVideo(
        values(),
        context({ audioDurationSeconds: null, audioDurationPending: true }),
      ),
      [],
    );
  });

  it("leaves the no-A2V-cell machine to the feature-unavailable message", () => {
    // No advertised cell means no cap to quote, and `isUnavailable` already
    // says the machine cannot run this feature at all.
    assert.deepEqual(
      validateAudioToVideo(values(), context({ specs: null })),
      [],
    );
  });

  it("toCreateBody sends 540p/24 and the selected offering with no length field", () => {
    const body = toCreateBody(
      values({
        prompt: "  sings  ",
        startFrame: startAsset,
        aspectRatio: "auto",
      }),
      context({ audioDurationSeconds: 7.2 }),
    );
    assert.deepEqual(body, {
      contract_version: 1,
      inputs: {
        audio: { assetId: "audio-1" },
        startFrame: { assetId: "start-1" },
      },
      params: {
        prompt: "sings",
        model: "ltx-2.5-fast",
        aspectRatio: "auto",
        resolution: "540p",
        fps: 24,
        cameraMotion: "none",
        negativePrompt: "",
        promptProvenance: "typed",
      },
    });
    assert.equal(JSON.stringify(body).includes("/"), false);
    assert.equal(JSON.stringify(body).includes("blob:"), false);
  });

  it("sends a null startFrame when the optional image slot is empty", () => {
    const body = toCreateBody(
      values({ startFrame: null, aspectRatio: "16:9" }),
      context({ audioDurationSeconds: 20 }),
    );
    assert.equal(body.inputs.startFrame, null);
    assert.equal("duration" in body.params, false);
    assert.equal(body.params.aspectRatio, "16:9");
  });

  it("refuses to build a body without audio or an in-envelope length", () => {
    assert.throws(() => toCreateBody(values({ audio: null }), context()));
    assert.throws(() =>
      toCreateBody(values(), context({ audioDurationSeconds: 21 })),
    );
    assert.throws(() =>
      toCreateBody(values(), context({ audioDurationSeconds: null })),
    );
    assert.throws(() =>
      toCreateBody(values(), context({ audioDurationSeconds: 1.0 })),
    );
  });
});

describe("audioToVideo Generate gate", () => {
  it("stays disabled while specs are unresolved or advertise no A2V cell", () => {
    assert.equal(
      audioToVideoDefinition.isReady?.(values(), context({ specs: null })),
      false,
    );
    assert.equal(
      audioToVideoDefinition.isReady?.(values(), context({ specs: undefined })),
      false,
    );
    assert.equal(
      audioToVideoDefinition.isReady?.(
        values(),
        context({
          specs: {
            api_models: [],
            local_models: [
              {
                pipeline: "fast",
                spec: {
                  display_name: "LTX 2.5 Fast",
                  supported_resolutions_durations: {
                    "540p": a2vCell([6, 8]),
                  },
                },
              },
            ],
          },
        }),
      ),
      false,
    );
  });

  it("isUnavailable when nothing is downloaded even if local_models advertises Fast A2V", () => {
    assert.equal(
      audioToVideoDefinition.isUnavailable?.(
        values(),
        context({
          specs: {
            api_models: [],
            local_models: a2vSpecs().local_models,
            downloaded_local_models: [],
          },
        }),
      ),
      true,
    );
  });

  it("isUnavailable only when no A2V cell is advertised at 540p/24", () => {
    // Downloaded Fast has plain 540p/24 cells but no A2V matrix at all.
    const spec = {
      display_name: "LTX 2.5 Fast",
      supported_resolutions_durations: {
        "540p": a2vCell([6, 8, 10]),
      },
    };
    const audioUnsupported: VideoGenerationModelSpecsResponse = {
      api_models: [],
      local_models: [{ pipeline: "fast", spec }],
      downloaded_local_models: [
        { model: "ltx-2.5-fast", pipeline: "fast", spec },
      ],
    };

    assert.equal(
      audioToVideoDefinition.isUnavailable?.(
        values(),
        context({ specs: audioUnsupported }),
      ),
      true,
    );
    assert.equal(
      audioToVideoDefinition.isUnavailable?.(values(), context()),
      false,
    );
    // A 5s-only A2V envelope is a valid machine: every advertised cell stays
    // available, independent of GenSpace's 6s picker floor.
    assert.equal(
      audioToVideoDefinition.isUnavailable?.(
        values(),
        context({ specs: a2vSpecs([5]) }),
      ),
      false,
    );
  });

  it("keeps a 5s-only advertised envelope generatable for 2s+ audio under cap +0.1", () => {
    // Backend parity: Fast 540p/24 [5] is valid; every A2V cell stays available.
    const fiveOnly = context({
      specs: a2vSpecs([5]),
      audioDurationSeconds: 4.5,
    });
    assert.equal(
      audioToVideoDefinition.isUnavailable?.(values(), fiveOnly),
      false,
    );
    assert.deepEqual(validateAudioToVideo(values(), fiveOnly), []);
    assert.equal(audioToVideoDefinition.isReady?.(values(), fiveOnly), true);
    const body = toCreateBody(values(), fiveOnly);
    assert.equal("duration" in body.params, false);
    assert.equal("numFrames" in body.params, false);
    assert.equal(body.params.resolution, "540p");
    assert.equal(body.params.fps, 24);
    // 2s floor still holds on the small envelope.
    assert.equal(
      audioToVideoDefinition.isReady?.(
        values(),
        context({ specs: a2vSpecs([5]), audioDurationSeconds: 2.0 }),
      ),
      true,
    );
    assert.equal(
      audioToVideoDefinition.isReady?.(
        values(),
        context({ specs: a2vSpecs([5]), audioDurationSeconds: 1.9 }),
      ),
      false,
    );
    // At/over cap +0.1 stays rejected with the 5s cap quoted.
    assert.deepEqual(
      validateAudioToVideo(
        values(),
        context({ specs: a2vSpecs([5]), audioDurationSeconds: 5.1 }),
      ),
      [
        {
          id: "audio-longer-than-cap",
          fieldId: "audio",
          message: "This audio is longer than 5s at 540p. Choose a lower resolution or replace it with a shorter clip.",
          alwaysRevealed: true,
        },
      ],
    );
  });
});

describe("audioToVideo fromGeneration", () => {
  it("restores prompt, aspect, and both input slots from spec asset IDs", () => {
    const parsed = fromGeneration({
      params: {
        prompt: "a singer",
        model: "ltx-2.5-fast",
        aspectRatio: "9:16",
        resolution: "540p",
        numFrames: 169,
        fps: 24,
      },
      inputs: {
        audio: { assetId: "from-audio" },
        startFrame: { assetId: "from-start" },
      },
    });
    assert.deepEqual(parsed, {
      startFrame: { assetId: "from-start" },
      prompt: "a singer",
      audio: { assetId: "from-audio" },
      model: "ltx-2.5-fast",
      resolution: "540p",
      aspectRatio: "9:16",
    });
  });

  it("restores a missing start frame as null and keeps Auto", () => {
    const parsed = fromGeneration({
      params: { prompt: "", aspectRatio: "auto" },
      inputs: { audio: { assetId: "only-audio" } },
    });
    assert.equal(parsed?.startFrame, null);
    assert.equal(parsed?.aspectRatio, "auto");
    assert.equal(parsed?.prompt, "");
    assert.equal(parsed?.model, "ltx-2.5-fast");
  });

  it("returns null when spec.inputs.audio is missing", () => {
    assert.equal(fromGeneration({ params: { prompt: "x" } }), null);
    assert.equal(fromGeneration({ inputs: {} }), null);
    assert.equal(
      fromGeneration({
        params: { prompt: "x" },
        inputs: { startFrame: { assetId: "start" } },
      }),
      null,
    );
  });
});

describe("audioToVideo seed and hydration paths", () => {
  it("seeds both packaged example slots on a first Desktop visit", () => {
    const seeded = resolveFeatureSeedValues(audioToVideoDefinition, {
      context: context(),
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: undefined,
    });
    assert.deepEqual(seeded.audio, seed.audio);
    assert.deepEqual(seeded.startFrame, seed.startFrame);
    assert.equal(seeded.prompt, AUDIO_TO_VIDEO_EXAMPLE_PROMPT);
    assert.equal(seeded.aspectRatio, "auto");
    assert.equal(seeded.model, "ltx-2.5-fast");
  });

  it("starts with empty slots and 16:9 where no packaged seed exists", () => {
    const seeded = resolveFeatureSeedValues(audioToVideoDefinition, {
      context: context({ seed: null }),
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: undefined,
    });
    assert.equal(seeded.audio, null);
    assert.equal(seeded.startFrame, null);
    assert.equal(seeded.aspectRatio, "16:9");
  });

  it("does not reinsert the seed after an explicitly persisted clear", () => {
    const next = resolveFeatureSeedValues(audioToVideoDefinition, {
      context: context(),
      hasStoredValues: true,
      storedValues: values({ audio: null, startFrame: null, prompt: "kept" }),
      lastGenerationSpec: undefined,
    });
    assert.equal(next.audio, null);
    assert.equal(next.startFrame, null);
    assert.equal(next.prompt, "kept");
  });

  it("hydrates prior-generation inputs before applying the seed", () => {
    const next = resolveFeatureSeedValues(audioToVideoDefinition, {
      context: context(),
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: {
        params: { prompt: "from history", aspectRatio: "9:16" },
        inputs: {
          audio: { assetId: "history-audio" },
          startFrame: { assetId: "history-start" },
        },
      },
    });
    assert.deepEqual(next.audio, { assetId: "history-audio" });
    assert.deepEqual(next.startFrame, { assetId: "history-start" });
    assert.equal(next.prompt, "from history");
    assert.equal(next.aspectRatio, "9:16");
  });
});

describe("audioToVideo offering picker", () => {
  it("lists every downloaded offering that advertises A2V at 540p/24", () => {
    const fields = audioToVideoDefinition.form(
      values(),
      context({ specs: specFixtureBothOfferings("LTX 2.3 Fast") }),
    ).fields;
    const modelField = fields.find((field) => field.dataKey === "model");
    assert.deepEqual(
      modelField?.kind === "options"
        ? modelField.options.map((option) => option.value)
        : undefined,
      ["ltx-2.5-fast", "ltx-2.3-fast"],
    );
  });

  it("seeds the offering that matches Settings' active local row", () => {
    const seeded = resolveFeatureSeedValues(audioToVideoDefinition, {
      context: context({ specs: specFixtureBothOfferings("LTX 2.3 Fast") }),
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: undefined,
    });
    assert.equal(seeded.model, "ltx-2.3-fast");
  });

  it("maps a stored pipeline id to the Settings offering", () => {
    const parsed = fromGeneration(
      {
        params: {
          prompt: "a singer",
          model: "fast",
          aspectRatio: "9:16",
        },
        inputs: { audio: { assetId: "from-audio" } },
      },
      context({ specs: specFixtureBothOfferings("LTX 2.3 Fast") }),
    );
    assert.equal(parsed?.model, "ltx-2.3-fast");
  });

  it("keeps a last-generation offering id even when Settings is a different row", () => {
    const seeded = resolveFeatureSeedValues(audioToVideoDefinition, {
      context: context({ specs: specFixtureBothOfferings("LTX 2.3 Fast") }),
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: {
        params: {
          prompt: "keep this prompt",
          model: "ltx-2.5-fast",
          aspectRatio: "16:9",
        },
        inputs: { audio: { assetId: "history-audio" } },
      },
    });
    assert.equal(seeded.model, "ltx-2.5-fast");
    assert.equal(seeded.prompt, "keep this prompt");
  });

  it("shows the Settings-matching offering on first paint before the store write", () => {
    const displayed = resolveFeatureFormValues(audioToVideoDefinition, {
      context: context({ specs: specFixtureBothOfferings("LTX 2.3 Fast") }),
      contextReady: true,
      storedValues: undefined,
      generationsFetched: true,
      lastGenerationSpec: undefined,
    });
    assert.equal(displayed.model, "ltx-2.3-fast");
  });

  it("resetValues picks the Settings-matching offering, not hardcoded 2.5", () => {
    const reset = audioToVideoDefinition.resetValues?.(
      context({ specs: specFixtureBothOfferings("LTX 2.3 Fast") }),
    );
    assert.equal(reset?.model, "ltx-2.3-fast");
  });

  it("toCreateBody sends the selected 2.3 offering", () => {
    const body = toCreateBody(
      values({ model: "ltx-2.3-fast" }),
      context({
        specs: specFixtureBothOfferings("LTX 2.5 Fast"),
        audioDurationSeconds: 4.5,
      }),
    );
    assert.equal(body.params.model, "ltx-2.3-fast");
  });

  it("quotes the selected offering's cap when 2.3 is 5s-only and 2.5 is 20s", () => {
    const specs: VideoGenerationModelSpecsResponse = {
      api_models: [],
      local_models: specFixtureBothOfferings("LTX 2.3 Fast").local_models,
      downloaded_local_models: [
        {
          model: "ltx-2.5-fast",
          pipeline: "fast",
          spec: {
            display_name: "LTX 2.5 Fast",
            supported_resolutions_durations: {},
            a2v_supported_resolutions_durations: {
              "540p": a2vCell([5, 6, 8, 10, 20]),
            },
          },
        },
        {
          model: "ltx-2.3-fast",
          pipeline: "fast",
          spec: {
            display_name: "LTX 2.3 Fast",
            supported_resolutions_durations: {},
            a2v_supported_resolutions_durations: {
              "540p": a2vCell([5]),
            },
          },
        },
      ],
    };
    const longOn23 = context({
      specs,
      audioDurationSeconds: 7.2,
    });
    assert.deepEqual(
      validateAudioToVideo(values({ model: "ltx-2.3-fast" }), longOn23),
      [
        {
          id: "audio-longer-than-cap",
          fieldId: "audio",
          message: "This audio is longer than 5s at 540p. Choose a lower resolution or replace it with a shorter clip.",
          alwaysRevealed: true,
        },
      ],
    );
    assert.deepEqual(
      validateAudioToVideo(values({ model: "ltx-2.5-fast" }), longOn23),
      [],
    );
  });
});

describe("audioToVideo resolution", () => {
  it("lists advertised A2V resolutions for the selected offering", () => {
    const fields = audioToVideoDefinition.form(
      values(),
      context({ specs: specFixture() }),
    ).fields;
    const resolutionField = fields.find((field) => field.dataKey === "resolution");
    assert.deepEqual(
      resolutionField?.kind === "options"
        ? resolutionField.options.map((option) => option.value)
        : undefined,
      ["270p", "360p", "540p", "720p", "1080p"],
    );
    assert.equal(
      resolutionField?.kind === "options"
        ? resolutionField.maxOptionCount
        : undefined,
      7,
    );
  });

  it("sends the selected resolution on create", () => {
    const body = toCreateBody(
      values({ resolution: "1080p" }),
      context({ specs: specFixture(), audioDurationSeconds: 8 }),
    );
    assert.equal(body.params.resolution, "1080p");
    assert.equal("duration" in body.params, false);
  });

  it("keeps the audio asset when resolution changes", () => {
    const next = audioToVideoDefinition.applyChange(
      values({ audio: audioAsset }),
      {
        kind: "options",
        fieldId: "resolution",
        dataKey: "resolution",
        value: "1080p",
      },
      context({ specs: specFixture(), audioDurationSeconds: 8 }),
    );
    assert.deepEqual(next.audio, audioAsset);
    assert.equal(next.resolution, "1080p");
  });

  it("uses the first 10s of 20s audio at 1080p and the whole clip at 540p", () => {
    const long = context({
      specs: specFixture(),
      audioDurationSeconds: 20,
    });
    const at1080p = values({ resolution: "1080p" });
    assert.equal(audioToVideoDefinition.isReady?.(at1080p, long), true);
    assert.deepEqual(validateAudioToVideo(at1080p, long), [
      {
        id: "audio-longer-than-resolution",
        fieldId: "audio",
        message:
          "At 1080p, the maximum is 10 seconds. We'll use only the first 10s, or you can choose a lower resolution.",
        alwaysRevealed: true,
        blocksGenerate: false,
      },
    ]);
    const body = toCreateBody(at1080p, long);
    assert.equal(body.params.resolution, "1080p");
    const at540p = values({ resolution: "540p" });
    assert.equal(audioToVideoDefinition.isReady?.(at540p, long), true);
    assert.deepEqual(validateAudioToVideo(at540p, long), []);
  });
});

