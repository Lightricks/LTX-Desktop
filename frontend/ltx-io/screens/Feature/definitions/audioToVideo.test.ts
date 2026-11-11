import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type {
  DroppedFastAspectCell,
  FastCellAsk,
  VideoGenerationDuration,
  VideoGenerationModelSpecsResponse,
} from "../../../../lib/video-generation-model-specs.ts";
import { resolveFeatureFormValues, resolveFeatureSeedValues } from "../types.ts";

import {
  AUDIO_TO_VIDEO_DEFAULTS,
  AUDIO_TO_VIDEO_EXAMPLE_NEGATIVE_PROMPT,
  AUDIO_TO_VIDEO_EXAMPLE_PROMPT,
  type AudioToVideoContext,
  type AudioToVideoValues,
  a2vAudioNumFrames,
  audioToVideoDefinition,
  buildAudioToVideoForm,
  fromGeneration,
  toCreateBody,
  validateAudioToVideo,
} from "./audioToVideo.ts";
import { specFixture, specFixtureBothOfferings } from "./specFixture.ts";

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

function context(overrides: Partial<AudioToVideoContext> = {}): AudioToVideoContext {
  return {
    specs: a2vSpecs(),
    seed,
    audioDurationSeconds: 7.2,
    audioDurationPending: false,
    startFrameSize: null,
    ...overrides,
  };
}

function values(overrides: Partial<AudioToVideoValues> = {}): AudioToVideoValues {
  return {
    ...AUDIO_TO_VIDEO_DEFAULTS,
    audio: audioAsset,
    startFrame: startAsset,
    ...overrides,
  };
}

/** Audio-grid asks against a 32 GB GPU, at 24 fps. */
function withA2vAsks(
  specs: VideoGenerationModelSpecsResponse,
  asks: { resolution: FastCellAsk["resolution"]; duration: FastCellAsk["duration"]; ask_gib: number }[],
): VideoGenerationModelSpecsResponse {
  const cellAsks: FastCellAsk[] = asks.map((ask) => ({
    ...ask,
    fps: 24,
    audio: true,
    confidence: "solid",
    windows: 1,
    tight: false,
  }));
  for (const item of [...specs.local_models, ...(specs.downloaded_local_models ?? [])]) {
    item.spec.cell_asks = cellAsks;
  }
  return { ...specs, memory: { device: "gpu", capacity_gib: 32 } };
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
      negativePrompt: AUDIO_TO_VIDEO_EXAMPLE_NEGATIVE_PROMPT,
      audio: null,
      model: "ltx-2.5-fast",
      resolution: "540p",
      aspectRatio: "auto",
      passShape: null,
    });
    assert.deepEqual(audioToVideoDefinition.defaults, AUDIO_TO_VIDEO_DEFAULTS);
  });

  it("renders start frame, prompt, negativePrompt, audio, model, resolution, then aspect — no duration/fps", () => {
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
        { kind: "textarea", id: "negativePrompt", dataKey: "negativePrompt" },
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
      audioToVideoDefinition.normalize(values({ aspectRatio: "auto" }), context())
        .aspectRatio,
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
      validateAudioToVideo(values({ prompt: "   ", startFrame: null }), context()),
      [
        {
          id: "prompt-or-start-frame-required",
          fieldId: "prompt",
          message: "Add a prompt or a start image.",
        },
      ],
    );
    assert.deepEqual(
      validateAudioToVideo(values({ prompt: "", startFrame: startAsset }), context()),
      [],
    );
    assert.deepEqual(
      validateAudioToVideo(values({ prompt: "sing", startFrame: null }), context()),
      [],
    );
  });

  it("considers in-envelope 10.563s audio valid without snapping to 20s", () => {
    assert.deepEqual(
      validateAudioToVideo(values(), context({ audioDurationSeconds: 10.563 })),
      [],
    );
    assert.equal(
      audioToVideoDefinition.isReady?.(
        values(),
        context({ audioDurationSeconds: 10.563 }),
      ),
      true,
    );
    const body = toCreateBody(values(), context({ audioDurationSeconds: 10.563 }));
    assert.equal("duration" in body.params, false);
    assert.equal("numFrames" in body.params, false);
  });

  it("explains an over-cap audio instead of leaving Generate dead", () => {
    const issues = validateAudioToVideo(values(), context({ audioDurationSeconds: 21 }));
    assert.deepEqual(issues, [
      {
        id: "audio-longer-than-cap",
        fieldId: "audio",
        message:
          "This audio is longer than 20s at 540p. Choose a lower resolution or replace it with a shorter clip.",
        alwaysRevealed: true,
      },
    ]);
    assert.equal(
      audioToVideoDefinition.isReady?.(values(), context({ audioDurationSeconds: 21 })),
      false,
    );
  });

  it("explains audio shorter than the 2s minimum", () => {
    const issues = validateAudioToVideo(values(), context({ audioDurationSeconds: 1.0 }));
    assert.deepEqual(issues, [
      {
        id: "audio-shorter-than-minimum",
        fieldId: "audio",
        message: "This audio is shorter than 2s. Replace it to continue.",
        alwaysRevealed: true,
      },
    ]);
    assert.equal(
      audioToVideoDefinition.isReady?.(values(), context({ audioDurationSeconds: 1.0 })),
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
          message: "We couldn't read this audio's length. Replace it to continue.",
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
    assert.deepEqual(validateAudioToVideo(values(), context({ specs: null })), []);
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
        negativePrompt: AUDIO_TO_VIDEO_EXAMPLE_NEGATIVE_PROMPT,
        promptProvenance: "typed",
        passShape: "auto",
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
    assert.throws(() => toCreateBody(values(), context({ audioDurationSeconds: 21 })));
    assert.throws(() => toCreateBody(values(), context({ audioDurationSeconds: null })));
    assert.throws(() => toCreateBody(values(), context({ audioDurationSeconds: 1.0 })));
  });
});

function aspectRefusal(
  duration: DroppedFastAspectCell["duration"],
  message: string,
  audio = true,
): DroppedFastAspectCell {
  return {
    resolution: "540p",
    fps: 24,
    duration,
    aspectRatio: "21:9",
    reason: "stream_over",
    message,
    audio,
  };
}

function a2vSpecsWithAspectRefusals(
  cells: DroppedFastAspectCell[],
): VideoGenerationModelSpecsResponse {
  const specs = a2vSpecs([5, 6, 8, 10, 20]);
  for (const item of specs.downloaded_local_models ?? []) {
    item.spec.dropped_aspect_cells = cells;
  }
  return specs;
}

function aspectOption(
  formValues: AudioToVideoValues,
  formContext: AudioToVideoContext,
  value: string,
) {
  return aspectOptions(formValues, formContext)?.find((option) => option.value === value);
}

describe("audioToVideo aspect refusals", () => {
  it("greys out a 21:9 shape refused at or under the audio length, naming the nearest refusal", () => {
    // 7.2s of audio against refusals at 5s and 6s: the shape stays
    // selectable without this mapping, then 422s at queue-create.
    const formContext = context({
      specs: a2vSpecsWithAspectRefusals([
        aspectRefusal(5, "needs 11.0 at 5s"),
        aspectRefusal(6, "needs 11.9 at 6s"),
      ]),
      audioDurationSeconds: 7.2,
    });
    assert.equal(
      aspectOption(values(), formContext, "21:9")?.disabledReason,
      "needs 11.9 at 6s",
    );
    assert.equal(aspectOption(values(), formContext, "16:9")?.disabledReason, undefined);
  });

  it("leaves 21:9 enabled when it only refuses past the audio length", () => {
    const formContext = context({
      specs: a2vSpecsWithAspectRefusals([aspectRefusal(10, "needs 15.3")]),
      audioDurationSeconds: 7.2,
    });
    assert.equal(aspectOption(values(), formContext, "21:9")?.disabledReason, undefined);
  });

  it("leaves aspects enabled while the audio length is unknown", () => {
    const formContext = context({
      specs: a2vSpecsWithAspectRefusals([aspectRefusal(5, "needs 11.0")]),
      audioDurationSeconds: null,
      audioDurationPending: true,
    });
    assert.equal(aspectOption(values(), formContext, "21:9")?.disabledReason, undefined);
  });

  it("ignores non-audio aspect cells from the plain matrix", () => {
    const formContext = context({
      specs: a2vSpecsWithAspectRefusals([
        aspectRefusal(5, "plain-matrix refusal", false),
      ]),
      audioDurationSeconds: 7.2,
    });
    assert.equal(aspectOption(values(), formContext, "21:9")?.disabledReason, undefined);
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
      downloaded_local_models: [{ model: "ltx-2.5-fast", pipeline: "fast", spec }],
    };

    assert.equal(
      audioToVideoDefinition.isUnavailable?.(
        values(),
        context({ specs: audioUnsupported }),
      ),
      true,
    );
    assert.equal(audioToVideoDefinition.isUnavailable?.(values(), context()), false);
    // A 5s-only A2V envelope is a valid machine: every advertised cell stays
    // available, independent of GenSpace's 6s picker floor.
    assert.equal(
      audioToVideoDefinition.isUnavailable?.(values(), context({ specs: a2vSpecs([5]) })),
      false,
    );
  });

  it("keeps a 5s-only advertised envelope generatable for 2s+ audio under cap +0.1", () => {
    // Backend parity: Fast 540p/24 [5] is valid; every A2V cell stays available.
    const fiveOnly = context({
      specs: a2vSpecs([5]),
      audioDurationSeconds: 4.5,
    });
    assert.equal(audioToVideoDefinition.isUnavailable?.(values(), fiveOnly), false);
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
          message:
            "This audio is longer than 5s at 540p. Choose a lower resolution or replace it with a shorter clip.",
          alwaysRevealed: true,
        },
      ],
    );
  });
});

describe("audioToVideo pass shape", () => {
  // 7.2 s of audio prices the 8 s cell, which runs in parts with a one-pass alternative.
  function shapedA2vSpecs(): VideoGenerationModelSpecsResponse {
    const specs = a2vSpecs();
    for (const item of [...specs.local_models, ...(specs.downloaded_local_models ?? [])]) {
      item.spec.cell_asks = [
        {
          resolution: "540p",
          fps: 24,
          duration: 8,
          audio: true,
          ask_gib: 12,
          confidence: "solid",
          windows: 2,
          tight: false,
          alternative: { ask_gib: 15, confidence: "solid", windows: 1, tight: false },
        },
      ];
    }
    return { ...specs, budgetMode: "brave", memory: { device: "gpu", capacity_gib: 16 } };
  }

  it("audio to video keeps a restored shape until the audio's length is known", () => {
    const specs = shapedA2vSpecs();
    const restored = resolveFeatureSeedValues(audioToVideoDefinition, {
      context: context({ specs, audioDurationSeconds: null, audioDurationPending: true }),
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: {
        params: { prompt: "a singer", aspectRatio: "16:9", passShape: "one_pass" },
        inputs: { audio: audioAsset, startFrame: startAsset },
      },
    });
    assert.equal(restored.passShape, "one_pass");

    const loaded = context({ specs, audioDurationSeconds: 7.2 });
    const next = audioToVideoDefinition.normalize(restored, loaded);
    assert.equal(next.passShape, "one_pass");
    assert.equal(toCreateBody(next, loaded).params.passShape, "one_pass");
  });
});

describe("audioToVideo fromGeneration", () => {
  it("restores prompt, negativePrompt, aspect, and both input slots from spec asset IDs", () => {
    const parsed = fromGeneration({
      params: {
        prompt: "a singer",
        negativePrompt: "closed mouth",
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
      negativePrompt: "closed mouth",
      audio: { assetId: "from-audio" },
      model: "ltx-2.5-fast",
      resolution: "540p",
      aspectRatio: "9:16",
      passShape: null,
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
    assert.deepEqual(validateAudioToVideo(values({ model: "ltx-2.3-fast" }), longOn23), [
      {
        id: "audio-longer-than-cap",
        fieldId: "audio",
        message:
          "This audio is longer than 5s at 540p. Choose a lower resolution or replace it with a shorter clip.",
        alwaysRevealed: true,
      },
    ]);
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
      resolutionField?.kind === "options" ? resolutionField.maxOptionCount : undefined,
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

  it("blocks a standard prefix of a dropped ceiling when a longer resolution still exists", () => {
    const specs = structuredClone(specFixture());
    const spec = specs.downloaded_local_models[0]?.spec;
    if (spec?.a2v_supported_resolutions_durations?.["1080p"] == null) {
      throw new Error("missing 1080p");
    }
    spec.a2v_supported_resolutions_durations["1080p"].fps_to_durations["24"] = [5];
    spec.dropped_cells = [
      {
        resolution: "1080p",
        fps: 24,
        duration: 10,
        reason: "stream_over",
        message: "needs 40",
        audio: true,
      },
    ];
    const long = context({ specs, audioDurationSeconds: 15 });
    const at1080p = values({ resolution: "1080p" });
    const issues = validateAudioToVideo(at1080p, long);
    assert.equal(issues[0]?.id, "audio-past-kept");
    assert.equal(issues[0]?.blocksGenerate, undefined);
    assert.match(issues[0]?.message ?? "", /up to 5s/);
    assert.equal(audioToVideoDefinition.isReady?.(at1080p, long), false);
  });

  it("offers a lower resolution at that resolution's own length, never a shorter clip", () => {
    // 15 s of audio runs as the 10 s cell at 1080p (its longest) but as the 20 s cell at 720p
    // and 540p. 720p fits at 10 s and is over at 20 s, so only 540p is a fix.
    const specs = withA2vAsks(structuredClone(specFixture()), [
      { resolution: "1080p", duration: 10, ask_gib: 40 },
      { resolution: "720p", duration: 10, ask_gib: 20 },
      { resolution: "720p", duration: 20, ask_gib: 36 },
      { resolution: "540p", duration: 20, ask_gib: 25 },
    ]);
    const clip = context({ specs, audioDurationSeconds: 15 });
    const memory = buildAudioToVideoForm(values({ resolution: "1080p" }), clip).memory;
    assert.equal(memory?.over, true);
    assert.deepEqual(
      memory?.fixes.map((fix) => [fix.dataKey, fix.value]),
      [["resolution", "540p"]],
    );
  });

  it("shows no meter before the audio is chosen", () => {
    const specs = withA2vAsks(structuredClone(specFixture()), [
      { resolution: "1080p", duration: 5, ask_gib: 12 },
      { resolution: "1080p", duration: 10, ask_gib: 20 },
    ]);
    const memory = buildAudioToVideoForm(
      values({ resolution: "1080p", audio: null }),
      context({ specs, audioDurationSeconds: 7.2 }),
    ).memory;
    assert.equal(memory, null);
  });

  it("prices 10.2s audio as the 10s cell, not the next longer one", () => {
    const specs = withA2vAsks(structuredClone(specFixture()), [
      { resolution: "1080p", duration: 10, ask_gib: 20 },
      { resolution: "1080p", duration: 20, ask_gib: 40 },
    ]);
    const spec = specs.downloaded_local_models[0]?.spec;
    if (spec?.a2v_supported_resolutions_durations?.["1080p"] == null) {
      throw new Error("missing 1080p");
    }
    spec.a2v_supported_resolutions_durations["1080p"].fps_to_durations["24"] = [5, 10, 20];
    const memory = buildAudioToVideoForm(
      values({ resolution: "1080p" }),
      context({ specs, audioDurationSeconds: 10.2 }),
    ).memory;
    assert.equal(memory?.askGib, 20);
    assert.equal(memory?.over, false);
  });

  it("accepts 10.2s audio when it floors onto the kept 10s frame count", () => {
    const specs = structuredClone(specFixture());
    const spec = specs.downloaded_local_models[0]?.spec;
    if (spec == null) throw new Error("missing spec");
    spec.dropped_cells = [
      {
        resolution: "1080p",
        fps: 24,
        duration: 20,
        reason: "stream_over",
        message: "needs 40",
        audio: true,
      },
    ];
    const clip = context({ specs, audioDurationSeconds: 10.2 });
    const at1080p = values({ resolution: "1080p" });
    assert.deepEqual(validateAudioToVideo(at1080p, clip), []);
    assert.equal(audioToVideoDefinition.isReady?.(at1080p, clip), true);
    assert.equal(a2vAudioNumFrames(clip, "ltx-2.5-fast", "1080p"), 241);
  });

  it("does not claim a fully dropped resolution can run up to the static ceiling", () => {
    const specs = structuredClone(specFixture());
    const spec = specs.downloaded_local_models[0]?.spec;
    if (spec?.a2v_supported_resolutions_durations == null) {
      throw new Error("missing a2v map");
    }
    delete spec.a2v_supported_resolutions_durations["1080p"];
    spec.dropped_cells = [
      {
        resolution: "1080p",
        fps: 24,
        duration: 20,
        reason: "stream_over",
        message: "needs 40",
        audio: true,
      },
    ];
    const clip = context({ specs, audioDurationSeconds: 8 });
    const at1080p = values({ resolution: "1080p" });
    const issues = validateAudioToVideo(at1080p, clip);
    assert.equal(issues[0]?.id, "audio-past-kept");
    assert.doesNotMatch(issues[0]?.message ?? "", /up to/);
    assert.match(issues[0]?.message ?? "", /can't run at 1080p/);
    assert.equal(audioToVideoDefinition.isReady?.(at1080p, clip), false);
  });

  it("blocks a clip past the kept max when a dropped duration still sits in the ceiling", () => {
    const specs = specFixture();
    const spec = specs.downloaded_local_models[0]?.spec;
    if (spec == null) throw new Error("missing spec");
    spec.dropped_cells = [
      {
        resolution: "1080p",
        fps: 24,
        duration: 20,
        reason: "stream_over",
        message: "needs 40",
        audio: true,
      },
    ];
    const long = context({ specs, audioDurationSeconds: 15 });
    const issues = validateAudioToVideo(values({ resolution: "1080p" }), long);
    assert.equal(issues[0]?.id, "audio-past-kept");
    assert.equal(issues[0]?.blocksGenerate, undefined);
    assert.match(issues[0]?.message ?? "", /up to 10s/);
  });
});
