import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { createApiClient } from "../../lib/api-client.ts";
import { resetBackendCredentials } from "../../lib/backend.ts";
import { ApiResultError } from "../lib/unwrapApiResult.ts";
import {
  AUDIO_ACCEPT,
  AUDIO_EXTENSIONS,
  AUDIO_OR_VIDEO_ACCEPT,
  AUDIO_OR_VIDEO_EXTENSIONS,
} from "../screens/Feature/fields/audioAssetInput.ts";
import {
  VIDEO_ACCEPT,
  VIDEO_EXTENSIONS,
} from "../screens/Feature/fields/videoAssetInput.ts";
import {
  IMAGE_ACCEPT,
  IMAGE_EXTENSIONS,
} from "../screens/Feature/fields/imageAssetInput.ts";

import {
  type DesktopExploreHost,
  createDesktopExploreRuntime,
} from "./createDesktopExploreRuntime.ts";
import { DESKTOP_GENERATION_POLLING_POLICY } from "./generationPollingPolicy.ts";

const ingestedAsset = {
  created_at: 1,
  id: "asset-1",
  media_kind: "image",
  metadata: {
    mediaType: "image",
    metadata: { width: 8, height: 8, sizeBytes: 12 },
  },
  mime_type: "image/png",
  name: "photo.png",
  origin: "uploaded",
  path: "/tmp/photo.png",
};

const chooseRequest = {
  title: "Start Frame",
  accept: IMAGE_ACCEPT,
  extensions: IMAGE_EXTENSIONS,
  fileInput: null,
};

function recordingIngestClient(calls: string[]) {
  return createApiClient(async (path, init) => {
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : {};
    calls.push(typeof body.path === "string" ? body.path : path);
    return new Response(JSON.stringify(ingestedAsset), { status: 200 });
  });
}

function hostStub(overrides: Partial<DesktopExploreHost> = {}): DesktopExploreHost {
  return {
    showOpenFileDialog: async () => ["/Users/me/still.png"],
    getPathForFile: () => "/Users/me/dropped.png",
    showItemInFolder: () => {},
    getExploreLoraRecipeSeedPath: async ({ seedId }) =>
      `/app/${seedId}.jpg`,
    getExploreAudioToVideoSeedPaths: async () => ({
      audio: "/app/seed.mp3",
      startFrame: "/app/a2v-seed.jpg",
    }),
    getPackagedSeedPath: async ({ kind }) => {
      switch (kind) {
        case "image-to-video-start-frame":
          return "/app/seed.png";
        case "dolly-in-start-frame":
          return "/app/dolly-in-seed.jpg";
        case "retake-video":
          return "/app/retake-seed.mp4";
        case "extend-video":
          return "/app/extend-seed.mp4";
        case "day-to-night-video":
          return "/app/day-to-night-reference.mp4";
        case "alpha-gen-video":
          return "/app/alpha-gen-reference.mp4";
        case "deblur-video":
          return "/app/deblur-reference.mp4";
        case "colorization-video":
          return "/app/colorization-reference.mp4";
        case "clean-plate-video":
          return "/app/clean-plate-reference.mp4";
        case "decompression-video":
          return "/app/decompression-reference.mp4";
        case "water-simulation-video":
          return "/app/water-simulation-reference.mp4";
        case "layout-to-render-video":
          return "/app/layout-to-render-reference.mp4";
        case "layout-to-render-image":
          return "/app/layout-to-render-first-frame.jpg";
        case "restore-video":
          return "/app/restore-reference.mp4";
        case "restore-image":
          return "/app/restore-first-frame.jpg";
        default:
          throw new Error(`unexpected packaged seed ${kind satisfies never}`);
      }
    },
    writeTempFile: async () => "/tmp/recording.wav",
    openHuggingFaceAuth: async () => true,
    ...overrides,
  };
}

const audioChooseRequest = {
  title: "Audio",
  accept: AUDIO_ACCEPT,
  extensions: AUDIO_EXTENSIONS,
  fileInput: null,
};

const audioOrVideoChooseRequest = {
  title: "Audio",
  accept: AUDIO_OR_VIDEO_ACCEPT,
  extensions: AUDIO_OR_VIDEO_EXTENSIONS,
  fileInput: null,
};

const videoChooseRequest = {
  title: "Video",
  accept: VIDEO_ACCEPT,
  extensions: VIDEO_EXTENSIONS,
  fileInput: null,
};

const ingestedAudio = {
  created_at: 1,
  id: "audio-1",
  media_kind: "audio",
  metadata: { mediaType: "audio", metadata: { durationMs: 4000 } },
  mime_type: "audio/wav",
  name: "clip.wav",
  origin: "uploaded",
  path: "/tmp/clip.wav",
};

/** Records each request path so tests can assert the ingest → extract sequence. */
function recordingAudioClient(calls: string[]) {
  return createApiClient(async (path, init) => {
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : {};
    calls.push(typeof body.path === "string" ? body.path : path);
    return new Response(JSON.stringify(ingestedAudio), { status: 200 });
  });
}

describe("Desktop Explore runtime", () => {
  it("keeps the numeric modelsVersion for specs invalidation", () => {
    const first = createDesktopExploreRuntime(1, {
      api: recordingIngestClient([]),
      host: hostStub(),
      wsUrl: async (path) => `ws://desktop${path}`,
    });
    const later = createDesktopExploreRuntime(8, {
      api: recordingIngestClient([]),
      host: hostStub(),
      wsUrl: async (path) => `ws://desktop${path}`,
    });

    assert.equal(first.modelsVersion, 1);
    assert.equal(later.modelsVersion, 8);
  });

  it("uses the 1s Desktop generation polling policy", () => {
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingIngestClient([]),
      host: hostStub(),
      wsUrl: async (path) => `ws://desktop${path}`,
    });

    assert.equal(runtime.generationPolling, DESKTOP_GENERATION_POLLING_POLICY);
    assert.equal(runtime.generationPolling.queueActiveIntervalMs, 1000);
  });

  it("forwards autoEnhancePrompts onto the runtime", () => {
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingIngestClient([]),
      host: hostStub(),
      wsUrl: async (path) => `ws://desktop${path}`,
      autoEnhancePrompts: false,
    });

    assert.equal(runtime.autoEnhancePrompts, false);
  });
});

describe("Desktop Explore media input", () => {
  it("path-ingests a native dialog selection", async () => {
    const calls: string[] = [];
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingIngestClient(calls),
      host: hostStub(),
      wsUrl: async (path) => `ws://desktop${path}`,
    });

    const asset = await runtime.mediaInput.choose(chooseRequest);

    assert.equal(asset?.id, "asset-1");
    assert.deepEqual(calls, ["/Users/me/still.png"]);
  });

  it("path-ingests a dropped file through the host path resolver", async () => {
    const calls: string[] = [];
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingIngestClient(calls),
      host: hostStub(),
      wsUrl: async (path) => `ws://desktop${path}`,
    });
    const file = new File(["pixels"], "dropped.png", { type: "image/png" });

    const asset = await runtime.mediaInput.ingestDroppedFile(file);

    assert.equal(asset.id, "asset-1");
    assert.deepEqual(calls, ["/Users/me/dropped.png"]);
  });

  it("rejects All-files gif picks before ingest", async () => {
    const calls: string[] = [];
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingIngestClient(calls),
      host: hostStub({
        showOpenFileDialog: async () => ["/Users/me/loop.gif"],
      }),
      wsUrl: async (path) => `ws://desktop${path}`,
    });

    await assert.rejects(
      () => runtime.mediaInput.choose(chooseRequest),
      (error: unknown) => {
        assert.ok(error instanceof ApiResultError);
        assert.equal(error.code, "UNSUPPORTED_MEDIA");
        return true;
      },
    );
    assert.deepEqual(calls, []);
  });

  it("labels the native dialog filter after the accepted media", async () => {
    const filters: Array<{ name: string; extensions: string[] } | undefined> = [];
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingIngestClient([]),
      host: hostStub({
        showOpenFileDialog: async (input) => {
          filters.push(input.filters?.[0]);
          return null;
        },
      }),
      wsUrl: async (path) => `ws://desktop${path}`,
    });

    await runtime.mediaInput.choose(chooseRequest);
    await runtime.mediaInput.choose(audioOrVideoChooseRequest);

    assert.deepEqual(
      filters.map((filter) => filter?.name),
      ["Images", "Audio or video"],
    );
  });

  it("path-ingests a native dialog audio selection", async () => {
    const calls: string[] = [];
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingAudioClient(calls),
      host: hostStub({
        showOpenFileDialog: async () => ["/Users/me/song.wav"],
      }),
      wsUrl: async (path) => `ws://desktop${path}`,
    });

    const asset = await runtime.mediaInput.choose(audioChooseRequest);

    assert.equal(asset?.id, "audio-1");
    assert.deepEqual(calls, ["/Users/me/song.wav"]);
  });

  it("rejects All-files non-audio picks before ingest", async () => {
    const calls: string[] = [];
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingAudioClient(calls),
      host: hostStub({
        showOpenFileDialog: async () => ["/Users/me/notes.txt"],
      }),
      wsUrl: async (path) => `ws://desktop${path}`,
    });

    await assert.rejects(
      () => runtime.mediaInput.choose(audioChooseRequest),
      (error: unknown) => {
        assert.ok(error instanceof ApiResultError);
        assert.equal(error.code, "UNSUPPORTED_MEDIA");
        return true;
      },
    );
    assert.deepEqual(calls, []);
  });

  it("ingests a native dialog video selection then extracts its audio track", async () => {
    const calls: string[] = [];
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingAudioClient(calls),
      host: hostStub({
        showOpenFileDialog: async () => ["/Users/me/talk.mp4"],
      }),
      wsUrl: async (path) => `ws://desktop${path}`,
    });

    const asset = await runtime.mediaInput.choose(audioOrVideoChooseRequest);

    assert.equal(asset?.id, "audio-1");
    assert.deepEqual(calls, ["/Users/me/talk.mp4", "/api/assets/audio-1/extract-audio"]);
  });

  it("path-ingests a native dialog video selection for a video-only request", async () => {
    const calls: string[] = [];
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingIngestClient(calls),
      host: hostStub({
        showOpenFileDialog: async () => ["/Users/me/clip.mp4"],
      }),
      wsUrl: async (path) => `ws://desktop${path}`,
    });

    const asset = await runtime.mediaInput.choose(videoChooseRequest);

    assert.equal(asset?.id, "asset-1");
    assert.deepEqual(calls, ["/Users/me/clip.mp4"]);
  });

  it("rejects a video pick for an image-only request", async () => {
    const calls: string[] = [];
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingIngestClient(calls),
      host: hostStub({
        showOpenFileDialog: async () => ["/Users/me/talk.mp4"],
      }),
      wsUrl: async (path) => `ws://desktop${path}`,
    });

    await assert.rejects(
      () => runtime.mediaInput.choose(chooseRequest),
      (error: unknown) => {
        assert.ok(error instanceof ApiResultError);
        assert.equal(error.code, "UNSUPPORTED_MEDIA");
        return true;
      },
    );
    assert.deepEqual(calls, []);
  });

  it("writes a recorded wav to a temp path then path-ingests it", async () => {
    const calls: string[] = [];
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingAudioClient(calls),
      host: hostStub({
        writeTempFile: async () => "/tmp/recorded.wav",
      }),
      wsUrl: async (path) => `ws://desktop${path}`,
    });
    const file = new File([new Uint8Array([1, 2, 3])], "rec.wav", {
      type: "audio/wav",
    });

    const asset = await runtime.ingestRecordedFile(file);

    assert.equal(asset.id, "audio-1");
    assert.deepEqual(calls, ["/tmp/recorded.wav"]);
  });

  it("exposes live microphone capture on Desktop", () => {
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingAudioClient([]),
      host: hostStub(),
      wsUrl: async (path) => `ws://desktop${path}`,
    });

    assert.equal(runtime.microphoneAccess.kind, "live");
  });

  it("ingests a dropped video then extracts its audio track", async () => {
    const calls: string[] = [];
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingAudioClient(calls),
      host: hostStub({
        getPathForFile: () => "/Users/me/dropped.mp4",
      }),
      wsUrl: async (path) => `ws://desktop${path}`,
    });
    const file = new File(["frames"], "dropped.mp4", { type: "video/mp4" });

    assert.ok(runtime.ingestDroppedVideoAudio);
    const asset = await runtime.ingestDroppedVideoAudio(file);

    assert.equal(asset.id, "audio-1");
    assert.deepEqual(calls, [
      "/Users/me/dropped.mp4",
      "/api/assets/audio-1/extract-audio",
    ]);
  });

  it("loads packaged single-file seeds through path ingest", async () => {
    const cases = [
      ["image-to-video-start-frame", "/app/seed.png"],
      ["dolly-in-start-frame", "/app/dolly-in-seed.jpg"],
      ["retake-video", "/app/retake-seed.mp4"],
      ["extend-video", "/app/extend-seed.mp4"],
      ["day-to-night-video", "/app/day-to-night-reference.mp4"],
      ["alpha-gen-video", "/app/alpha-gen-reference.mp4"],
      ["deblur-video", "/app/deblur-reference.mp4"],
      ["colorization-video", "/app/colorization-reference.mp4"],
      ["clean-plate-video", "/app/clean-plate-reference.mp4"],
      ["decompression-video", "/app/decompression-reference.mp4"],
      ["water-simulation-video", "/app/water-simulation-reference.mp4"],
      ["layout-to-render-video", "/app/layout-to-render-reference.mp4"],
      ["layout-to-render-image", "/app/layout-to-render-first-frame.jpg"],
      ["restore-video", "/app/restore-reference.mp4"],
      ["restore-image", "/app/restore-first-frame.jpg"],
    ] as const;

    for (const [id, seedPath] of cases) {
      const calls: string[] = [];
      const runtime = createDesktopExploreRuntime(1, {
        api: recordingIngestClient(calls),
        host: hostStub(),
        wsUrl: async (path) => `ws://desktop${path}`,
      });
      const asset = await runtime.loadPackagedAsset(id);
      assert.equal(asset.id, "asset-1");
      assert.deepEqual(calls, [seedPath]);
    }
  });

  it("loads LoRA recipe seeds through the LoRA seed path", async () => {
    const calls: string[] = [];
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingIngestClient(calls),
      host: hostStub(),
      wsUrl: async (path) => `ws://desktop${path}`,
    });

    const asset = await runtime.loadPackagedAsset("cinemagraph-start-frame");
    assert.equal(asset.id, "asset-1");
    assert.deepEqual(calls, ["/app/cinemagraph-start-frame.jpg"]);
  });

  it("loads both packaged A2V seed slots through path ingest", async () => {
    const calls: string[] = [];
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingIngestClient(calls),
      host: hostStub(),
      wsUrl: async (path) => `ws://desktop${path}`,
    });

    const audio = await runtime.loadPackagedAsset("audio-to-video-audio");
    const startFrame = await runtime.loadPackagedAsset("audio-to-video-start-frame");

    assert.equal(audio.id, "asset-1");
    assert.equal(startFrame.id, "asset-1");
    assert.deepEqual(calls, ["/app/seed.mp3", "/app/a2v-seed.jpg"]);
  });

  it("keeps Finder reveal and file:// media URLs", async () => {
    const revealed: string[] = [];
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingIngestClient([]),
      host: hostStub({
        showItemInFolder: ({ filePath }) => {
          revealed.push(filePath);
        },
      }),
      wsUrl: async (path) => `ws://desktop${path}`,
    });

    assert.equal(
      runtime.mediaUrlForAsset({ id: "asset-1", path: "/tmp/photo.png" }),
      "file:///tmp/photo.png",
    );
    runtime.revealInFolder?.("/tmp/photo.png");
    assert.deepEqual(revealed, ["/tmp/photo.png"]);
    assert.equal(runtime.hoverPreview, true);
    const catalogInstall = runtime.catalogInstall;
    assert.ok(catalogInstall);
    assert.equal(typeof catalogInstall.openHuggingFaceAuth, "function");
    assert.equal(await runtime.wsUrl("/ws"), "ws://desktop/ws");
  });

  it("lists Desktop assets with host path fields through the shared API", async () => {
    const listed = {
      ...ingestedAsset,
      in_use: true,
      thumbnail_path: "/tmp/photo-thumb.jpg",
    };
    const runtime = createDesktopExploreRuntime(1, {
      api: createApiClient(async () => {
        return new Response(JSON.stringify({ items: [listed], next_cursor: null }), {
          status: 200,
        });
      }),
      host: hostStub(),
      wsUrl: async (path) => `ws://desktop${path}`,
    });

    const result = await runtime.api.listAssets({
      media_kind: "image",
      sort: "created_at-desc",
      q: "photo",
    });

    assert.equal(result.ok, true);
    if (!result.ok) return;
    const item = result.data.items[0];
    assert.ok(item);
    assert.equal(item.path, "/tmp/photo.png");
    assert.equal(item.in_use, true);
    assert.equal(item.thumbnail_path, "/tmp/photo-thumb.jpg");
    assert.equal(runtime.thumbUrlForAsset(item), "file:///tmp/photo-thumb.jpg");
  });

  it("builds file:// thumb URLs and exposes deleteAsset", () => {
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingIngestClient([]),
      host: hostStub(),
      wsUrl: async (path) => `ws://desktop${path}`,
    });
    assert.equal(
      runtime.thumbUrlForAsset({
        ...ingestedAsset,
        thumbnail_path: "/tmp/photo-thumb.jpg",
      }),
      "file:///tmp/photo-thumb.jpg",
    );
    assert.equal(runtime.thumbUrlForAsset(ingestedAsset), null);
    assert.equal(typeof runtime.deleteAsset, "function");
  });

  it("unwraps Desktop deleteAsset through the API client", async () => {
    const calls: Array<{ path: string; method: string | undefined }> = [];
    const runtime = createDesktopExploreRuntime(1, {
      api: createApiClient(async (path, init) => {
        calls.push({ path, method: init?.method });
        return new Response(JSON.stringify({ status: "ok" }), { status: 200 });
      }),
      host: hostStub(),
      wsUrl: async (path) => `ws://desktop${path}`,
    });

    assert.ok(runtime.deleteAsset);
    await runtime.deleteAsset("asset-1");
    assert.deepEqual(calls, [{ path: "/api/assets/asset-1", method: "DELETE" }]);
  });

  it("fetches waveform bytes through backendFetch as a blob URL", async () => {
    const previousFetch = globalThis.fetch;
    const previousWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
    const previousCreateObjectURL = URL.createObjectURL;
    const calls: string[] = [];
    resetBackendCredentials();
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: {
        electronAPI: {
          getBackend: async () => ({ url: "http://backend", token: "tok" }),
        },
      },
    });
    URL.createObjectURL = () => "blob:desktop-waveform";

    try {
      const runtime = createDesktopExploreRuntime(1, {
        api: recordingIngestClient([]),
        host: hostStub(),
        wsUrl: async (path) => `ws://desktop${path}`,
      });

      globalThis.fetch = (async (input: RequestInfo | URL) => {
        calls.push(String(input));
        return new Response(new Uint8Array([1, 2, 3]), { status: 200 });
      }) as typeof fetch;
      assert.equal(
        await runtime.waveformUrlForAsset({ id: "asset-1" }),
        "blob:desktop-waveform",
      );
      assert.deepEqual(calls, ["http://backend/api/assets/asset-1/bytes"]);

      globalThis.fetch = (async () => new Response("", { status: 404 })) as typeof fetch;
      assert.equal(await runtime.waveformUrlForAsset({ id: "missing" }), null);
    } finally {
      globalThis.fetch = previousFetch;
      URL.createObjectURL = previousCreateObjectURL;
      resetBackendCredentials();
      if (previousWindow) {
        Object.defineProperty(globalThis, "window", previousWindow);
      } else {
        Reflect.deleteProperty(globalThis, "window");
      }
    }
  });

  it("streams trimmer bytes off the asset-bytes route, not base64 over IPC", async () => {
    const requested: string[] = [];
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingAudioClient([]),
      host: hostStub(),
      wsUrl: async (path) => `ws://desktop${path}`,
      fetchBackend: async (path) => {
        requested.push(path);
        return new Response(new Blob(["RIFF"], { type: "audio/wav" }), {
          status: 200,
        });
      },
    });

    const source = await runtime.audioSourceUrl({
      ...ingestedAudio,
      media_kind: "audio",
    } as never);

    assert.deepEqual(requested, ["/api/assets/audio-1/bytes"]);
    assert.equal(source.url.startsWith("blob:"), true);
    assert.equal(typeof source.release, "function");
  });

  it("surfaces a missing asset instead of handing back an empty preview", async () => {
    const runtime = createDesktopExploreRuntime(1, {
      api: recordingAudioClient([]),
      host: hostStub(),
      wsUrl: async (path) => `ws://desktop${path}`,
      fetchBackend: async () => new Response("", { status: 404 }),
    });

    await assert.rejects(
      () =>
        runtime.audioSourceUrl({
          ...ingestedAudio,
          media_kind: "audio",
        } as never),
      (error: unknown) => {
        assert.ok(error instanceof ApiResultError);
        assert.equal(error.code, "ASSET_NOT_FOUND");
        return true;
      },
    );
  });
});
