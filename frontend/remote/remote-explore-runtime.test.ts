import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { createApiClient } from "../lib/api-client.ts";
import { ApiResultError } from "../ltx-io/lib/unwrapApiResult.ts";
import { REMOTE_GENERATION_POLLING_POLICY } from "../ltx-io/runtime/generationPollingPolicy.ts";
import {
  AUDIO_ACCEPT,
  AUDIO_EXTENSIONS,
} from "../ltx-io/screens/Feature/fields/audioAssetInput.ts";
import {
  IMAGE_ACCEPT,
  IMAGE_EXTENSIONS,
} from "../ltx-io/screens/Feature/fields/imageAssetInput.ts";

import { createRemoteExploreRuntime } from "./remote-explore-runtime.ts";

const uploadedAsset = {
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
  bytes_url: "/api/assets/asset-1/bytes?exp=1&sig=abc",
};

const uploadedAudio = {
  created_at: 1,
  id: "audio-1",
  media_kind: "audio",
  metadata: { mediaType: "audio", metadata: { durationMs: 4000 } },
  mime_type: "audio/wav",
  name: "clip.wav",
  origin: "uploaded",
  bytes_url: "/api/assets/audio-1/bytes?exp=1&sig=abc",
};

const unusedPackagedFile = async () => {
  throw new Error("unused packaged file");
};

const chooseRequest = {
  title: "Start Frame",
  accept: IMAGE_ACCEPT,
  extensions: IMAGE_EXTENSIONS,
  fileInput: { click() {} } as HTMLInputElement,
};

const audioChooseRequest = {
  title: "Audio",
  accept: AUDIO_ACCEPT,
  extensions: AUDIO_EXTENSIONS,
  fileInput: { click() {} } as HTMLInputElement,
};

function recordingUploadClient(
  calls: Array<{ path: string; file: unknown }>,
  asset: unknown = uploadedAsset,
) {
  return createApiClient(async (path, init) => {
    const body = init?.body;
    calls.push({
      path,
      file: body instanceof FormData ? body.get("file") : null,
    });
    return new Response(JSON.stringify(asset), { status: 200 });
  });
}

describe("Remote Explore media input", () => {
  it("uploads a chosen mounted-input file as multipart", async () => {
    const calls: Array<{ path: string; file: unknown }> = [];
    const file = new File(["pixels"], "photo.png", { type: "image/png" });
    const waited: HTMLInputElement[] = [];
    const runtime = createRemoteExploreRuntime({
      api: recordingUploadClient(calls),
      loadPackagedFile: unusedPackagedFile,
      waitForFile: async (input) => {
        waited.push(input);
        return file;
      },
      wsUrl: async (path) => `ws://remote${path}`,
    });

    const asset = await runtime.mediaInput.choose(chooseRequest);

    assert.equal(asset?.id, "asset-1");
    assert.equal(asset && "path" in asset ? asset.path : undefined, undefined);
    assert.deepEqual(waited, [chooseRequest.fileInput]);
    assert.deepEqual(calls, [{ path: "/api/assets/upload", file }]);
  });

  it("throws when choose is called without a mounted file input", async () => {
    const runtime = createRemoteExploreRuntime({
      api: recordingUploadClient([]),
      loadPackagedFile: unusedPackagedFile,
      wsUrl: async (path) => `ws://remote${path}`,
    });

    await assert.rejects(
      () => runtime.mediaInput.choose({ ...chooseRequest, fileInput: null }),
      /mounted file input/,
    );
  });

  it("uploads a dropped file as multipart", async () => {
    const calls: Array<{ path: string; file: unknown }> = [];
    const file = new File(["pixels"], "photo.png", { type: "image/png" });
    const runtime = createRemoteExploreRuntime({
      api: recordingUploadClient(calls),
      loadPackagedFile: unusedPackagedFile,
      wsUrl: async (path) => `ws://remote${path}`,
    });

    const asset = await runtime.mediaInput.ingestDroppedFile(file);

    assert.equal(asset.id, "asset-1");
    assert.deepEqual(calls, [{ path: "/api/assets/upload", file }]);
  });

  it("uploads a chosen mounted-input audio file as multipart", async () => {
    const calls: Array<{ path: string; file: unknown }> = [];
    const file = new File(["pcm"], "clip.wav", { type: "audio/wav" });
    const waited: HTMLInputElement[] = [];
    const runtime = createRemoteExploreRuntime({
      api: recordingUploadClient(calls, uploadedAudio),
      loadPackagedFile: unusedPackagedFile,
      waitForFile: async (input) => {
        waited.push(input);
        return file;
      },
      wsUrl: async (path) => `ws://remote${path}`,
    });

    const asset = await runtime.mediaInput.choose(audioChooseRequest);

    assert.equal(asset?.id, "audio-1");
    assert.deepEqual(waited, [audioChooseRequest.fileInput]);
    assert.deepEqual(calls, [{ path: "/api/assets/upload", file }]);
  });

  it("rejects a chosen non-audio file before uploading", async () => {
    const calls: Array<{ path: string; file: unknown }> = [];
    const file = new File(["frames"], "clip.mp4", { type: "video/mp4" });
    const runtime = createRemoteExploreRuntime({
      api: recordingUploadClient(calls, uploadedAudio),
      loadPackagedFile: unusedPackagedFile,
      waitForFile: async () => file,
      wsUrl: async (path) => `ws://remote${path}`,
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

  it("uploads a dropped audio file as multipart", async () => {
    const calls: Array<{ path: string; file: unknown }> = [];
    const file = new File(["pcm"], "clip.wav", { type: "audio/wav" });
    const runtime = createRemoteExploreRuntime({
      api: recordingUploadClient(calls, uploadedAudio),
      loadPackagedFile: unusedPackagedFile,
      wsUrl: async (path) => `ws://remote${path}`,
    });

    const asset = await runtime.mediaInput.ingestDroppedFile(file);

    assert.equal(asset.id, "audio-1");
    assert.deepEqual(calls, [{ path: "/api/assets/upload", file }]);
  });

  it("uploads a recorded wav as multipart", async () => {
    const calls: Array<{ path: string; file: unknown }> = [];
    const file = new File(["pcm"], "rec.wav", { type: "audio/wav" });
    const runtime = createRemoteExploreRuntime({
      api: recordingUploadClient(calls, uploadedAudio),
      loadPackagedFile: unusedPackagedFile,
      wsUrl: async (path) => `ws://remote${path}`,
    });

    const asset = await runtime.ingestRecordedFile(file);

    assert.equal(asset.id, "audio-1");
    assert.deepEqual(calls, [{ path: "/api/assets/upload", file }]);
  });

  it("does not expose in-app recording; Import is the file path", () => {
    const runtime = createRemoteExploreRuntime({
      api: createApiClient(async () => new Response("{}", { status: 200 })),
      loadPackagedFile: unusedPackagedFile,
      wsUrl: async (path) => `ws://remote${path}`,
    });

    assert.equal(runtime.microphoneAccess, null);
  });

  it("cannot extract audio from a dropped video and serves bytes URLs to the trimmer", async () => {
    const runtime = createRemoteExploreRuntime({
      api: createApiClient(async () => new Response("{}", { status: 200 })),
      loadPackagedFile: unusedPackagedFile,
      wsUrl: async (path) => `ws://remote${path}`,
    });

    assert.equal(runtime.ingestDroppedVideoAudio, null);
    assert.deepEqual(await runtime.audioSourceUrl({ ...uploadedAudio } as never), {
      url: "/api/assets/audio-1/bytes?exp=1&sig=abc",
    });
    await assert.rejects(
      () => runtime.audioSourceUrl({ id: "audio-1" } as never),
      /missing its media URL/,
    );
  });

  it("has no Desktop modelsVersion or Finder reveal, and uses host-signed media URLs", async () => {
    const runtime = createRemoteExploreRuntime({
      api: createApiClient(async () => new Response("{}", { status: 200 })),
      loadPackagedFile: unusedPackagedFile,
      wsUrl: async (path) => `ws://remote${path}`,
    });

    assert.equal(runtime.modelsVersion, null);
    assert.equal(runtime.revealInFolder, null);
    assert.equal(runtime.catalogInstall, null);
    assert.equal(runtime.hoverPreview, true);
    assert.equal(typeof runtime.loadPackagedAsset, "function");
    assert.equal(runtime.mediaUrlForAsset({ id: "asset-9" }), null);
    assert.equal(
      runtime.mediaUrlForAsset({
        id: "asset-9",
        bytes_url: "/api/assets/asset-9/bytes?exp=1&sig=abc",
      }),
      "/api/assets/asset-9/bytes?exp=1&sig=abc",
    );
    assert.equal(await runtime.wsUrl("/ws"), "ws://remote/ws");
  });

  it("uploads the packaged I2V still and previews it from the bytes URL", async () => {
    const calls: Array<{ path: string; file: unknown }> = [];
    const seedFile = new File(["seed-pixels"], "image-to-video-input.jpg", {
      type: "image/jpeg",
    });
    const loadedIds: string[] = [];
    const runtime = createRemoteExploreRuntime({
      api: recordingUploadClient(calls),
      loadPackagedFile: async (id) => {
        loadedIds.push(id);
        return seedFile;
      },
      wsUrl: async (path) => `ws://remote${path}`,
    });

    const asset = await runtime.loadPackagedAsset("image-to-video-start-frame");

    assert.equal(asset.id, "asset-1");
    assert.deepEqual(loadedIds, ["image-to-video-start-frame"]);
    assert.deepEqual(calls, [{ path: "/api/assets/upload", file: seedFile }]);
    assert.equal(runtime.mediaUrlForAsset(asset), uploadedAsset.bytes_url);
    assert.equal(runtime.mediaUrlForAsset({ id: "other-asset" }), null);
  });

  it("uploads both packaged A2V catalog assets", async () => {
    const calls: Array<{ path: string; file: unknown }> = [];
    const seedAudio = new File(["seed-pcm"], "audio-to-video-input-v2.mp3", {
      type: "audio/mpeg",
    });
    const seedImage = new File(["seed-pixels"], "audio-to-video-input.jpg", {
      type: "image/jpeg",
    });
    const loadedIds: string[] = [];
    const runtime = createRemoteExploreRuntime({
      api: recordingUploadClient(calls, uploadedAudio),
      loadPackagedFile: async (id) => {
        loadedIds.push(id);
        return id === "audio-to-video-audio" ? seedAudio : seedImage;
      },
      wsUrl: async (path) => `ws://remote${path}`,
    });

    const audio = await runtime.loadPackagedAsset("audio-to-video-audio");
    const startFrame = await runtime.loadPackagedAsset("audio-to-video-start-frame");

    assert.equal(audio.id, "audio-1");
    assert.equal(startFrame.id, "audio-1");
    assert.deepEqual(loadedIds, ["audio-to-video-audio", "audio-to-video-start-frame"]);
    assert.deepEqual(calls, [
      { path: "/api/assets/upload", file: seedAudio },
      { path: "/api/assets/upload", file: seedImage },
    ]);
  });

  it("uploads the packaged retake video", async () => {
    const calls: Array<{ path: string; file: unknown }> = [];
    const seedFile = new File(["seed-frames"], "retake-input.mp4", {
      type: "video/mp4",
    });
    const loadedIds: string[] = [];
    const runtime = createRemoteExploreRuntime({
      api: recordingUploadClient(calls),
      loadPackagedFile: async (id) => {
        loadedIds.push(id);
        return seedFile;
      },
      wsUrl: async (path) => `ws://remote${path}`,
    });

    const asset = await runtime.loadPackagedAsset("retake-video");

    assert.equal(asset.id, "asset-1");
    assert.deepEqual(loadedIds, ["retake-video"]);
    assert.deepEqual(calls, [{ path: "/api/assets/upload", file: seedFile }]);
  });

  it("uploads the packaged extend video", async () => {
    const calls: Array<{ path: string; file: unknown }> = [];
    const seedFile = new File(["seed-frames"], "extend-input-v3.mp4", {
      type: "video/mp4",
    });
    const loadedIds: string[] = [];
    const runtime = createRemoteExploreRuntime({
      api: recordingUploadClient(calls),
      loadPackagedFile: async (id) => {
        loadedIds.push(id);
        return seedFile;
      },
      wsUrl: async (path) => `ws://remote${path}`,
    });

    const asset = await runtime.loadPackagedAsset("extend-video");

    assert.equal(asset.id, "asset-1");
    assert.deepEqual(loadedIds, ["extend-video"]);
    assert.deepEqual(calls, [{ path: "/api/assets/upload", file: seedFile }]);
  });

  it("uses the 2s Remote generation polling policy", async () => {
    const runtime = createRemoteExploreRuntime({
      api: createApiClient(async () => new Response("{}", { status: 200 })),
      loadPackagedFile: unusedPackagedFile,
      wsUrl: async (path) => `ws://remote${path}`,
    });

    assert.equal(runtime.generationPolling, REMOTE_GENERATION_POLLING_POLICY);
    assert.equal(runtime.generationPolling.queueActiveIntervalMs, 2000);
  });

  it("lists path-free Remote assets through the shared API", async () => {
    const listed = {
      ...uploadedAsset,
      in_use: false,
      has_thumbnail: true,
      thumbnail_url: "/api/assets/asset-1/thumbnail/bytes?exp=1&sig=abc",
    };
    const runtime = createRemoteExploreRuntime({
      api: createApiClient(async () => {
        return new Response(JSON.stringify({ items: [listed], next_cursor: null }), {
          status: 200,
        });
      }),
      loadPackagedFile: unusedPackagedFile,
      wsUrl: async (path) => `ws://remote${path}`,
    });

    const result = await runtime.api.listAssets({
      sort: "created_at-desc",
      q: "",
    });

    assert.equal(result.ok, true);
    if (!result.ok) return;
    const item = result.data.items[0];
    assert.ok(item);
    assert.equal(item.id, "asset-1");
    assert.equal("path" in item ? item.path : undefined, undefined);
    assert.equal(item.in_use, false);
    assert.equal(item.has_thumbnail, true);
    assert.equal(
      runtime.thumbUrlForAsset(item),
      "/api/assets/asset-1/thumbnail/bytes?exp=1&sig=abc",
    );
  });

  it("persists the Enhance provider choice in localStorage", () => {
    const store = new Map<string, string>();
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: {
        getItem: (key: string) => store.get(key) ?? null,
        setItem: (key: string, value: string) => {
          store.set(key, value);
        },
        removeItem: (key: string) => {
          store.delete(key);
        },
      },
    });
    const runtime = createRemoteExploreRuntime({
      api: createApiClient(async () => new Response("{}", { status: 200 })),
      loadPackagedFile: unusedPackagedFile,
      wsUrl: async (path) => `ws://remote${path}`,
    });

    assert.equal(runtime.readEnhanceProviderPreference?.(), null);
    runtime.persistEnhanceProviderPreference?.("local");
    assert.equal(runtime.readEnhanceProviderPreference?.(), "local");
  });

  it("does not expose Remote delete and uses signed thumbnail URLs", () => {
    const runtime = createRemoteExploreRuntime({
      api: createApiClient(async () => new Response("{}", { status: 200 })),
      loadPackagedFile: unusedPackagedFile,
      wsUrl: async (path) => `ws://remote${path}`,
    });
    assert.equal(runtime.deleteAsset, null);
    assert.equal(runtime.revealInFolder, null);
    assert.equal(runtime.hoverPreview, true);
    assert.equal(
      runtime.thumbUrlForAsset({
        ...uploadedAsset,
        has_thumbnail: true,
        thumbnail_url: "/api/assets/asset-1/thumbnail/bytes?exp=1&sig=abc",
      }),
      "/api/assets/asset-1/thumbnail/bytes?exp=1&sig=abc",
    );
    assert.equal(runtime.thumbUrlForAsset(uploadedAsset), null);
    assert.equal(
      runtime.thumbUrlForAsset({ ...uploadedAsset, has_thumbnail: false }),
      null,
    );
  });

  it("fetches waveform bytes through the signed bytes URL as a blob URL", async () => {
    const previousFetch = globalThis.fetch;
    const previousCreateObjectURL = URL.createObjectURL;
    const calls: string[] = [];
    URL.createObjectURL = () => "blob:remote-waveform";
    const runtime = createRemoteExploreRuntime({
      api: createApiClient(async () => new Response("{}", { status: 200 })),
      loadPackagedFile: unusedPackagedFile,
      wsUrl: async (path) => `ws://remote${path}`,
    });

    try {
      globalThis.fetch = (async (input: RequestInfo | URL) => {
        calls.push(String(input));
        return new Response(new Uint8Array([1, 2, 3]), { status: 200 });
      }) as typeof fetch;
      assert.equal(
        await runtime.waveformUrlForAsset({
          id: "asset-1",
          bytes_url: "/api/assets/asset-1/bytes?exp=1&sig=abc",
        }),
        "blob:remote-waveform",
      );
      assert.deepEqual(calls, ["/api/assets/asset-1/bytes?exp=1&sig=abc"]);

      assert.equal(await runtime.waveformUrlForAsset({ id: "missing" }), null);
    } finally {
      globalThis.fetch = previousFetch;
      URL.createObjectURL = previousCreateObjectURL;
    }
  });
});
