import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { createApiClient } from "../../lib/api-client.ts";
import {
  fetchAsset,
  fetchGenerations,
  fetchVideoGenerationModelSpecs,
} from "./generationQueryKeys.ts";

const asset = {
  created_at: 1,
  id: "asset-1",
  media_kind: "image",
  metadata: {
    mediaType: "image",
    metadata: { width: 8, height: 8 },
  },
  mime_type: "image/png",
  name: "photo.png",
  origin: "uploaded",
};

const generation = {
  id: "gen-1",
  created_at: 1,
  feature: "text-to-video",
  status: "succeeded",
  spec: {},
  outputs: [],
};

const specs = { api_models: [], local_models: [] };

describe("Explore hooks bound API helpers", () => {
  it("send generation list/specs/asset reads through the given client", async () => {
    const callsA: string[] = [];
    const callsB: string[] = [];
    const apiA = createApiClient(async (path) => {
      callsA.push(path);
      if (path.startsWith("/api/generations")) {
        return new Response(JSON.stringify([generation]), { status: 200 });
      }
      if (path.startsWith("/api/assets/")) {
        return new Response(JSON.stringify(asset), { status: 200 });
      }
      return new Response(JSON.stringify(specs), { status: 200 });
    });
    const apiB = createApiClient(async (path) => {
      callsB.push(path);
      return new Response(JSON.stringify(specs), { status: 200 });
    });

    const listed = await fetchGenerations(apiA, "text-to-video");
    const loaded = await fetchAsset(apiA, "asset-1");
    await fetchVideoGenerationModelSpecs(apiB);

    assert.equal(listed[0]?.id, "gen-1");
    assert.equal(loaded.id, "asset-1");
    assert.deepEqual(callsA, [
      "/api/generations?feature=text-to-video",
      "/api/assets/asset-1",
    ]);
    assert.deepEqual(callsB, ["/api/generate/models-specs"]);
  });

  it("sends prompt-enhancer status and enhance through the given client", async () => {
    const calls: string[] = [];
    const api = createApiClient(async (path, init) => {
      calls.push(`${init?.method ?? "GET"} ${path}`);
      if (path === "/api/prompt-enhancer") {
        return new Response(
          JSON.stringify({
            exploreAutoEnhancePrompts: true,
            hasGeminiApiKey: true,
            localEnhancementSupported: false,
            defaultProvider: "api",
            canToggleProvider: false,
            showManualEnhance: true,
          }),
          { status: 200 },
        );
      }
      return new Response(JSON.stringify({ enhancedPrompt: "rewritten" }), {
        status: 200,
      });
    });

    const status = await api.getPromptEnhancer();
    const enhanced = await api.enhancePrompt({
      prompt: "a cat",
      provider: "api",
      mediaType: "video",
    });

    assert.equal(status.ok, true);
    if (status.ok) {
      assert.equal(status.data.defaultProvider, "api");
    }
    assert.equal(enhanced.ok, true);
    if (enhanced.ok) {
      assert.equal(enhanced.data.enhancedPrompt, "rewritten");
    }
    assert.deepEqual(calls, [
      "GET /api/prompt-enhancer",
      "POST /api/enhance-prompt",
    ]);
  });
});
