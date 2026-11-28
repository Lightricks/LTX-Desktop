import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";

import { decodeAudio, releaseDecodedAudio } from "./waveformUtils.ts";

const realFetch = globalThis.fetch;
const globals = globalThis as unknown as {
  fetch: unknown;
  AudioContext: unknown;
};

/** Counts network reads so the test can tell a cache hit from a re-decode. */
function stubAudioDecoding() {
  let fetches = 0;
  globals.fetch = async () => {
    fetches += 1;
    return new Response(new ArrayBuffer(8));
  };
  globals.AudioContext = class {
    async decodeAudioData(): Promise<unknown> {
      return { sampleRate: 48_000, length: 0 };
    }
  };
  return () => fetches;
}

afterEach(() => {
  globals.fetch = realFetch;
  delete globals.AudioContext;
});

describe("decoded audio cache", () => {
  it("decodes a src once while it is still reachable", async () => {
    const fetches = stubAudioDecoding();

    await decodeAudio("blob:cached");
    await decodeAudio("blob:cached");

    assert.equal(fetches(), 1);
  });

  it("drops the buffer once the URL is released so reopens do not pin PCM", async () => {
    const fetches = stubAudioDecoding();

    await decodeAudio("blob:released");
    releaseDecodedAudio("blob:released");
    await decodeAudio("blob:released");

    assert.equal(fetches(), 2);
  });
});
