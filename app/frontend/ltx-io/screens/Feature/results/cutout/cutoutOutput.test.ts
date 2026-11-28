import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  cutoutFileTargets,
  hasCutoutOutput,
  isAlphaWebmMime,
  pickPlayback,
  supportsAlphaWebm,
} from "./cutoutOutput.ts";

const CHROME_MAC =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36";
const ELECTRON =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) LTX-Desktop/1.2.7 Chrome/130.0.0.0 Electron/33.0.0 Safari/537.36";
const FIREFOX = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:131.0) Gecko/20100101 Firefox/131.0";
const SAFARI_MAC =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15";
const SAFARI_IOS =
  "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1";
const CHROME_IOS =
  "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/130.0.0.0 Mobile/15E148 Safari/604.1";
const CHROME_ANDROID =
  "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36";

const CUTOUT = { outputs: [{ mime_type: "video/webm" }, { mime_type: "video/mp4" }] };

describe("hasCutoutOutput", () => {
  it("is true when the first output is the WebM cutout", () => {
    assert.equal(hasCutoutOutput(CUTOUT), true);
  });

  it("plays a run from before the bake, or with no output, as a plain video", () => {
    assert.equal(hasCutoutOutput({ outputs: [{ mime_type: "video/mp4" }] }), false);
    assert.equal(hasCutoutOutput({ outputs: [] }), false);
  });
});

describe("isAlphaWebmMime", () => {
  it("is true for WebM only", () => {
    assert.equal(isAlphaWebmMime("video/webm"), true);
    assert.equal(isAlphaWebmMime("video/mp4"), false);
  });
});

describe("supportsAlphaWebm", () => {
  it("is true for Chromium, Electron and Firefox", () => {
    for (const agent of [CHROME_MAC, ELECTRON, FIREFOX, CHROME_ANDROID]) {
      assert.equal(supportsAlphaWebm(agent), true, agent);
    }
  });

  it("is false for Safari and for every browser on iOS", () => {
    for (const agent of [SAFARI_MAC, SAFARI_IOS, CHROME_IOS]) {
      assert.equal(supportsAlphaWebm(agent), false, agent);
    }
  });
});

describe("cutoutFileTargets", () => {
  it("lists the WebM, then the matte, of a baked cutout", () => {
    assert.deepEqual(cutoutFileTargets(CUTOUT, true), [
      { kind: "alpha", output: CUTOUT.outputs[0] },
      { kind: "matte", output: CUTOUT.outputs[1] },
    ]);
  });

  it("adds the GIF when the run has one", () => {
    const withGif = { outputs: [...CUTOUT.outputs, { mime_type: "image/gif" }] };
    assert.deepEqual(
      cutoutFileTargets(withGif, true).map((target) => target.kind),
      ["alpha", "matte", "gif"],
    );
  });

  it("does not take a third output that is not a GIF", () => {
    const other = { outputs: [...CUTOUT.outputs, { mime_type: "video/mp4" }] };
    assert.deepEqual(
      cutoutFileTargets(other, true).map((target) => target.kind),
      ["alpha", "matte"],
    );
  });

  it("lists nothing for a feature that is not a cutout recipe", () => {
    assert.deepEqual(cutoutFileTargets(CUTOUT, false), []);
  });

  it("lists nothing for a run from before the bake", () => {
    const plain = { outputs: [{ mime_type: "video/mp4" }] };
    assert.deepEqual(cutoutFileTargets(plain, true), []);
  });

  it("lists nothing when the matte is missing", () => {
    assert.deepEqual(cutoutFileTargets({ outputs: [{ mime_type: "video/webm" }] }, true), []);
  });
});

describe("pickPlayback", () => {
  it("shows the cutout where alpha plays", () => {
    assert.deepEqual(pickPlayback(CUTOUT, ELECTRON, true), {
      output: CUTOUT.outputs[0],
      asCutout: true,
    });
  });

  it("falls back to the matte as a plain video where alpha does not play", () => {
    assert.deepEqual(pickPlayback(CUTOUT, SAFARI_IOS, true), {
      output: CUTOUT.outputs[1],
      asCutout: false,
    });
  });

  it("plays a cutout recipe run from before the bake as a plain video", () => {
    const plain = { outputs: [{ mime_type: "video/mp4" }] };
    assert.deepEqual(pickPlayback(plain, SAFARI_IOS, true), {
      output: plain.outputs[0],
      asCutout: false,
    });
  });

  it("never treats a run of another feature as a cutout, whatever its first output is", () => {
    assert.deepEqual(pickPlayback(CUTOUT, ELECTRON, false), {
      output: CUTOUT.outputs[0],
      asCutout: false,
    });
  });

  it("leaves other runs alone", () => {
    const plain = { outputs: [{ mime_type: "video/mp4" }] };
    assert.deepEqual(pickPlayback(plain, SAFARI_IOS, false), {
      output: plain.outputs[0],
      asCutout: false,
    });
  });
});
