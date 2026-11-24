import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { resolveResultFileAction } from "./resultFileAction.ts";

type Output = { mime_type: string; name: string; path?: string | null };

const WEBM: Output = { mime_type: "video/webm", name: "cutout.webm", path: "/out/cutout.webm" };
const MATTE: Output = { mime_type: "video/mp4", name: "matte.mp4", path: "/out/matte.mp4" };
const GIF: Output = { mime_type: "image/gif", name: "output.gif", path: "/out/output.gif" };
const VIDEO: Output = { mime_type: "video/mp4", name: "clip.mp4", path: "/out/clip.mp4" };

const mediaUrlForAsset = (output: Output) => `/bytes/${output.name}`;

function resolve(
  outputs: Output[],
  options: { isCutoutFeature?: boolean; reveal?: boolean; played?: Output } = {},
) {
  const calls: string[] = [];
  const action = resolveResultFileAction({
    generation: { outputs },
    playedOutput: options.played ?? outputs[0],
    isCutoutFeature: options.isCutoutFeature ?? false,
    revealInFolder: options.reveal ? (path) => calls.push(`reveal ${path}`) : null,
    mediaUrlForAsset,
    download: (output) => calls.push(`download ${output.name}`),
  });
  return { action, calls };
}

describe("resolveResultFileAction", () => {
  it("downloads the one output of any feature where the host has no file manager", () => {
    const { action, calls } = resolve([VIDEO]);
    assert.equal(action?.kind, "download");
    assert.equal(action?.files.length, 1);
    action?.files[0]?.onSelect();
    assert.deepEqual(calls, ["download clip.mp4"]);
  });

  it("reveals the one output where the host has a file manager", () => {
    const { action, calls } = resolve([VIDEO], { reveal: true });
    assert.equal(action?.kind, "reveal");
    action?.files[0]?.onSelect();
    assert.deepEqual(calls, ["reveal /out/clip.mp4"]);
  });

  it("lists the WebM and the matte of a baked cutout, to download", () => {
    const { action, calls } = resolve([WEBM, MATTE], { isCutoutFeature: true });
    assert.equal(action?.kind, "download");
    assert.deepEqual(
      action?.files.map((file) => file.label),
      ["Cutout with alpha (WebM)", "Matte (MP4)"],
    );
    action?.files[1]?.onSelect();
    assert.deepEqual(calls, ["download matte.mp4"]);
  });

  it("lists the GIF last when the run has one", () => {
    const { action, calls } = resolve([WEBM, MATTE, GIF], { isCutoutFeature: true });
    assert.deepEqual(
      action?.files.map((file) => file.label),
      ["Cutout with alpha (WebM)", "Matte (MP4)", "GIF"],
    );
    action?.files[2]?.onSelect();
    assert.deepEqual(calls, ["download output.gif"]);
  });

  it("lists the WebM and the matte of a baked cutout, to reveal", () => {
    const { action, calls } = resolve([WEBM, MATTE], { isCutoutFeature: true, reveal: true });
    assert.equal(action?.kind, "reveal");
    action?.files[0]?.onSelect();
    assert.deepEqual(calls, ["reveal /out/cutout.webm"]);
  });

  it("offers the one played output of a cutout run from before the bake", () => {
    const { action } = resolve([MATTE], { isCutoutFeature: true });
    assert.equal(action?.files.length, 1);
  });

  it("offers one file for a feature that is not a cutout recipe, whatever its outputs are", () => {
    const { action } = resolve([WEBM, MATTE], { isCutoutFeature: false });
    assert.equal(action?.files.length, 1);
  });

  it("skips a file with no path when it reveals", () => {
    const { action } = resolve([WEBM, { ...MATTE, path: null }], {
      isCutoutFeature: true,
      reveal: true,
    });
    assert.deepEqual(
      action?.files.map((file) => file.label),
      ["Cutout with alpha (WebM)"],
    );
  });

  it("returns null when there is no output", () => {
    assert.equal(resolve([], { played: undefined }).action, null);
  });
});
