import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import { electronAPISchemas } from "../shared/electron-api-schema.ts";
import {
  PACKAGED_SEED_KINDS,
  packagedSeedSpec,
  resolvePackagedSeedPath,
  type PackagedSeedKind,
} from "./explore-seed.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const JPEG_HEADER = [0xff, 0xd8, 0xff];
const MP4_FTYP = [...Buffer.from("ftyp")];

const SEED_CASES: readonly {
  kind: PackagedSeedKind;
  suffix: RegExp;
  header: { offset: number; bytes: number[] };
}[] = [
  {
    kind: "image-to-video-start-frame",
    suffix: /image-to-video-input\.jpg$/,
    header: { offset: 0, bytes: JPEG_HEADER },
  },
  {
    kind: "dolly-in-start-frame",
    suffix: /dolly-in-start-frame\.jpg$/,
    header: { offset: 0, bytes: JPEG_HEADER },
  },
  {
    kind: "retake-video",
    suffix: /retake-input\.mp4$/,
    header: { offset: 4, bytes: MP4_FTYP },
  },
  {
    kind: "extend-video",
    suffix: /extend-input-v3\.mp4$/,
    header: { offset: 4, bytes: MP4_FTYP },
  },
  {
    kind: "day-to-night-video",
    suffix: /day-to-night-reference\.mp4$/,
    header: { offset: 4, bytes: MP4_FTYP },
  },
  {
    kind: "alpha-gen-video",
    suffix: /alpha-gen-reference\.mp4$/,
    header: { offset: 4, bytes: MP4_FTYP },
  },
  {
    kind: "deblur-video",
    suffix: /deblur-reference\.mp4$/,
    header: { offset: 4, bytes: MP4_FTYP },
  },
  {
    kind: "colorization-video",
    suffix: /colorization-reference\.mp4$/,
    header: { offset: 4, bytes: MP4_FTYP },
  },
  {
    kind: "clean-plate-video",
    suffix: /clean-plate-reference\.mp4$/,
    header: { offset: 4, bytes: MP4_FTYP },
  },
  {
    kind: "decompression-video",
    suffix: /decompression-reference\.mp4$/,
    header: { offset: 4, bytes: MP4_FTYP },
  },
  {
    kind: "water-simulation-video",
    suffix: /water-simulation-reference\.mp4$/,
    header: { offset: 4, bytes: MP4_FTYP },
  },
  {
    kind: "layout-to-render-video",
    suffix: /layout-to-render-reference\.mp4$/,
    header: { offset: 4, bytes: MP4_FTYP },
  },
  {
    kind: "layout-to-render-image",
    suffix: /layout-to-render-first-frame\.jpg$/,
    header: { offset: 0, bytes: JPEG_HEADER },
  },
  {
    kind: "restore-video",
    suffix: /restore-reference\.mp4$/,
    header: { offset: 4, bytes: MP4_FTYP },
  },
  {
    kind: "restore-image",
    suffix: /restore-first-frame\.jpg$/,
    header: { offset: 0, bytes: JPEG_HEADER },
  },
];

describe("packaged seed IPC", () => {
  it("allowlists kind and ignores caller-supplied resource paths", () => {
    const schema = electronAPISchemas.getPackagedSeedPath;
    for (const kind of PACKAGED_SEED_KINDS) {
      assert.deepEqual(schema.input.parse({ kind }), { kind });
      assert.equal(typeof schema.output.parse("/tmp/seed"), "string");
    }
    assert.throws(() => schema.input.parse({}));
    const parsed = schema.input.parse({
      kind: "retake-video",
      resourcePath: "/etc/passwd",
      relativePath: "secret.mp4",
    });
    assert.deepEqual(parsed, { kind: "retake-video" });
    assert.equal("resourcePath" in parsed, false);
  });
});

describe("packaged seed path resolution", () => {
  for (const seed of SEED_CASES) {
    const spec = packagedSeedSpec(seed.kind);

    it(`resolves ${seed.kind} from repo resources while unpackaged`, () => {
      const resolved = resolvePackagedSeedPath({
        kind: seed.kind,
        isPackaged: false,
        projectRoot: "/repo",
        resourcesPath: "/Applications/LTX Desktop.app/Contents/Resources",
      });
      assert.equal(
        resolved,
        path.join("/repo", "resources", spec.dir, spec.filename),
      );
    });

    it(`resolves only the allowlisted packaged ${seed.kind} file`, () => {
      const resourcesPath = "/packaged/Resources";
      const resolved = resolvePackagedSeedPath({
        kind: seed.kind,
        isPackaged: true,
        projectRoot: "/repo",
        resourcesPath,
      });
      assert.equal(
        resolved,
        path.join(resourcesPath, spec.dir, spec.filename),
      );
      assert.equal(resolved.startsWith(resourcesPath + path.sep), true);
      assert.match(resolved, seed.suffix);
    });

    it(`keeps the ${seed.kind} file in repo resources`, () => {
      const seedPath = path.join(repoRoot, "resources", spec.dir, spec.filename);
      assert.equal(fs.existsSync(seedPath), true);
      const header = fs
        .readFileSync(seedPath)
        .subarray(seed.header.offset, seed.header.offset + seed.header.bytes.length);
      assert.deepEqual([...header], seed.header.bytes);
    });
  }

  it("packages explore-assets under extraResources", () => {
    const yaml = fs.readFileSync(path.join(repoRoot, "electron-builder.yml"), "utf8");
    assert.match(yaml, /from:\s+resources\/explore-assets/);
    assert.match(yaml, /to:\s+explore-assets/);
  });
});
