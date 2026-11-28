import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import { electronAPISchemas } from "../shared/electron-api-schema.ts";
import {
  EXPLORE_AUDIO_TO_VIDEO_SEED_AUDIO_FILENAME,
  EXPLORE_AUDIO_TO_VIDEO_SEED_DIR,
  EXPLORE_AUDIO_TO_VIDEO_SEED_IMAGE_FILENAME,
  resolveExploreAudioToVideoSeedPaths,
} from "./explore-a2v-seed.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

describe("explore A2V seed IPC", () => {
  it("does not accept a caller-supplied resource path", () => {
    const schema = electronAPISchemas.getExploreAudioToVideoSeedPaths;
    const parsed = schema.input.parse({
      resourcePath: "/etc/passwd",
      relativePath: "secret.mp3",
    });
    assert.deepEqual(parsed, {});
    assert.equal("resourcePath" in parsed, false);
  });
});

describe("explore A2V seed path resolution", () => {
  it("resolves the repo resources files while unpackaged", () => {
    const resolved = resolveExploreAudioToVideoSeedPaths({
      isPackaged: false,
      projectRoot: "/repo",
      resourcesPath: "/Applications/LTX Desktop.app/Contents/Resources",
    });
    assert.deepEqual(resolved, {
      audio: path.join(
        "/repo",
        "resources",
        EXPLORE_AUDIO_TO_VIDEO_SEED_DIR,
        EXPLORE_AUDIO_TO_VIDEO_SEED_AUDIO_FILENAME,
      ),
      startFrame: path.join(
        "/repo",
        "resources",
        EXPLORE_AUDIO_TO_VIDEO_SEED_DIR,
        EXPLORE_AUDIO_TO_VIDEO_SEED_IMAGE_FILENAME,
      ),
    });
  });

  it("resolves only the allowlisted packaged resources files", () => {
    const resourcesPath = "/packaged/Resources";
    const resolved = resolveExploreAudioToVideoSeedPaths({
      isPackaged: true,
      projectRoot: "/repo",
      resourcesPath,
    });
    assert.deepEqual(resolved, {
      audio: path.join(
        resourcesPath,
        EXPLORE_AUDIO_TO_VIDEO_SEED_DIR,
        EXPLORE_AUDIO_TO_VIDEO_SEED_AUDIO_FILENAME,
      ),
      startFrame: path.join(
        resourcesPath,
        EXPLORE_AUDIO_TO_VIDEO_SEED_DIR,
        EXPLORE_AUDIO_TO_VIDEO_SEED_IMAGE_FILENAME,
      ),
    });
    assert.equal(resolved.audio.startsWith(resourcesPath + path.sep), true);
    assert.equal(resolved.startFrame.startsWith(resourcesPath + path.sep), true);
    assert.match(resolved.audio, /audio-to-video-input-v2\.mp3$/);
    assert.match(resolved.startFrame, /audio-to-video-input\.jpg$/);
  });

  it("keeps both packaged example files in the repo resources directory", () => {
    const seedDir = path.join(
      repoRoot,
      "resources",
      EXPLORE_AUDIO_TO_VIDEO_SEED_DIR,
    );
    const audioPath = path.join(
      seedDir,
      EXPLORE_AUDIO_TO_VIDEO_SEED_AUDIO_FILENAME,
    );
    const imagePath = path.join(
      seedDir,
      EXPLORE_AUDIO_TO_VIDEO_SEED_IMAGE_FILENAME,
    );

    assert.equal(fs.existsSync(audioPath), true);
    assert.equal(fs.existsSync(imagePath), true);
    assert.deepEqual(
      [...fs.readFileSync(imagePath).subarray(0, 3)],
      [0xff, 0xd8, 0xff],
    );
    // MP3 files start with an ID3 tag or a raw frame sync.
    const audioHeader = fs.readFileSync(audioPath).subarray(0, 3);
    assert.equal(
      audioHeader.toString("latin1") === "ID3" || audioHeader[0] === 0xff,
      true,
    );
  });
});
