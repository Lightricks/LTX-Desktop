import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import { electronAPISchemas } from "../shared/electron-api-schema.ts";
import {
  EXPLORE_LORA_RECIPE_SEED_DIR,
  EXPLORE_LORA_RECIPE_SEED_FILES,
  EXPLORE_LORA_RECIPE_SEED_IDS,
  resolveExploreLoraRecipeSeedPath,
} from "./explore-dolly-in-seed.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

describe("LoRA recipe seed IPC", () => {
  it("allowlists seed ids and strips caller-supplied paths", () => {
    const schema = electronAPISchemas.getExploreLoraRecipeSeedPath;
    assert.equal(
      schema.input.parse({ seedId: "transition-end-frame" }).seedId,
      "transition-end-frame",
    );
    assert.throws(() => schema.input.parse({ seedId: "../passwd" }));
    const parsed = schema.input.parse({
      seedId: "vbvr-start-frame",
      resourcePath: "/etc/passwd",
    });
    assert.equal(parsed.seedId, "vbvr-start-frame");
    assert.equal("resourcePath" in parsed, false);
  });

  it("does not accept a free-form resource path", () => {
    const schema = electronAPISchemas.getExploreLoraRecipeSeedPath;
    const parsed = schema.input.parse({
      seedId: "dolly-in-start-frame",
      resourcePath: "/etc/passwd",
      relativePath: "secret.jpg",
    });
    assert.equal(parsed.seedId, "dolly-in-start-frame");
    assert.equal("resourcePath" in parsed, false);
    assert.equal("relativePath" in parsed, false);
  });

  it("resolves every packaged LoRA recipe seed jpeg", () => {
    for (const seedId of EXPLORE_LORA_RECIPE_SEED_IDS) {
      const resolved = resolveExploreLoraRecipeSeedPath({
        seedId,
        isPackaged: false,
        projectRoot: repoRoot,
        resourcesPath: "/packaged/Resources",
      });
      assert.equal(
        resolved,
        path.join(
          repoRoot,
          "resources",
          EXPLORE_LORA_RECIPE_SEED_DIR,
          EXPLORE_LORA_RECIPE_SEED_FILES[seedId],
        ),
      );
      assert.equal(fs.existsSync(resolved), true, resolved);
      const header = fs.readFileSync(resolved).subarray(0, 3);
      assert.deepEqual([...header], [0xff, 0xd8, 0xff]);
    }
  });

  it("resolves only the allowlisted packaged resources file", () => {
    const resourcesPath = "/packaged/Resources";
    const resolved = resolveExploreLoraRecipeSeedPath({
      seedId: "dolly-in-start-frame",
      isPackaged: true,
      projectRoot: "/repo",
      resourcesPath,
    });
    assert.equal(
      resolved,
      path.join(
        resourcesPath,
        EXPLORE_LORA_RECIPE_SEED_DIR,
        EXPLORE_LORA_RECIPE_SEED_FILES["dolly-in-start-frame"],
      ),
    );
    assert.equal(resolved.startsWith(resourcesPath + path.sep), true);
    assert.match(resolved, /dolly-in-start-frame\.jpg$/);
  });
});
