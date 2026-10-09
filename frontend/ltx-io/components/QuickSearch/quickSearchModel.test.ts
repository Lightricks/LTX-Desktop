import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { IC_LORA_RECIPES } from "../../../lib/ic-lora-recipes.ts";

import {
  buildQuickSearchBrowseSections,
  searchQuickSearchFeatures,
  searchQuickSearchRows,
  tokenizeQuickSearchQuery,
} from "./quickSearchModel.ts";

describe("quickSearchModel", () => {
  it("tokenizes queries on word boundaries", () => {
    assert.deepEqual(tokenizeQuickSearchQuery("Text-to Video"), ["text", "to", "video"]);
  });

  it("finds features by title and aliases", () => {
    const hits = searchQuickSearchFeatures("image photo");
    assert.ok(hits.some((feature) => feature.id === "image-to-video"));
  });

  it("returns browse rows for empty search in the UI layer", () => {
    assert.deepEqual(searchQuickSearchRows(""), []);
  });

  it("orders featured before recent in browse sections", () => {
    const sections = buildQuickSearchBrowseSections(["text-to-video"]);
    assert.equal(sections[0]?.id, "featured");
    assert.equal(sections[1]?.id, "recent");
  });

  it("includes workflows and LoRA gallery sections in browse", () => {
    const sections = buildQuickSearchBrowseSections([]);
    const ids = sections.map((section) => section.id);
    assert.ok(ids.includes("workflows"));
    assert.ok(ids.includes("styles"));
    assert.ok(ids.includes("camera-motion"));
    const workflows = sections.find((section) => section.id === "workflows");
    assert.deepEqual(
      workflows?.features.map((feature) => feature.id),
      ["text-to-video", "image-to-video", "audio-to-video"],
    );
    const postProduction = sections.find((section) => section.id === "post-production");
    assert.deepEqual(
      postProduction?.features.map((feature) => feature.id),
      ["retake", "extend", ...IC_LORA_RECIPES.filter((recipe) => recipe.section === "post-production").map((recipe) => recipe.id)],
    );
    const vfx = sections.find((section) => section.id === "vfx");
    assert.deepEqual(vfx?.features.map((feature) => feature.id), ["layout-to-render", "restore"]);
    const styles = sections.find((section) => section.id === "styles");
    assert.ok(styles?.features.some((feature) => feature.id === "cozy-felt"));
  });

  it("keeps a recent post-production tool in Post production and out of Featured", () => {
    const sections = buildQuickSearchBrowseSections(["day-to-night", "extend"]);
    const idsOf = (id: string) =>
      sections.find((section) => section.id === id)?.features.map((f) => f.id);
    assert.deepEqual(idsOf("recent"), ["day-to-night", "extend"]);
    assert.deepEqual(idsOf("featured"), ["image-to-video", "audio-to-video"]);
    assert.deepEqual(idsOf("post-production"), ["retake", "extend", ...IC_LORA_RECIPES.filter((recipe) => recipe.section === "post-production").map((recipe) => recipe.id)]);
    assert.deepEqual(idsOf("vfx"), ["layout-to-render", "restore"]);
  });

  it("finds LoRA recipes by lora alias and title", () => {
    const loraHits = searchQuickSearchFeatures("lora");
    assert.ok(loraHits.some((feature) => feature.id === "cozy-felt"));
    const feltHits = searchQuickSearchFeatures("felt");
    assert.ok(feltHits.some((feature) => feature.id === "cozy-felt"));
  });
});
