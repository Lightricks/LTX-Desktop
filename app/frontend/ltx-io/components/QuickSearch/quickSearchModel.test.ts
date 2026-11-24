import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { IC_LORA_RECIPES } from "../../../lib/ic-lora-recipes.ts";
import { customLoraRunPath } from "../../../paths.ts";

import {
  browseSectionsToRows,
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
    const sections = buildQuickSearchBrowseSections([], {});
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
      [
        "retake",
        "extend",
        ...IC_LORA_RECIPES.filter((recipe) => recipe.section === "post-production").map(
          (recipe) => recipe.id,
        ),
      ],
    );
    const vfx = sections.find((section) => section.id === "vfx");
    assert.deepEqual(
      vfx?.features.map((feature) => feature.id),
      IC_LORA_RECIPES.filter((recipe) => recipe.section === "vfx").map(
        (recipe) => recipe.id,
      ),
    );
    const styles = sections.find((section) => section.id === "styles");
    assert.ok(styles?.features.some((feature) => feature.id === "cozy-felt"));
  });

  it("keeps a recent post-production tool in Post production and out of Featured", () => {
    const sections = buildQuickSearchBrowseSections(["day-to-night", "extend"], {});
    const idsOf = (id: string) =>
      sections.find((section) => section.id === id)?.features.map((f) => f.id);
    assert.deepEqual(idsOf("recent"), ["day-to-night", "extend"]);
    assert.deepEqual(idsOf("featured"), [
      "image-to-video",
      "audio-to-video",
      "alpha-gen",
      "cinemagraph",
    ]);
    assert.deepEqual(idsOf("post-production"), [
      "retake",
      "extend",
      ...IC_LORA_RECIPES.filter((recipe) => recipe.section === "post-production").map(
        (recipe) => recipe.id,
      ),
    ]);
    assert.deepEqual(
      idsOf("vfx"),
      IC_LORA_RECIPES.filter((recipe) => recipe.section === "vfx").map(
        (recipe) => recipe.id,
      ),
    );
  });

  it("never lists the Add custom LoRA page in search or browse, even as a recent tool", () => {
    const search = searchQuickSearchFeatures("custom lora add", undefined, {}).map(
      (feature) => feature.id,
    );
    const browse = buildQuickSearchBrowseSections(["custom-lora"], {}).flatMap(
      (section) => section.features.map((feature) => feature.id),
    );

    assert.ok(!search.includes("custom-lora"));
    assert.ok(!browse.includes("custom-lora"));
  });

  it("hides the My LoRAs section when the user has no saved LoRA", () => {
    const sections = buildQuickSearchBrowseSections([], {}, []);

    assert.ok(!sections.some((section) => section.id === "custom-lora"));
  });

  it("lists saved LoRAs in My LoRAs and finds them by name", () => {
    const lora = {
      id: "ink",
      name: "Ink Wash",
      source: "file",
      page_url: null,
      description: "An ink wash style.",
      trigger: null,
      cover_url: null,
    } as const;

    const browse = browseSectionsToRows(
      buildQuickSearchBrowseSections([], {}, [lora]),
    );
    const search = searchQuickSearchRows("ink", {}, [lora]);

    const listed = browse.filter((row) => row.kind === "custom-lora");
    assert.deepEqual(
      listed.map((row) => [row.sectionId, row.kind === "custom-lora" && row.path]),
      [["custom-lora", customLoraRunPath("ink")]],
    );
    assert.equal(search.filter((row) => row.kind === "custom-lora").length, 1);
  });

  it("finds LoRA recipes by lora alias and title", () => {
    const loraHits = searchQuickSearchFeatures("lora");
    assert.ok(loraHits.some((feature) => feature.id === "cozy-felt"));
    const feltHits = searchQuickSearchFeatures("felt");
    assert.ok(feltHits.some((feature) => feature.id === "cozy-felt"));
  });
});
