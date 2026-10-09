import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { HOME_FEATURES } from "../../../../lib/home-features.ts";
import { paths } from "../../../../paths.ts";

import {
  buildFeatureDetails,
  COMMUNITY_FEATURE_DISCLAIMER,
  featureCrumbTrail,
} from "./featureChromeModel.ts";

describe("feature chrome model", () => {
  it("builds a Home / title trail from HOME_FEATURES", () => {
    const feature = HOME_FEATURES[0];
    assert.deepEqual(featureCrumbTrail(feature), [
      { id: "home", label: "Home", to: paths.home },
      { id: feature.id, label: feature.title, to: null },
    ]);
  });

  it("requires listing copy for every Home feature", () => {
    for (const feature of HOME_FEATURES) {
      const details = buildFeatureDetails(feature.id);
      assert.ok(details.blurb.length, feature.id);
      const type = details.rows.find((row) => row.id === "type");
      assert.ok(type?.value.length, feature.id);
    }
  });

  it("matches LTX.io Text to Video details", () => {
    const details = buildFeatureDetails("text-to-video");
    assert.equal(details.credit.affiliation, "ltx");
    assert.equal(details.credit.name, "LTX Team");
    assert.equal(details.rows[0]?.value, "Workflow");
    assert.equal(details.rows[1]?.value, "Text to Video");
    assert.equal(details.rows.length, 2);
    assert.equal(details.externalLink, null);
    assert.match(details.blurb, /synchronized audio/);
  });

  it("matches LTX.io Image to Video and Audio to Video categories", () => {
    assert.equal(buildFeatureDetails("image-to-video").rows[0]?.value, "Motion");
    assert.match(
      buildFeatureDetails("image-to-video").blurb,
      /still image to life/,
    );
    assert.equal(
      buildFeatureDetails("audio-to-video").rows[0]?.value,
      "Audio, Animation",
    );
    assert.match(
      buildFeatureDetails("audio-to-video").blurb,
      /audio track into video/,
    );
  });

  it("matches LTX.io Retake, Extend, and Dolly In type labels", () => {
    assert.equal(
      buildFeatureDetails("retake").rows[1]?.value,
      "Retake video",
    );
    assert.equal(
      buildFeatureDetails("extend").rows[1]?.value,
      "Extend Video",
    );
    assert.equal(
      buildFeatureDetails("dolly-in").rows[1]?.value,
      "Image to Video",
    );
    assert.equal(buildFeatureDetails("dolly-in").credit.affiliation, "ltx");
  });

  it("matches LTX.io Cozy Felt community details", () => {
    const details = buildFeatureDetails("cozy-felt");
    assert.equal(details.credit.affiliation, "community");
    assert.equal(details.credit.name, "vrgamedevgirl84");
    assert.equal(details.rows[0]?.value, "Community");
    assert.equal(details.rows[1]?.value, "Text to Video");
    assert.equal(details.rows[2]?.value, "ltx-2.3-fast");
    assert.equal(details.rows[3]?.value, "LTX");
    assert.equal(
      details.rows[3]?.href,
      "https://github.com/Lightricks/LTX-2/blob/main/LICENSE",
    );
    assert.deepEqual(details.externalLink, {
      label: "Hugging Face",
      url: "https://huggingface.co/vrgamedevgirl84/LTX2.3_Cozy_Felt_Style_LoRa",
    });
    assert.match(details.blurb, /handcrafted felt world/);
    assert.match(COMMUNITY_FEATURE_DISCLAIMER, /community member/);
  });
});
