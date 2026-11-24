import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { featureDetailsHeroPlayerProps } from "./featureDetailsHeroPlayer.ts";

describe("featureDetailsHeroPlayerProps", () => {
  const hero = { videoUrl: "clip.webm", posterUrl: "poster.jpg" };

  it("keeps the poster, cover crop, and title, and autoplays", () => {
    assert.deepEqual(featureDetailsHeroPlayerProps(hero, "Retake", false), {
      src: "clip.webm",
      posterUrl: "poster.jpg",
      objectFit: "cover",
      label: "Retake",
      autoPlay: true,
    });
  });

  it("does not autoplay when reduced motion is on", () => {
    assert.equal(
      featureDetailsHeroPlayerProps(hero, "Retake", true).autoPlay,
      false,
    );
  });
});
