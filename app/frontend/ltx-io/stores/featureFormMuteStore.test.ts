import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";

import { useFeatureFormMuteStore } from "./featureFormMuteStore.ts";

describe("useFeatureFormMuteStore", () => {
  afterEach(() => {
    useFeatureFormMuteStore.setState({ isMuted: true });
  });

  it("starts muted and keeps the last toggle", () => {
    assert.equal(useFeatureFormMuteStore.getState().isMuted, true);

    useFeatureFormMuteStore.getState().setMuted(false);
    assert.equal(useFeatureFormMuteStore.getState().isMuted, false);

    useFeatureFormMuteStore.getState().setMuted(true);
    assert.equal(useFeatureFormMuteStore.getState().isMuted, true);
  });
});
