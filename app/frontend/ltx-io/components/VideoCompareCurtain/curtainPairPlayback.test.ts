import assert from "node:assert/strict";
import { describe, it, mock } from "node:test";

import {
  bindCurtainPairPlayback,
  restartCurtainPair,
  isAutoplayBlockedError,
  seekMedia,
  syncCurtainPairFromAfter,
  syncCurtainPlayheads,
  transferMediaPlayhead,
} from "./curtainPairPlayback.ts";

type PlayMock = ReturnType<typeof mock.fn<() => Promise<void>>>;

async function waitUntil(condition: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50 && !condition(); attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  assert.equal(condition(), true);
}

type MockMedia = {
  duration: number;
  readyState: number;
  seeking: boolean;
  paused: boolean;
  loop: boolean;
  muted: boolean;
  currentTime: number;
  play: PlayMock;
  pause: ReturnType<typeof mock.fn>;
  addEventListener: (type: string, listener: () => void) => void;
  removeEventListener: (type: string, listener: () => void) => void;
  emit: (type: string) => void;
};

function asMedia(media: MockMedia): HTMLMediaElement {
  return media as unknown as HTMLMediaElement;
}

function createMedia(init?: {
  currentTime?: number;
  duration?: number;
  readyState?: number;
  seeking?: boolean;
  muted?: boolean;
  throwOnSeek?: boolean;
  play?: () => Promise<void>;
}): MockMedia {
  let currentTime = init?.currentTime ?? 0;
  const listeners = new Map<string, Set<() => void>>();
  return {
    duration: init?.duration ?? 10,
    readyState: init?.readyState ?? 2,
    seeking: init?.seeking ?? false,
    paused: true,
    loop: true,
    muted: init?.muted ?? false,
    get currentTime() {
      return currentTime;
    },
    set currentTime(value: number) {
      if (init?.throwOnSeek) {
        throw new Error("InvalidStateError");
      }
      currentTime = value;
    },
    play: mock.fn(init?.play ?? (() => Promise.resolve())),
    pause: mock.fn(),
    addEventListener(type: string, listener: () => void) {
      const group = listeners.get(type) ?? new Set();
      group.add(listener);
      listeners.set(type, group);
    },
    removeEventListener(type: string, listener: () => void) {
      listeners.get(type)?.delete(listener);
    },
    emit(type: string) {
      listeners.get(type)?.forEach((listener) => listener());
    },
  };
}

describe("seekMedia", () => {
  it("ignores seek errors when the element is not ready", () => {
    const media = createMedia({ throwOnSeek: true });
    assert.doesNotThrow(() => seekMedia(asMedia(media), 1.5));
  });
});

describe("transferMediaPlayhead", () => {
  it("clamps the source playhead to the destination duration", () => {
    const from = createMedia({ currentTime: 8, duration: 12 });
    const to = createMedia({ currentTime: 0, duration: 5 });
    transferMediaPlayhead(asMedia(from), asMedia(to));
    assert.equal(to.currentTime, 5);
  });
});

describe("syncCurtainPlayheads", () => {
  it("does not seek while either clip is still loading", () => {
    const before = createMedia({ currentTime: 0, readyState: 1 });
    const after = createMedia({ currentTime: 1.2, readyState: 2 });
    const restartPair = mock.fn();

    syncCurtainPlayheads(asMedia(before), asMedia(after), restartPair);

    assert.equal(before.currentTime, 0);
    assert.equal(restartPair.mock.callCount(), 0);
  });

  it("restarts both clips when playback reaches the shorter duration", () => {
    const before = createMedia({ currentTime: 4.9, duration: 5 });
    const after = createMedia({ currentTime: 4.96, duration: 5 });
    const restartPair = mock.fn();

    syncCurtainPlayheads(asMedia(before), asMedia(after), restartPair);

    assert.equal(restartPair.mock.callCount(), 1);
  });
});

describe("restartCurtainPair", () => {
  it("seeks both clips to the start and plays them", () => {
    const before = createMedia({ currentTime: 3 });
    const after = createMedia({ currentTime: 3 });

    restartCurtainPair(asMedia(before), asMedia(after));

    assert.equal(before.currentTime, 0);
    assert.equal(after.currentTime, 0);
    assert.equal(before.play.mock.callCount(), 1);
    assert.equal(after.play.mock.callCount(), 1);
  });

  it("mutes both clips and retries when unmuted autoplay is blocked", async () => {
    const before = createMedia({ muted: true });
    const after = createMedia({
      muted: false,
      play: () =>
        Promise.reject(
          Object.assign(new Error("play() failed"), { name: "NotAllowedError" }),
        ),
    });
    const onAutoplayBlocked = mock.fn();

    restartCurtainPair(asMedia(before), asMedia(after), onAutoplayBlocked);
    await waitUntil(() => onAutoplayBlocked.mock.callCount() === 1);

    assert.equal(before.muted, true);
    assert.equal(after.muted, true);
    assert.equal(before.play.mock.callCount(), 2);
    assert.equal(after.play.mock.callCount(), 2);
  });

  it("does not mute or retry when play() is aborted", async () => {
    const before = createMedia();
    const after = createMedia({
      play: () => Promise.reject(new DOMException("interrupted", "AbortError")),
    });
    const onAutoplayBlocked = mock.fn();

    restartCurtainPair(asMedia(before), asMedia(after), onAutoplayBlocked);
    await Promise.resolve();
    await Promise.resolve();

    assert.equal(onAutoplayBlocked.mock.callCount(), 0);
    assert.equal(before.muted, false);
    assert.equal(after.muted, false);
    assert.equal(before.play.mock.callCount(), 1);
    assert.equal(after.play.mock.callCount(), 1);
  });
});

describe("bindCurtainPairPlayback", () => {
  it("disables native loop and starts both clips once they have data", () => {
    const before = createMedia({ readyState: 1 });
    const after = createMedia({ readyState: 1 });

    bindCurtainPairPlayback(asMedia(before), asMedia(after));

    assert.equal(before.loop, false);
    assert.equal(after.loop, false);
    assert.equal(before.play.mock.callCount(), 0);

    before.readyState = 2;
    before.emit("loadeddata");
    assert.equal(before.play.mock.callCount(), 0);

    after.readyState = 2;
    after.emit("loadeddata");
    assert.equal(before.play.mock.callCount(), 1);
    assert.equal(after.play.mock.callCount(), 1);
  });

  it("restarts the pair when either clip ends", () => {
    const before = createMedia({ currentTime: 4 });
    const after = createMedia({ currentTime: 4 });

    bindCurtainPairPlayback(asMedia(before), asMedia(after));
    before.play.mock.resetCalls();
    after.play.mock.resetCalls();

    before.emit("ended");

    assert.equal(before.currentTime, 0);
    assert.equal(after.currentTime, 0);
    assert.equal(before.play.mock.callCount(), 1);
    assert.equal(after.play.mock.callCount(), 1);
  });

  it("pauses both clips and ignores later events after teardown", () => {
    const before = createMedia();
    const after = createMedia();

    const unbind = bindCurtainPairPlayback(asMedia(before), asMedia(after));
    unbind();
    before.play.mock.resetCalls();
    after.play.mock.resetCalls();

    assert.equal(before.pause.mock.callCount(), 1);
    assert.equal(after.pause.mock.callCount(), 1);

    before.emit("ended");
    assert.equal(before.play.mock.callCount(), 0);
  });

  it("does not treat teardown AbortError as an autoplay block", async () => {
    let rejectBefore: (reason: unknown) => void = () => {};
    const before = createMedia({
      play: () =>
        new Promise((_, reject) => {
          rejectBefore = reject;
        }),
    });
    const after = createMedia();
    const onAutoplayBlocked = mock.fn();

    const unbind = bindCurtainPairPlayback(
      asMedia(before),
      asMedia(after),
      onAutoplayBlocked,
    );
    unbind();
    rejectBefore(new DOMException("The play() request was interrupted", "AbortError"));
    await Promise.resolve();
    await Promise.resolve();

    assert.equal(onAutoplayBlocked.mock.callCount(), 0);
    assert.equal(before.muted, false);
    assert.equal(after.muted, false);
    assert.equal(before.play.mock.callCount(), 1);
  });
});

describe("syncCurtainPairFromAfter", () => {
  it("no-ops when either element is missing", () => {
    const after = createMedia({ currentTime: 1.2 });
    assert.doesNotThrow(() => syncCurtainPairFromAfter(null, asMedia(after)));
  });
});

describe("isAutoplayBlockedError", () => {
  it("is true for a NotAllowedError", () => {
    assert.equal(isAutoplayBlockedError({ name: "NotAllowedError" }), true);
  });

  it("is false for an AbortError, other errors, and non-objects", () => {
    assert.equal(isAutoplayBlockedError({ name: "AbortError" }), false);
    assert.equal(isAutoplayBlockedError(new Error("x")), false);
    assert.equal(isAutoplayBlockedError(null), false);
    assert.equal(isAutoplayBlockedError("NotAllowedError"), false);
  });
});
