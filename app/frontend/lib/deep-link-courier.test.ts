import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { DeepLinkTarget } from "../../shared/deep-link.ts";

import { type DeepLinkCourierIntent, startDeepLinkCourier } from "./deep-link-courier.ts";

function intent(id: string, target: DeepLinkTarget): DeepLinkCourierIntent {
  return { id, target };
}

function inbox(initial: DeepLinkCourierIntent | null = null) {
  let pending = initial;
  return {
    take: async () => {
      const next = pending;
      pending = null;
      return next;
    },
    offer: (next: DeepLinkCourierIntent) => {
      pending = next;
    },
  };
}

describe("startDeepLinkCourier", () => {
  it("subscribes first, then takes the pending intent before navigating", async () => {
    const navigated: string[] = [];
    const pending = inbox(intent("1", { kind: "fetcher-tool", slug: "t2v" }));
    const stop = startDeepLinkCourier({
      subscribe: () => () => {},
      take: pending.take,
      navigate: (path) => {
        navigated.push(path);
      },
      pathForTarget: (target) => `/fetcher/${target.slug}`,
    });

    await Promise.resolve();
    assert.deepEqual(navigated, ["/fetcher/t2v"]);
    assert.equal(await pending.take(), null);
    stop();
  });

  it("does not navigate on a second mount after take cleared the inbox", async () => {
    const navigated: string[] = [];
    const pending = inbox(intent("1", { kind: "fetcher-tool", slug: "t2v" }));
    const deps = {
      subscribe: () => () => {},
      take: pending.take,
      navigate: (path: string) => {
        navigated.push(path);
      },
      pathForTarget: (target: DeepLinkTarget) => `/fetcher/${target.slug}`,
    };

    const stopFirst = startDeepLinkCourier(deps);
    await Promise.resolve();
    stopFirst();

    const stopSecond = startDeepLinkCourier(deps);
    await Promise.resolve();
    assert.deepEqual(navigated, ["/fetcher/t2v"]);
    stopSecond();
  });

  it("takes from the inbox on a live event, ignoring the event payload", async () => {
    const navigated: string[] = [];
    const pending = inbox();
    let listener: ((next: DeepLinkCourierIntent) => void) | null = null;

    const stop = startDeepLinkCourier({
      subscribe: (cb) => {
        listener = cb;
        return () => {
          listener = null;
        };
      },
      take: pending.take,
      navigate: (path) => {
        navigated.push(path);
      },
      pathForTarget: (target) =>
        target.kind === "home-shell" ? `/${target.slug}` : `/fetcher/${target.slug}`,
    });

    await Promise.resolve();
    assert.deepEqual(navigated, []);
    pending.offer(intent("2", { kind: "home-shell", slug: "assets" }));
    listener?.(intent("1", { kind: "fetcher-tool", slug: "t2v" }));
    await Promise.resolve();
    assert.deepEqual(navigated, ["/assets"]);
    stop();
  });
});
