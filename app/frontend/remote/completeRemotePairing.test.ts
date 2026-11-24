import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";
import { setBearerToken } from "./api.ts";
import { completeRemotePairing } from "./completeRemotePairing.ts";

function installPairingWindow(href: string) {
  const current = new URL(href);
  const store = new Map<string, string>();
  const history = {
    state: null as unknown,
    replaceState(state: unknown, _title: string, url: string) {
      history.state = state;
      current.href = new URL(url, current.origin).href;
    },
  };
  const location = {
    get href() {
      return current.href;
    },
    get pathname() {
      return current.pathname;
    },
    get search() {
      return current.search;
    },
    get hash() {
      return current.hash;
    },
  };
  const localStorage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value);
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
  };
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: localStorage,
  });
  Object.defineProperty(globalThis, "history", {
    configurable: true,
    value: history,
  });
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: { location, history, localStorage },
  });
}

afterEach(() => {
  setBearerToken(null);
});

describe("completeRemotePairing", () => {
  it("replaces to / after a successful pair", async () => {
    installPairingWindow("http://192.168.1.9:41955/pairing?t=grant");
    const navigated: Array<{ to: string; replace: boolean }> = [];
    const previousFetch = globalThis.fetch;
    globalThis.fetch = (async () =>
      new Response(
        JSON.stringify({
          token: "session-token",
          device_id: "dev-1",
          name: "Phone",
        }),
        { status: 200 },
      )) as typeof fetch;
    try {
      assert.equal(
        await completeRemotePairing("grant", (to, options) => {
          navigated.push({ to, replace: options.replace });
        }),
        "ok",
      );
      assert.deepEqual(navigated, [{ to: "/", replace: true }]);
    } finally {
      globalThis.fetch = previousFetch;
    }
  });

  it("does not navigate when pairing fails", async () => {
    installPairingWindow("http://192.168.1.9:41955/pairing?t=grant");
    const navigated: string[] = [];
    const previousFetch = globalThis.fetch;
    globalThis.fetch = (async () =>
      new Response("", { status: 401 })) as typeof fetch;
    try {
      assert.equal(
        await completeRemotePairing("grant", (to) => {
          navigated.push(to);
        }),
        "failed",
      );
      assert.deepEqual(navigated, []);
    } finally {
      globalThis.fetch = previousFetch;
    }
  });
});
