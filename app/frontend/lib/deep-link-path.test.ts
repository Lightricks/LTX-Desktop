import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { paths } from "../paths.ts";

import { pathForDeepLinkTarget } from "./deep-link-path.ts";

describe("pathForDeepLinkTarget", () => {
  it("sends generation tools through the Fetcher landing route", () => {
    assert.equal(
      pathForDeepLinkTarget({ kind: "fetcher-tool", slug: "t2v" }),
      "/fetcher/t2v",
    );
    assert.equal(
      pathForDeepLinkTarget({ kind: "fetcher-tool", slug: "cozy-felt" }),
      "/fetcher/cozy-felt",
    );
  });

  it("opens Assets directly, skipping the offering gate", () => {
    assert.equal(
      pathForDeepLinkTarget({ kind: "home-shell", slug: "assets" }),
      paths.assets,
    );
  });
});
