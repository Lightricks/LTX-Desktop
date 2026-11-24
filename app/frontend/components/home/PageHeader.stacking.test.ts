import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";

function declaration(source: string, selector: string, property: string): string {
  const block = source.match(new RegExp(`${selector}\\s*\\{([^}]*)\\}`));
  assert.ok(block, `missing ${selector}`);
  const value = block[1].match(new RegExp(`${property}:\\s*([^;]+);`));
  assert.ok(value, `missing ${property} in ${selector}`);
  return value[1].trim();
}

function tokenValue(source: string, name: string): number {
  const match = source.match(new RegExp(`${name}:\\s*(-?\\d+)`));
  assert.ok(match, `missing ${name}`);
  return Number(match[1]);
}

describe("page header stacking", () => {
  it("paints the inset header above the generation metadata row and below modals", () => {
    const header = readFileSync(
      new URL("./PageHeader.module.scss", import.meta.url),
      "utf8",
    );
    const meta = readFileSync(
      new URL(
        "../../ltx-io/screens/Feature/results/ResultMetaBar.module.scss",
        import.meta.url,
      ),
      "utf8",
    );
    const tokens = readFileSync(
      new URL("../../ds/styles/themes/reference.module.scss", import.meta.url),
      "utf8",
    );

    const headerZ = declaration(header, String.raw`\.headerInsetFromShell`, "z-index");
    assert.equal(headerZ, "var(--z-dropdown)");

    const metaZ = Number(declaration(meta, String.raw`\.container`, "z-index"));
    const dropdown = tokenValue(tokens, "--z-dropdown");
    const sticky = tokenValue(tokens, "--z-sticky");
    const modal = tokenValue(tokens, "--z-modal");

    assert.ok(dropdown > metaZ);
    assert.ok(dropdown > sticky);
    assert.ok(dropdown < modal);
  });
});
