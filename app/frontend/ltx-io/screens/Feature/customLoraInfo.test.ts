import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  applyCardSuggestions,
  customLoraInfoRequest,
  EMPTY_CUSTOM_LORA_INFO,
  importErrorField,
} from "./customLoraInfo.ts";

describe("applyCardSuggestions", () => {
  it("keeps the text the user already typed and fills only the empty fields", () => {
    const typed = { trigger: "", description: "My own words", examples: [""] };

    const next = applyCardSuggestions(typed, {
      trigger: "InkW4sh",
      description: "From the card",
      examples: ["a fox, InkW4sh"],
    });

    assert.equal(next.description, "My own words");
    assert.equal(next.trigger, "InkW4sh");
    assert.deepEqual(next.examples, ["a fox, InkW4sh"]);
  });

  it("keeps the examples the user typed", () => {
    const typed = { ...EMPTY_CUSTOM_LORA_INFO, examples: ["mine", ""] };

    const next = applyCardSuggestions(typed, { examples: ["from the card"] });

    assert.deepEqual(next.examples, ["mine", ""]);
  });
});

describe("customLoraInfoRequest", () => {
  it("drops empty examples and leaves out an empty trigger or description", () => {
    const request = customLoraInfoRequest({
      trigger: "  ",
      description: " Ink wash look ",
      examples: ["a fox", "  ", ""],
    });

    assert.deepEqual(request, { description: "Ink wash look", examples: ["a fox"] });
    assert.equal("trigger" in request, false);

    const empty = customLoraInfoRequest({ trigger: "", description: "  ", examples: [""] });

    assert.deepEqual(empty, { examples: [] });
  });
});

describe("importErrorField", () => {
  it("routes field codes to their field, other request codes to form, and poll errors to the submitted source", () => {
    assert.equal(importErrorField("CUSTOM_LORA_NAME_INVALID", "file"), "name");
    assert.equal(importErrorField("CUSTOM_LORA_NAME_TAKEN", "link"), "name");
    assert.equal(importErrorField("CUSTOM_LORA_FILE_INVALID", "file"), "file");
    assert.equal(importErrorField("CUSTOM_LORA_URL_INVALID", "link"), "link");
    assert.equal(importErrorField("CUSTOM_LORA_COVER_INVALID", "file"), "cover");
    assert.equal(importErrorField("HTTP_409", "file"), "form");
    assert.equal(importErrorField("CUSTOM_LORA_IMPORT_RUNNING", "link"), "form");
    assert.equal(importErrorField(undefined, "file"), "file");
    assert.equal(importErrorField(undefined, "link"), "link");
    assert.equal(importErrorField(undefined, null), "form");
  });
});
