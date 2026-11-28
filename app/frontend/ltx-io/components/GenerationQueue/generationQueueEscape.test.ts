import assert from "node:assert/strict";
import { test } from "node:test";

import { consumeQueueEscape } from "./generationQueueEscape.ts";

test("closing the queue consumes Escape so the assistant cannot also treat it as Stop", () => {
  const escape = new Event("keydown", { cancelable: true, bubbles: true });
  let open = true;

  consumeQueueEscape(escape, false, () => {
    open = false;
  });

  assert.equal(open, false);
  assert.equal(escape.defaultPrevented, true);
  assert.equal(escape.cancelBubble, true);
});

test("Escape during a queue drag is prevented for the assistant but still reaches the drag sensor", () => {
  const escape = new Event("keydown", { cancelable: true, bubbles: true });
  let open = true;

  consumeQueueEscape(escape, true, () => {
    open = false;
  });

  assert.equal(open, true);
  assert.equal(escape.defaultPrevented, true);
  assert.equal(escape.cancelBubble, false);
});
