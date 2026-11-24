import assert from "node:assert/strict";
import { it } from "node:test";

import type { components } from "../../generated/backend-openapi.ts";
import {
  addAssistantAttachments,
  assistantChatBusy,
  assistantLiveGenerationIds,
  assistantTurnState,
} from "./assistant-session.ts";

type Turn = components["schemas"]["AssistantConversationTurn"];

const request = {
  request_id: "2b1c0d4e-1f2a-4b3c-8d4e-5f6a7b8c9d0e",
  conversation_id: "7c1d2e3f-4a5b-4c6d-8e7f-9a0b1c2d3e4f",
  text: "Generate",
  asset_ids: ["image"],
};

function chat(turn: Partial<Turn>) {
  return {
    conversation_id: request.conversation_id,
    turns: [{ request, response: null, provider: "gemini" as const, model: "", ...turn }],
  };
}

it("shows a turn another device is still running as working, not as a failed retry", () => {
  assert.equal(assistantTurnState(chat({ running: true })).kind, "working");
  const continued = chat({
    running: true,
    response: { run_id: request.request_id, status: "paused", text: "", generation_ids: [], asset_ids: [] },
  });
  assert.equal(assistantTurnState(continued).kind, "working");
});

it("offers the original request for retry once a turn stops running without a response", () => {
  assert.deepEqual(assistantTurnState(chat({ running: false })), {
    kind: "failed",
    runId: request.request_id,
    request,
  });
});

it("a retry still pending in another chat does not make this chat busy", () => {
  const newChat = { conversation_id: "new", turns: [] };
  assert.equal(
    assistantChatBusy(newChat, "new", {
      retryConversationId: request.conversation_id,
      stopRunId: request.request_id,
    }),
    false,
  );
  assert.equal(
    assistantChatBusy(chat({}), request.conversation_id, { retryConversationId: request.conversation_id }),
    true,
  );
});

it("jobs in a turn that is still waiting keep refreshing even after they failed", () => {
  const response = { run_id: request.request_id, text: "", generation_ids: ["retried"], asset_ids: [] };
  assert.deepEqual([...assistantLiveGenerationIds(chat({ response: { ...response, status: "waiting" } }))], [
    "retried",
  ]);
  assert.equal(assistantLiveGenerationIds(chat({ response: { ...response, status: "complete" } })).size, 0);
});

it("caps concurrent attachment additions without duplicating existing assets", () => {
  const seven = Array.from({ length: 7 }, (_, index) => String(index));
  const fromPicker = addAssistantAttachments(seven, ["picked"]);
  const fromImport = addAssistantAttachments(fromPicker, ["imported", "picked"]);
  assert.equal(fromImport.length, 8);
  assert.deepEqual(fromImport, [...seven, "picked"]);
});
