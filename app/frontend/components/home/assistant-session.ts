import type { components } from "../../generated/backend-openapi.ts";

type AssistantConversation = components["schemas"]["AssistantConversation"];
type AssistantTurnRequest = components["schemas"]["AssistantTurnRequest"];

export const MAX_ASSISTANT_ATTACHMENTS = 8;

export function addAssistantAttachments(
  current: string[],
  added: string[],
  limit = MAX_ASSISTANT_ATTACHMENTS,
): string[] {
  return [...new Set([...current, ...added])].slice(0, limit);
}

export function assistantConversationQueryKey(conversationId: string | null) {
  return ["assistant", "conversation", conversationId] as const;
}

export type AssistantChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  assetIds: string[];
  generationIds: string[];
};

export function assistantChatMessages(
  conversation: AssistantConversation | null | undefined,
): AssistantChatMessage[] {
  return (conversation?.turns ?? []).flatMap(({ request, response }) => {
    const user: AssistantChatMessage = {
      id: `${request.request_id}:user`,
      role: "user",
      text: request.text,
      assetIds: request.asset_ids ?? [],
      generationIds: [],
    };
    if (!response) return [user];
    return [
      user,
      {
        id: `${response.run_id}:assistant`,
        role: "assistant",
        text: response.text,
        assetIds: response.asset_ids ?? [],
        generationIds: response.generation_ids ?? [],
      },
    ];
  });
}

/** What the chat's last turn needs from whoever is looking at it, on any device. */
export type AssistantTurnState =
  | { kind: "idle" }
  | { kind: "working"; runId: string }
  | { kind: "waiting"; runId: string }
  | { kind: "paused"; runId: string }
  | { kind: "failed"; runId: string; request: AssistantTurnRequest };

export function assistantTurnState(
  conversation: AssistantConversation | null | undefined,
): AssistantTurnState {
  const last = conversation?.turns.at(-1);
  if (!last) return { kind: "idle" };
  const runId = last.request.request_id;
  // A continued turn keeps its paused response until the step finishes.
  if (last.running) return { kind: "working", runId };
  if (!last.response) return { kind: "failed", runId, request: last.request };
  if (last.response.status === "waiting") return { kind: "waiting", runId };
  if (last.response.status === "paused") return { kind: "paused", runId };
  return { kind: "idle" };
}

/** Jobs in a turn that is still running. A retry can restart one whose cached status is final. */
export function assistantLiveGenerationIds(
  conversation: AssistantConversation | null | undefined,
): Set<string> {
  const { kind } = assistantTurnState(conversation);
  if (kind !== "working" && kind !== "waiting") return new Set();
  return new Set(conversation?.turns.at(-1)?.response?.generation_ids ?? []);
}

export type AssistantPendingActions = {
  stopRunId?: string;
  continueRunId?: string;
  retryConversationId?: string;
  deleteConversationId?: string;
};

/** Whether a pending action belongs to this chat; one started in another chat can still be in flight. */
export function assistantChatBusy(
  conversation: AssistantConversation | null | undefined,
  conversationId: string | null,
  pending: AssistantPendingActions,
): boolean {
  const ownsRun = (runId: string | undefined) =>
    !!runId && !!conversation?.turns.some((turn) => turn.request.request_id === runId);
  const isThisChat = (id: string | undefined) => !!id && id === conversationId;
  return (
    ownsRun(pending.stopRunId) ||
    ownsRun(pending.continueRunId) ||
    isThisChat(pending.retryConversationId) ||
    isThisChat(pending.deleteConversationId)
  );
}

/** The model the last turn ran with, so the footer names it while that turn is live. */
export function assistantTurnSelection(
  conversation: AssistantConversation | null | undefined,
): { provider: "gemini" | "local" | null; model: string | null } {
  const last = conversation?.turns.at(-1);
  return {
    provider: last?.provider ?? last?.request.provider ?? null,
    model: last?.model || last?.request.model || null,
  };
}
