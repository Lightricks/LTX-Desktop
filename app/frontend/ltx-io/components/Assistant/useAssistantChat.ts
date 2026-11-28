import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

import {
  assistantConversationQueryKey,
  assistantTurnState,
} from "@/components/home/assistant-session";
import { invalidateAffectedGenerationQueries } from "@/ltx-io/hooks/generationQueryKeys";
import { unwrapApiResult } from "@/ltx-io/lib/unwrapApiResult";
import type { ExploreApi } from "@/ltx-io/runtime/ExploreRuntime";
import { useExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";

import { ASSISTANT_CONVERSATIONS_QUERY_KEY } from "./AssistantConversationMenu";

// Another device can add turns to the open chat; poll faster while one is unfinished.
const CHAT_POLL_MS = 5000;
const UNFINISHED_CHAT_POLL_MS = 2000;

type AssistantQueueItem = Parameters<ExploreApi["enqueueAssistantMessage"]>[1];
type AssistantTurnRequest = Parameters<ExploreApi["assistantTurn"]>[0];

/** The open chat, owned by the Desktop. Nothing here is stored in the browser. */
export function useAssistantChat(
  conversationId: string | null,
  live: boolean,
  onError: (error: Error) => void,
) {
  const { api } = useExploreRuntime();
  const queryClient = useQueryClient();

  const conversation = useQuery({
    queryKey: assistantConversationQueryKey(conversationId),
    enabled: !!conversationId,
    queryFn: async () => unwrapApiResult(await api.getAssistantConversation(conversationId!)),
    // Focus covers a phone waking; the interval pauses while the page is hidden.
    refetchOnWindowFocus: live,
    refetchInterval: (query) =>
      !live
        ? false
        : assistantTurnState(query.state.data).kind === "idle" && !query.state.data?.queue?.length
          ? CHAT_POLL_MS
          : UNFINISHED_CHAT_POLL_MS,
  });
  const { refetch } = conversation;
  useEffect(() => {
    if (live && conversationId) void refetch();
    // A new chat id fetches on its own; this covers the docked panel reopening.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live, refetch]);

  const settle = () => {
    invalidateAffectedGenerationQueries(queryClient);
    void queryClient.invalidateQueries({ queryKey: assistantConversationQueryKey(conversationId) });
    void queryClient.invalidateQueries({ queryKey: ASSISTANT_CONVERSATIONS_QUERY_KEY });
  };
  const show = (result: Awaited<ReturnType<ExploreApi["getAssistantConversation"]>>) => {
    const data = unwrapApiResult(result);
    queryClient.setQueryData(assistantConversationQueryKey(data.conversation_id), data);
  };
  const options = { onError, onSettled: settle };

  return {
    conversation: conversation.data,
    loading: !!conversationId && conversation.isPending,
    // A failed background poll keeps the last good chat on screen.
    loadError: conversation.data ? null : conversation.error,
    refresh: () => void refetch(),
    enqueue: useMutation({
      mutationFn: async (item: AssistantQueueItem) =>
        show(await api.enqueueAssistantMessage(conversationId!, item)),
      ...options,
    }),
    removeQueued: useMutation({
      mutationFn: async (itemId: string) =>
        show(await api.removeQueuedAssistantMessage(conversationId!, itemId)),
      ...options,
    }),
    releaseQueue: useMutation({
      mutationFn: async () => show(await api.releaseAssistantQueue(conversationId!)),
      ...options,
    }),
    stop: useMutation({
      mutationFn: async (runId: string) => unwrapApiResult(await api.stopAssistant(runId)),
      ...options,
    }),
    continueRun: useMutation({
      mutationFn: async (runId: string) => unwrapApiResult(await api.continueAssistant(runId)),
      ...options,
    }),
    retryFailed: useMutation({
      mutationFn: async (request: AssistantTurnRequest) =>
        unwrapApiResult(await api.assistantTurn(request)),
      ...options,
    }),
    deleteChat: useMutation({
      mutationFn: async (id: string) => unwrapApiResult(await api.deleteAssistantConversation(id)),
      onSuccess: (_, id) => queryClient.removeQueries({ queryKey: assistantConversationQueryKey(id) }),
      ...options,
    }),
  };
}
