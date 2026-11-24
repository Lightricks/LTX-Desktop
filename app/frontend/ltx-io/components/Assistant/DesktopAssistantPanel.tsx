import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import {
  queryOptions,
  useMutation,
  useQueries,
  useQueryClient,
} from "@tanstack/react-query";
import {
  ArrowDown,
  ArrowUp,
  ChevronDown,
  ListPlus,
  LoaderCircle,
  Maximize2,
  PanelRight,
  Play,
  RotateCcw,
  Square,
  X,
} from "lucide-react";
import { type ReactNode, useCallback, useEffect, useRef, useState } from "react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { z } from "zod";

import {
  MAX_ASSISTANT_ATTACHMENTS,
  addAssistantAttachments,
  assistantChatBusy,
  assistantChatMessages,
  assistantLiveGenerationIds,
  assistantTurnSelection,
  assistantTurnState,
} from "@/components/home/assistant-session";
import { useAssistantProgress } from "./useAssistantProgress";
import { useAssistantChat } from "./useAssistantChat";
import { useAssistantChatId } from "./useAssistantChatId";
import { SettingsSelect } from "@/components/settings/SettingsContent";
import { useAssistantModels } from "@/hooks/use-assistant-models";
import {
  progressPercentByGenerationId,
  remainingSecondsByGenerationId,
} from "@/ltx-io/components/GenerationQueue/generationQueuePanelModel";
import { assetQueryKeys } from "@/ltx-io/hooks/assetQueryKeys";
import {
  generationQueryKeys,
} from "@/ltx-io/hooks/generationQueryKeys";
import { useCancelGeneration } from "@/ltx-io/hooks/useCancelGeneration";
import { useDeleteGeneration } from "@/ltx-io/hooks/useDeleteGeneration";
import { useGenerationQueue } from "@/ltx-io/hooks/useGenerationQueue";
import { useInvalidateAssetsOnSucceededSnapshot } from "@/ltx-io/hooks/useInvalidateAssetsOnGenerationSuccess";
import { useRetryGeneration } from "@/ltx-io/hooks/useRetryGeneration";
import { isInFlightGeneration } from "@/ltx-io/lib/resultsFeedModel";
import { unwrapApiResult } from "@/ltx-io/lib/unwrapApiResult";
import { useExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";
import {
  GENERATION_POLLING_QUERY_OPTIONS,
  generationRefetchInterval,
} from "@/ltx-io/runtime/generationPollingPolicy";
import { ResultFrame } from "@/ltx-io/screens/Feature/results/ResultFrame";

import { AssistantAssetPreview } from "./AssistantAssetPreview";
import { AssistantAttachmentList, AssistantAttachments } from "./AssistantAttachments";
import { AssistantConversationMenu } from "./AssistantConversationMenu";
import {
  assistantSelectionIssue,
} from "./assistant-selection";
import { useAssistantChoice } from "./useAssistantChoice";
import { geminiModelLabel } from "@/lib/gemini-model";
import { randomUUID } from "@/lib/random-uuid";
import styles from "./DesktopAssistantPanel.module.scss";

// Plain text: a model-written link would open from the paired phone.
const MARKDOWN_BLOCKED = ["img", "a"];
const MARKDOWN_PLUGINS = [remarkGfm];
const MARKDOWN_COMPONENTS: Components = {
  table: ({ node: _node, ...props }) => (
    <div className={styles.tableScroll}>
      <table {...props} />
    </div>
  ),
};

function AssistantMarkdown({ children }: { children: string }) {
  return (
    <Markdown
      skipHtml
      disallowedElements={MARKDOWN_BLOCKED}
      unwrapDisallowed
      remarkPlugins={MARKDOWN_PLUGINS}
      components={MARKDOWN_COMPONENTS}
    >
      {children}
    </Markdown>
  );
}
// Not a model: choosing it opens settings and the controlled value snaps back.
const SETTINGS_OPTION = "__assistant-settings__";

export function DesktopAssistantPanel({
  open,
  onClose,
  fullPage = false,
  attachmentRequest,
  onExpand,
  onDock,
  headerLeading,
  headerActions,
}: {
  open: boolean;
  onClose?: () => void;
  fullPage?: boolean;
  attachmentRequest?: { id: string } | null;
  onExpand?: () => void;
  onDock?: () => void;
  /** Full page only, before the title (the shell's Back and Forward buttons). */
  headerLeading?: ReactNode;
  headerActions?: ReactNode;
}) {
  const assistantChoice = useAssistantChoice(open);
  const { openSettings } = assistantChoice;
  const { chatId, openChat, newChat } = useAssistantChatId({ fullPage, open });
  const [error, setError] = useState<string | null>(null);
  const chat = useAssistantChat(chatId, open, (err) => setError(err.message));
  const conversation = chat.conversation;
  const messages = assistantChatMessages(conversation);
  const turn = assistantTurnState(conversation);
  const queue = conversation?.queue ?? [];
  const queueOnHold = conversation?.queue_on_hold ?? false;
  const [draft, setDraft] = useState("");
  const [attachmentIds, setAttachmentIds] = useState<string[]>([]);
  const [draggingAsset, setDraggingAsset] = useState(false);
  const cancelGeneration = useCancelGeneration();
  const retryGeneration = useRetryGeneration();
  const deleteGeneration = useDeleteGeneration();
  const jobBusy =
    cancelGeneration.isPending || retryGeneration.isPending || deleteGeneration.isPending;
  const { api, generationPolling, mediaInput } = useExploreRuntime();
  const assistantModels = useAssistantModels(open, api.listAssistantModels);
  const messagesRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const [atBottom, setAtBottom] = useState(true);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const lastAttachmentRequest = useRef<typeof attachmentRequest>(null);
  const addAttachments = useCallback(
    (ids: string[]) => {
      setAttachmentIds((current) =>
        addAssistantAttachments(current, ids, MAX_ASSISTANT_ATTACHMENTS),
      );
    },
    [],
  );
  useEffect(() => {
    if (!attachmentRequest || lastAttachmentRequest.current === attachmentRequest) return;
    lastAttachmentRequest.current = attachmentRequest;
    if (!attachmentIds.includes(attachmentRequest.id)) {
      if (attachmentIds.length >= MAX_ASSISTANT_ATTACHMENTS) {
        setError(`You can attach up to ${MAX_ASSISTANT_ATTACHMENTS} assets per message.`);
        return;
      }
      addAttachments([attachmentRequest.id]);
    }
    composerRef.current?.focus();
  }, [attachmentRequest, attachmentIds, addAttachments]);
  const draftChatId = useRef(chatId);
  useEffect(() => {
    // Only on a switch between chats: the first id can arrive after Ask assistant attached.
    if (draftChatId.current && draftChatId.current !== chatId) {
      setDraft("");
      setAttachmentIds([]);
    }
    draftChatId.current = chatId;
  }, [chatId]);
  const queryClient = useQueryClient();
  const importMutation = useMutation({
    mutationFn: async (files: File[]) => {
      for (const file of files) {
        try {
          const asset = await mediaInput.ingestDroppedFile(file);
          queryClient.setQueryData(assetQueryKeys.detail(asset.id), asset);
          addAttachments([asset.id]);
        } catch (error) {
          throw new Error(
            `Could not import ${file.name}: ${error instanceof Error ? error.message : "Unsupported or unreadable file."}`,
          );
        }
      }
    },
    onError: (error) => setError(error.message),
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: assetQueryKeys.all });
    },
    retry: false,
  });
  const resultIds = [
    ...new Set(messages.flatMap((message) => message.generationIds)),
  ];
  const liveIds = assistantLiveGenerationIds(conversation);
  const results = useQueries({
    queries: resultIds.map((id) =>
      queryOptions({
        queryKey: generationQueryKeys.detail(id),
        queryFn: async () => {
          const response = await api.getGeneration(id);
          if (!response.ok && response.error.code === "GENERATION_NOT_FOUND") return null;
          return unwrapApiResult(response);
        },
        ...GENERATION_POLLING_QUERY_OPTIONS,
        refetchOnWindowFocus: open,
        refetchInterval: (query) =>
          !open || query.state.data === null
            ? false
            : generationRefetchInterval(
                generationPolling,
                {
                  ...query.state,
                  data: query.state.data ? [query.state.data] : undefined,
                },
                liveIds.has(id),
              ),
      }),
    ),
  });
  useInvalidateAssetsOnSucceededSnapshot(
    results.flatMap((result) => (result.data ? [result.data] : [])),
  );
  const queueQuery = useGenerationQueue({
    api,
    generationPolling,
    isLive:
      open && results.some((result) => result.data && isInFlightGeneration(result.data)),
  });
  const progressById = progressPercentByGenerationId(queueQuery.data);
  const remainingById = remainingSecondsByGenerationId(queueQuery.data);
  const resultsById = new Map(resultIds.map((id, index) => [id, results[index]]));

  const busy = assistantChatBusy(conversation, chatId, {
    stopRunId: chat.stop.isPending ? chat.stop.variables : undefined,
    continueRunId: chat.continueRun.isPending ? chat.continueRun.variables : undefined,
    retryConversationId: chat.retryFailed.isPending
      ? chat.retryFailed.variables?.conversation_id
      : undefined,
    deleteConversationId: chat.deleteChat.isPending ? chat.deleteChat.variables : undefined,
  });
  const runId = turn.kind === "idle" ? null : turn.runId;
  const hasRun = turn.kind !== "idle" || busy;
  const turnSelection = assistantTurnSelection(conversation);
  const chosen = { provider: assistantChoice.provider, localModel: assistantChoice.localModel };
  // A running turn keeps the model it started with; the shared choice applies to the next message.
  const selected = {
    provider: (hasRun && turnSelection.provider) || chosen.provider,
    model: (hasRun && turnSelection.model) || chosen.localModel,
  };
  const willQueue = hasRun || queue.length > 0;
  const selectionInput = {
    provider: chosen.provider,
    model: chosen.localModel,
    hasGeminiApiKey: assistantChoice.hasGeminiApiKey,
    models: assistantModels.data?.models,
    settingsName: assistantChoice.settingsName,
  };
  const issue = assistantChoice.loaded
    ? assistantSelectionIssue(selectionInput)
    : "Loading assistant settings...";
  const selectionIssue = assistantChoice.loadError ?? issue;
  // Until a switch from here or Settings is saved, a send would still go out on the previous model.
  const switchingModel = assistantChoice.switching;
  function chooseModel(...args: Parameters<typeof assistantChoice.select>) {
    setError(null);
    assistantChoice.select(...args).catch((err: Error) => setError(err.message));
  }
  const canSend =
    !!draft.trim() &&
    !!chatId &&
    !chat.enqueue.isPending &&
    !importMutation.isPending &&
    !switchingModel &&
    issue === null;

  const progress = useAssistantProgress(
    api.streamAssistantEvents,
    open && (turn.kind === "working" || turn.kind === "waiting") ? turn.runId : null,
    chat.continueRun.submittedAt,
    chat.refresh,
  );
  const lastTurn = conversation?.turns.at(-1);
  // Keep the streamed reply until the stored one that replaces it has loaded.
  const streamedText =
    progress.active || lastTurn?.running || !lastTurn?.response ? progress.text : "";

  useEffect(() => {
    if (open) endRef.current?.scrollIntoView({ block: "end" });
  }, [open, messages.length, busy]);

  useEffect(() => {
    const root = messagesRef.current;
    const end = endRef.current;
    if (!open || !root || !end) return;
    const observer = new IntersectionObserver(
      ([entry]) => setAtBottom(entry.isIntersecting),
      { root, rootMargin: "0px 0px 48px 0px" },
    );
    observer.observe(end);
    return () => observer.disconnect();
  }, [open]);

  function send() {
    const text = draft.trim();
    if (!canSend) return;
    setError(null);
    chat.enqueue.mutate(
      {
        id: randomUUID(),
        text,
        asset_ids: attachmentIds,
        provider: chosen.provider,
        model:
          chosen.provider === "local"
            ? chosen.localModel
            : assistantChoice.geminiModel || undefined,
      },
      {
        onSuccess: () => {
          setDraft((current) => (current.trim() === text ? "" : current));
          setAttachmentIds([]);
          composerRef.current?.focus();
        },
      },
    );
  }

  function stopRun() {
    if (!runId || chat.stop.isPending) return;
    setError(null);
    chat.stop.mutate(runId);
  }

  async function jobAction(id: string, action: "cancel" | "retry" | "delete") {
    if (jobBusy) return;
    setError(null);
    try {
      const mutations = {
        cancel: cancelGeneration,
        retry: retryGeneration,
        delete: deleteGeneration,
      };
      await mutations[action].mutateAsync(id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not update the video job.");
    }
  }

  return (
    <aside
      className={styles.panel}
      data-full-page={fullPage || undefined}
      role={fullPage ? "main" : undefined}
      hidden={!open}
      aria-label="Video assistant"
      onKeyDown={(event) => {
        if (
          event.key === "Escape" &&
          !event.defaultPrevented &&
          !event.nativeEvent.isComposing &&
          !(
            event.target instanceof Element &&
            event.target.closest('[role="dialog"], [role="menu"], [role="listbox"]')
          ) &&
          hasRun
        ) {
          event.preventDefault();
          event.stopPropagation();
          if (!event.repeat) stopRun();
        }
      }}
    >
      <header className={styles.header}>
        {fullPage && headerLeading}
        <div className={styles.title}>
          <Text as="h2" variant="heading" size="xs">
            Assistant
          </Text>
        </div>
        <AssistantConversationMenu
          currentId={chatId}
          disabled={chat.deleteChat.isPending}
          onNew={() => newChat()}
          onChosen={() => composerRef.current?.focus()}
          onOpen={openChat}
          onDelete={(id) =>
            chat.deleteChat.mutate(id, {
              onSuccess: () => {
                if (id === chatId) newChat({ replace: true });
              },
            })
          }
        />
        {fullPage && headerActions}
        {(fullPage ? onDock : onExpand) && (
          <Tooltip content={fullPage ? "Open beside assets" : "Open full-page assistant"}>
            <Button
              appearance="neutral"
              hierarchy="plain"
              size="md"
              isIconOnly
              leftIcon={fullPage ? <PanelRight size={18} /> : <Maximize2 size={18} />}
              aria-label={fullPage ? "Open beside assets" : "Open full-page assistant"}
              onClick={fullPage ? onDock : onExpand}
            />
          </Tooltip>
        )}
        {!fullPage && onClose && (
          <Tooltip content="Close assistant">
            <Button
              appearance="neutral"
              hierarchy="plain"
              size="md"
              isIconOnly
              leftIcon={<X size={18} />}
              aria-label="Close assistant"
              onClick={onClose}
            />
          </Tooltip>
        )}
      </header>
      <div ref={messagesRef} className={styles.messages} aria-live="polite" aria-busy={busy}>
        {messages.length === 0 && !chat.loading && (
          <div className={styles.empty}>
            <Text as="h3" variant="heading" size="xs">
              What would you like to create?
            </Text>
          </div>
        )}
        {messages.map((message) => {
          const ids = message.generationIds;
          const generationPreviewIds = new Set(
            ids.map((id) => resultsById.get(id)?.data?.outputs[0]?.id),
          );
          return (
            <article
              key={message.id}
              className={styles.message}
              data-role={message.role}
              aria-label={message.role === "user" ? "You" : "Assistant"}
            >
              <div className={styles.messageText}>
                {message.role === "user" ? (
                  message.text
                ) : (
                  <AssistantMarkdown>{message.text}</AssistantMarkdown>
                )}
              </div>
              {open &&
                message.assetIds
                  ?.filter((id) => !generationPreviewIds.has(id))
                  .map((id) => <AssistantAssetPreview key={id} assetId={id} />)}
              {open &&
                ids.map((id) => {
                  const result = resultsById.get(id);
                  const generation = result?.data;
                  return generation ? (
                    <div
                      key={id}
                      aria-busy={jobBusy}
                      className={`${styles.generation} ${jobBusy ? styles.disabled : ""}`}
                    >
                      <ResultFrame
                        generation={generation}
                        progressPercent={progressById.get(generation.id)}
                        remainingSeconds={remainingById.get(generation.id)}
                        onCancel={(id) => void jobAction(id, "cancel")}
                        onRetry={(id) => void jobAction(id, "retry")}
                        onDelete={(id) => void jobAction(id, "delete")}
                      />
                    </div>
                  ) : (
                    <small key={id}>
                      {result?.isPending
                        ? "Loading video job..."
                        : result?.isError
                          ? "Could not load video status."
                          : "Video job is no longer in recent history."}
                    </small>
                  );
                })}
            </article>
          );
        })}
        {streamedText && (
          <div className={styles.messageText}>
            <AssistantMarkdown>{streamedText}</AssistantMarkdown>
          </div>
        )}
        {(turn.kind === "working" || busy || progress.active || chat.loading) && (
          <div className={styles.working}>
            <LoaderCircle size={16} className={styles.spinner} />{" "}
            {chat.stop.isPending
              ? "Stopping..."
              : chat.loading
                ? "Loading conversation..."
                : progress.activity || "Thinking..."}
          </div>
        )}
        <div ref={endRef} />
      </div>
      <div className={styles.actionArea}>
        {!atBottom && (
          <button
            type="button"
            className={styles.scrollToEnd}
            aria-label="Scroll to latest message"
            onClick={() => endRef.current?.scrollIntoView({ block: "end", behavior: "smooth" })}
          >
            <ArrowDown size={18} />
          </button>
        )}
        {(error || chat.loadError) && (
          <p className={styles.error} role="alert">
            {error ?? chat.loadError?.message}
          </p>
        )}
        {turn.kind === "paused" && (
          <Button
            appearance="neutral"
            hierarchy="secondary"
            size="md"
            leftIcon={<RotateCcw size={16} />}
            label="Continue"
            disabled={busy}
            onClick={() => {
              setError(null);
              chat.continueRun.mutate(turn.runId);
            }}
          />
        )}
        {turn.kind === "failed" && (
          <Button
            appearance="neutral"
            hierarchy="secondary"
            size="md"
            leftIcon={<RotateCcw size={16} />}
            label="Retry request"
            disabled={busy}
            onClick={() => {
              setError(null);
              chat.retryFailed.mutate(turn.request);
            }}
          />
        )}
        {queue.length > 0 && (
          <section className={styles.queue} aria-label="Queued messages">
            <div className={styles.queueHeader}>
              <Text as="span" variant="body" size="sm">
                {queue.length} queued{queueOnHold ? " · On hold" : ""}
              </Text>
              {queueOnHold && (
                <Tooltip content="Send queued messages">
                  <Button
                    appearance="neutral"
                    hierarchy="plain"
                    size="sm"
                    isIconOnly
                    leftIcon={<Play size={14} />}
                    aria-label="Send queued messages"
                    disabled={hasRun || chat.releaseQueue.isPending}
                    onClick={() =>
                      chat.releaseQueue.mutate(undefined)
                    }
                  />
                </Tooltip>
              )}
            </div>
            <ol className={styles.queueList}>
              {queue.map((message) => (
                <li key={message.id} className={styles.queuedMessage}>
                  <div className={styles.queuedText}>
                    <Text
                      as="span"
                      variant="body"
                      size="md"
                      shouldTruncate
                      title={message.text}
                    >
                      {message.text}
                      {message.asset_ids.length > 0
                        ? ` (${message.asset_ids.length} attached)`
                        : ""}
                    </Text>
                    {message.error && (
                      <span className={styles.queuedError}>
                        Could not start: {message.error}
                      </span>
                    )}
                  </div>
                  <Tooltip content="Remove queued message">
                    <Button
                      appearance="neutral"
                      hierarchy="plain"
                      size="sm"
                      isIconOnly
                      leftIcon={<X size={14} />}
                      aria-label={`Remove queued message: ${message.text}`}
                      disabled={chat.removeQueued.isPending}
                      onClick={() =>
                        chat.removeQueued.mutate(message.id)
                      }
                    />
                  </Tooltip>
                </li>
              ))}
            </ol>
          </section>
        )}
        <form
          className={styles.composer}
          data-dragging={draggingAsset || undefined}
          onDragOver={(event) => {
            const isAsset = event.dataTransfer.types.some((type) =>
              ["assetid", "assetids"].includes(type.toLowerCase()),
            );
            if (!isAsset && !event.dataTransfer.types.includes("Files")) return;
            event.preventDefault();
            event.stopPropagation();
            event.dataTransfer.dropEffect = importMutation.isPending ? "none" : "copy";
            setDraggingAsset(!importMutation.isPending);
          }}
          onDragLeave={(event) => {
            if (
              !(event.relatedTarget instanceof Node) ||
              !event.currentTarget.contains(event.relatedTarget)
            ) {
              setDraggingAsset(false);
            }
          }}
          onDrop={(event) => {
            event.preventDefault();
            event.stopPropagation();
            setDraggingAsset(false);
            if (importMutation.isPending) return;
            const files = Array.from(event.dataTransfer.files);
            const single = event.dataTransfer.getData("assetId");
            const multiple = event.dataTransfer.getData("assetIds");
            if (!single && !multiple && files.length > 0) {
              if (attachmentIds.length + files.length > MAX_ASSISTANT_ATTACHMENTS) {
                setError(`You can attach up to ${MAX_ASSISTANT_ATTACHMENTS} assets per message.`);
                return;
              }
              setError(null);
              importMutation.mutate(files);
              return;
            }
            try {
              const ids = z
                .array(z.string().min(1))
                .min(1)
                .parse(multiple ? JSON.parse(multiple) : [single]);
              const combined = [...new Set([...attachmentIds, ...ids])];
              if (combined.length > MAX_ASSISTANT_ATTACHMENTS) {
                setError(`You can attach up to ${MAX_ASSISTANT_ATTACHMENTS} assets per message.`);
                return;
              }
              addAttachments(ids);
              setError(null);
              composerRef.current?.focus();
            } catch {
              setError("Drop an asset from the app or a video, image or audio file.");
            }
          }}
          onSubmit={(event) => {
            event.preventDefault();
            send();
          }}
        >
          <AssistantAttachmentList
            ids={attachmentIds}
            onChange={setAttachmentIds}
            disabled={importMutation.isPending}
          />
          <textarea
            ref={composerRef}
            id="assistant-message"
            aria-label="Message assistant"
            placeholder="Message assistant..."
            value={draft}
            maxLength={12000}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing &&
                event.nativeEvent.keyCode !== 229
              ) {
                event.preventDefault();
                if (!event.repeat) send();
              }
            }}
            rows={2}
          />
          <div className={styles.composerActions}>
            <AssistantAttachments
              ids={attachmentIds}
              onAdd={(id) => {
                setError(null);
                addAttachments([id]);
              }}
              onError={setError}
              disabled={importMutation.isPending}
            />
            <div className={styles.modelSelect}>
              <SettingsSelect
                aria-label="Assistant model"
                value={selected.provider === "gemini" ? "gemini" : selected.model}
                disabled={willQueue || switchingModel}
                onChange={(event) => {
                  const value = event.target.value;
                  if (value === SETTINGS_OPTION) {
                    openSettings?.();
                  } else if (value === "gemini") {
                    chooseModel("gemini");
                  } else {
                    const model = assistantModels.data?.models.find(
                      (item) => item.model === value && item.installed,
                    );
                    if (model) chooseModel("local", model.model);
                  }
                }}
              >
                <option value="gemini" disabled={!assistantChoice.hasGeminiApiKey}>
                  {geminiModelLabel(assistantChoice.geminiModel)}
                  {assistantChoice.hasGeminiApiKey ? "" : " (add API key in Settings)"}
                </option>
                {assistantModels.data?.models.map((model) => (
                  <option
                    key={model.model}
                    value={model.model}
                    disabled={!model.installed}
                  >
                    {model.name}
                    {model.installed ? "" : " (not downloaded)"}
                  </option>
                ))}
                {openSettings && (
                  <option value={SETTINGS_OPTION}>Assistant settings...</option>
                )}
              </SettingsSelect>
              <ChevronDown size={14} aria-hidden="true" />
            </div>
            {importMutation.isPending && (
              <span role="status">
                <Text as="span" variant="body" size="sm">
                  Importing...
                </Text>
              </span>
            )}
            {hasRun && (
              <Tooltip content="Stop assistant (Esc)">
                <Button
                  appearance="neutral"
                  hierarchy="secondary"
                  size="md"
                  isIconOnly
                  leftIcon={<Square size={14} fill="currentColor" />}
                  aria-label="Stop assistant"
                  aria-keyshortcuts="Escape"
                  disabled={chat.stop.isPending}
                  onClick={stopRun}
                />
              </Tooltip>
            )}
            <Tooltip
              content={
                selectionIssue ??
                (willQueue ? "Queue message (Enter)" : "Send message (Enter)")
              }
            >
              <Button
                appearance="brand"
                hierarchy="primary"
                size="md"
                isIconOnly
                leftIcon={willQueue ? <ListPlus size={18} /> : <ArrowUp size={18} />}
                type="submit"
                aria-label={willQueue ? "Queue message" : "Send message"}
                disabled={!canSend}
              />
            </Tooltip>
          </div>
        </form>
        {(assistantChoice.loadError ?? (assistantChoice.loaded ? issue : null)) && (
          <footer className={styles.footer}>
            <Text as="span" variant="body" size="xs">
              {selectionIssue}
            </Text>
            {openSettings && !assistantChoice.loadError && (
              <button
                type="button"
                className={styles.footerLink}
                onClick={() =>
                  openSettings(chosen.provider === "gemini" ? "geminiKey" : "localModel")
                }
              >
                Open Settings
              </button>
            )}
          </footer>
        )}
      </div>
    </aside>
  );
}
