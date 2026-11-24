import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useState } from "react";
import { matchPath, useLocation, useNavigate } from "react-router";
import { z } from "zod";

import { randomUUID } from "@/lib/random-uuid";
import { useExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";
import { assistantChatPath, paths } from "@/paths";

import { assistantConversationsQuery } from "./AssistantConversationMenu";

const chatIdSchema = z.string().uuid();

/**
 * Which chat the panel shows. On the full page the URL decides; bare or invalid
 * /assistant URLs open the current, else most recent, chat. The docked panel keeps
 * its own choice and starts on the most recent chat. A new chat is a fresh id,
 * valid before its first message.
 */
export function useAssistantChatId({ fullPage, open }: { fullPage: boolean; open: boolean }) {
  const { api } = useExploreRuntime();
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const fromUrl = fullPage ? matchPath(paths.assistantChat, pathname)?.params.chatId : undefined;
  const urlChatId = chatIdSchema.safeParse(fromUrl).success ? fromUrl! : null;
  const [chatId, setChatId] = useState<string | null>(null);
  const [freshChatId] = useState(randomUUID);
  const conversations = useQuery({
    ...assistantConversationsQuery(api),
    enabled: open && !chatId && !urlChatId,
  });
  const fallback = conversations.isPending
    ? null
    : (conversations.data?.[0]?.conversation_id ?? freshChatId);

  useEffect(() => {
    if (!open) return;
    if (fullPage && urlChatId) {
      setChatId(urlChatId);
    } else if (fullPage) {
      const target = chatId ?? fallback;
      if (target) navigate(assistantChatPath(target), { replace: true });
    } else if (!chatId && fallback) {
      setChatId(fallback);
    }
  }, [open, fullPage, urlChatId, chatId, fallback, navigate]);

  const openChat = useCallback(
    (id: string, options?: { replace?: boolean }) => {
      if (fullPage) navigate(assistantChatPath(id), { replace: options?.replace });
      setChatId(id);
    },
    [fullPage, navigate],
  );
  const newChat = useCallback(
    (options?: { replace?: boolean }) => openChat(randomUUID(), options),
    [openChat],
  );

  // Expand lands on bare /assistant; keep the open chat until the URL names it.
  return { chatId: fullPage ? (urlChatId ?? chatId) : chatId, openChat, newChat };
}
