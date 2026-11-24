import { Button } from "@ds/Button/Button";
import { ItemAction } from "@ds/DropdownItems/ItemAction/ItemAction";
import { DropdownMenu } from "@ds/DropdownMenu/DropdownMenu";
import type { MenuItem } from "@ds/DropdownMenu/DropdownMenuItem";
import { useThemedPortalContainer } from "@ds/styles/themes/useTheme";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import { queryOptions, useQuery } from "@tanstack/react-query";
import { History, Plus, Trash2 } from "lucide-react";
import { useRef, useState } from "react";

import { formatRelativeTime } from "@/ltx-io/lib/formatRelativeTime";
import { unwrapApiResult } from "@/ltx-io/lib/unwrapApiResult";
import { type ExploreApi, useExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";

export const ASSISTANT_CONVERSATIONS_QUERY_KEY = ["assistant", "conversations"] as const;

export function assistantConversationsQuery(api: Pick<ExploreApi, "listAssistantConversations">) {
  return queryOptions({
    queryKey: ASSISTANT_CONVERSATIONS_QUERY_KEY,
    queryFn: async () => unwrapApiResult(await api.listAssistantConversations()),
  });
}

export function AssistantConversationMenu({
  currentId,
  disabled,
  onNew,
  onOpen,
  onDelete,
  onChosen,
}: {
  currentId: string | null;
  disabled: boolean;
  onNew: () => void;
  onOpen: (conversationId: string) => void;
  onDelete: (conversationId: string) => void;
  /** After a new or other chat is chosen and the menu has closed. */
  onChosen: () => void;
}) {
  const { api } = useExploreRuntime();
  const portalContainer = useThemedPortalContainer();
  const [open, setOpen] = useState(false);
  const chosen = useRef(false);
  const conversations = useQuery({ ...assistantConversationsQuery(api), enabled: open });

  const items: MenuItem[] = [
    {
      render: (
        <ItemAction
          text="New chat"
          leftIcon={<Plus size={16} />}
          onClick={() => {
            chosen.current = true;
            onNew();
          }}
        />
      ),
    },
  ];
  if (conversations.isPending || conversations.isError) {
    items.push({
      hasUpperSeparator: true,
      isVisualOnly: true,
      render: <ItemAction text={conversations.isError ? "Could not load chats" : "Loading chats…"} />,
    });
  }
  conversations.data?.forEach(({ conversation_id, title, updated_at }, index) => {
    items.push({
      hasUpperSeparator: index === 0,
      render: (
        <ItemAction
          text={title || "Untitled chat"}
          textTitle={title}
          caption={formatRelativeTime(updated_at)}
          isSelected={conversation_id === currentId}
          shouldTruncateText
          rightCta={
            <Tooltip content="Delete chat">
              <Button
                appearance="neutral"
                hierarchy="plain"
                size="sm"
                isIconOnly
                leftIcon={<Trash2 size={14} />}
                aria-label={`Delete ${title || "untitled chat"}`}
                onClick={(event) => {
                  // The row is a menu item; without this the click also opens the chat.
                  event.stopPropagation();
                  if (window.confirm("Delete this chat? Videos it made stay in your assets."))
                    onDelete(conversation_id);
                }}
              />
            </Tooltip>
          }
          onClick={() => {
            chosen.current = true;
            if (conversation_id !== currentId) onOpen(conversation_id);
          }}
        />
      ),
    });
  });

  return (
    <DropdownMenu
      items={items}
      align="end"
      minWidth="260px"
      maxWidth="340px"
      maxHeight="420px"
      disabled={disabled}
      portalContainer={portalContainer}
      onMenuStateChange={setOpen}
      onCloseAutoFocus={(event) => {
        if (!chosen.current) return;
        // Radix would return focus to the Chats button.
        chosen.current = false;
        event.preventDefault();
        onChosen();
      }}
    >
      <span>
        <Tooltip content={disabled ? "Deleting this chat" : "Chats"}>
          <Button
            appearance="neutral"
            hierarchy="plain"
            size="md"
            isIconOnly
            leftIcon={<History size={18} />}
            aria-label="Chats"
            disabled={disabled}
          />
        </Tooltip>
      </span>
    </DropdownMenu>
  );
}
