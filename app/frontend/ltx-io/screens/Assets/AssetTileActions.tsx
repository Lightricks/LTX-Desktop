import { useRef, type ReactNode } from "react";

import { FolderOpen, MessageSquare } from "lucide-react";
import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import RemoveIcon from "@ds/assets/Icons/Remove.svg?react";
import { Flex } from "@ds/layout/Flex/Flex";

import type { ExploreListedAsset } from "@/lib/explore-contract";
import { revealInFolderLabel } from "@/lib/revealInFolderLabel";
import { useModalContext } from "@/ltx-io/components/shared/Modal/ModalContext";
import { useAssetDownload } from "@/ltx-io/hooks/useAssetDownload";
import { useConfirmDeleteAction } from "@/ltx-io/hooks/useConfirmDeleteAction";
import { useExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";

import {
  resolveAssetLibraryFileAction,
  type AssetLibraryFileAction,
} from "./assetLibraryFileAction";
import {
  ASSETS_DELETE_CONFIRMATION,
  ASSETS_IN_USE_BUTTON,
  ASSETS_IN_USE_DESCRIPTION,
  ASSETS_IN_USE_TITLE,
} from "./assetsCopy";
import { isAssetDeleteInUseError } from "./useDeleteAsset";
import styles from "./MediaTile.module.scss";

export function OverlayButton({
  label,
  icon,
  disabled = false,
  onClick,
}: {
  label: string;
  icon: ReactNode;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <Tooltip content={label}>
      <span>
        <Button
          appearance="overlay"
          hierarchy="secondary"
          size="md"
          isIconOnly
          leftIcon={icon}
          aria-label={label}
          disabled={disabled}
          onClick={(event) => {
            event.stopPropagation();
            onClick();
          }}
        />
      </span>
    </Tooltip>
  );
}

function AssetInUseNotice({ onDismiss }: { onDismiss: () => void }) {
  return (
    <Flex direction="column" gap="md" style={{ maxWidth: 400 }}>
      <Flex direction="column" gap="xs">
        <Text as="h2" variant="heading" size="sm">
          {ASSETS_IN_USE_TITLE}
        </Text>
        <Text as="p" variant="body" size="md">
          {ASSETS_IN_USE_DESCRIPTION}
        </Text>
      </Flex>
      <Flex gap="sm" justify="end">
        <Button
          appearance="brand"
          hierarchy="primary"
          size="md"
          label={ASSETS_IN_USE_BUTTON}
          onClick={onDismiss}
        />
      </Flex>
    </Flex>
  );
}

export function AssetDeleteButton({
  asset,
  onDelete,
  isDeletePending,
}: {
  asset: ExploreListedAsset;
  onDelete: (assetId: string) => Promise<void>;
  isDeletePending: boolean;
}) {
  const { hideConfirmation, showConfirmation } = useModalContext();
  const confirmDelete = useConfirmDeleteAction();
  const isDeleteSubmittingRef = useRef(false);

  const showInUseConfirmation = () => {
    showConfirmation({
      confirmationName: "asset-delete-in-use",
      content: <AssetInUseNotice onDismiss={() => hideConfirmation("done")} />,
    });
  };

  const runDeleteAsset = () => {
    if (isDeleteSubmittingRef.current) return;
    isDeleteSubmittingRef.current = true;
    void onDelete(asset.id)
      .catch((error: unknown) => {
        if (isAssetDeleteInUseError(error)) showInUseConfirmation();
      })
      .finally(() => {
        isDeleteSubmittingRef.current = false;
      });
  };

  return (
    <OverlayButton
      label={asset.in_use ? "Used as an input" : "Delete file"}
      icon={<RemoveIcon />}
      disabled={asset.in_use || isDeletePending}
      onClick={() => {
        if (isDeletePending) return;
        confirmDelete(ASSETS_DELETE_CONFIRMATION, runDeleteAsset);
      }}
    />
  );
}

function DesktopRevealButton({
  action,
  onReveal,
}: {
  action: AssetLibraryFileAction;
  onReveal: (path: string) => void;
}) {
  switch (action.kind) {
    case "reveal":
      return (
        <OverlayButton
          label={revealInFolderLabel()}
          icon={<FolderOpen />}
          onClick={() => onReveal(action.path)}
        />
      );
    case "download":
    case "none":
      return null;
    default: {
      const _exhaustive: never = action;
      return _exhaustive;
    }
  }
}

function RemoteDownloadAction({
  action,
  asset,
}: {
  action: AssetLibraryFileAction;
  asset: ExploreListedAsset;
}) {
  const download = useAssetDownload();

  switch (action.kind) {
    case "download":
      return (
        <div className={styles.actions}>
          <button
            type="button"
            className={styles.download}
            aria-label={`Download ${asset.name}`}
            onClick={(event) => {
              event.stopPropagation();
              download(asset);
            }}
          >
            Download
          </button>
        </div>
      );
    case "reveal":
    case "none":
      return null;
    default: {
      const _exhaustive: never = action;
      return _exhaustive;
    }
  }
}

export function AssetTileActions({
  asset,
  mediaUrl,
  onDelete,
  isDeletePending,
}: {
  asset: ExploreListedAsset;
  mediaUrl: string | null;
  onDelete: (assetId: string) => Promise<void>;
  isDeletePending: boolean;
}) {
  const runtime = useExploreRuntime();
  const fileAction = resolveAssetLibraryFileAction({
    revealInFolder: runtime.revealInFolder,
    path: asset.path,
    mediaUrl,
  });

  if (runtime.deleteAsset !== null) {
    return (
      <div className={styles.actions}>
        {runtime.askAssistant && (
          <OverlayButton label="Ask assistant" icon={<MessageSquare />}
            onClick={() => runtime.askAssistant?.(asset.id)} />
        )}
        <AssetDeleteButton
          asset={asset}
          onDelete={onDelete}
          isDeletePending={isDeletePending}
        />
        <DesktopRevealButton
          action={fileAction}
          onReveal={(path) => runtime.revealInFolder?.(path)}
        />
      </div>
    );
  }

  return <RemoteDownloadAction action={fileAction} asset={asset} />;
}
