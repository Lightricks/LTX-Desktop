import { useRef, type ReactNode } from "react";

import { FolderOpen } from "lucide-react";
import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import RemoveIcon from "@ds/assets/Icons/Remove.svg?react";
import { Flex } from "@ds/layout/Flex/Flex";

import type { ExploreListedAsset } from "@/lib/explore-contract";
import { revealInFolderLabel } from "@/lib/revealInFolderLabel";
import { useModalContext } from "@/ltx-io/components/shared/Modal/ModalContext";
import { useAssetDownload } from "@/ltx-io/hooks/useAssetDownload";
import { useExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";

import {
  resolveAssetLibraryFileAction,
  type AssetLibraryFileAction,
} from "./assetLibraryFileAction";
import {
  ASSETS_DELETE_CONFIRM_BUTTON,
  ASSETS_DELETE_CONFIRM_DESCRIPTION,
  ASSETS_DELETE_CONFIRM_TITLE,
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

export function AssetDeleteConfirmation({
  title,
  description,
  onDismiss,
  onConfirm,
}: {
  title: string;
  description: string;
  onDismiss: () => void;
  onConfirm?: () => void;
}) {
  return (
    <Flex direction="column" gap="md" style={{ maxWidth: 400 }}>
      <Flex direction="column" gap="xs">
        <Text as="h2" variant="heading" size="sm">
          {title}
        </Text>
        <Text as="p" variant="body" size="md">
          {description}
        </Text>
      </Flex>
      <Flex gap="sm" justify="end">
        {onConfirm ? (
          <>
            <Button
              appearance="neutral"
              hierarchy="plain"
              size="md"
              label="Cancel"
              onClick={onDismiss}
            />
            <Button
              appearance="danger"
              hierarchy="primary"
              size="md"
              label={ASSETS_DELETE_CONFIRM_BUTTON}
              onClick={onConfirm}
            />
          </>
        ) : (
          <Button
            appearance="brand"
            hierarchy="primary"
            size="md"
            label={ASSETS_IN_USE_BUTTON}
            onClick={onDismiss}
          />
        )}
      </Flex>
    </Flex>
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
  const { hideConfirmation, showConfirmation } = useModalContext();
  const isDeleteSubmittingRef = useRef(false);
  const fileAction = resolveAssetLibraryFileAction({
    revealInFolder: runtime.revealInFolder,
    path: asset.path,
    mediaUrl,
  });

  const showInUseConfirmation = () => {
    showConfirmation({
      confirmationName: "asset-delete-in-use",
      content: (
        <AssetDeleteConfirmation
          title={ASSETS_IN_USE_TITLE}
          description={ASSETS_IN_USE_DESCRIPTION}
          onDismiss={() => hideConfirmation("done")}
        />
      ),
    });
  };

  const showDeleteConfirmation = () => {
    if (isDeletePending) return;
    showConfirmation({
      confirmationName: "asset-delete-confirm",
      content: (
        <AssetDeleteConfirmation
          title={ASSETS_DELETE_CONFIRM_TITLE}
          description={ASSETS_DELETE_CONFIRM_DESCRIPTION}
          onDismiss={() => hideConfirmation("cancel")}
          onConfirm={() => {
            if (isDeleteSubmittingRef.current) {
              hideConfirmation("confirm");
              return;
            }
            isDeleteSubmittingRef.current = true;
            hideConfirmation("confirm");
            void onDelete(asset.id)
              .catch((error: unknown) => {
                if (isAssetDeleteInUseError(error)) showInUseConfirmation();
              })
              .finally(() => {
                isDeleteSubmittingRef.current = false;
              });
          }}
        />
      ),
    });
  };

  if (runtime.deleteAsset !== null) {
    return (
      <div className={styles.actions}>
        <OverlayButton
          label={asset.in_use ? "Used as an input" : "Delete file"}
          icon={<RemoveIcon />}
          disabled={asset.in_use || isDeletePending}
          onClick={showDeleteConfirmation}
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
