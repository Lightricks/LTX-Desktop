import { Button } from "@ds/Button/Button";
import { ItemAction } from "@ds/DropdownItems/ItemAction/ItemAction";
import { DropdownMenu } from "@ds/DropdownMenu/DropdownMenu";
import type { MenuItem } from "@ds/DropdownMenu/DropdownMenuItem";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import { useThemedPortalContainer } from "@ds/styles/themes/useTheme";
import { useQueryClient } from "@tanstack/react-query";
import { Images, Plus, Upload, X } from "lucide-react";
import { useRef, useState } from "react";

import { MAX_ASSISTANT_ATTACHMENTS } from "@/components/home/assistant-session";
import type { ExploreAsset } from "@/lib/explore-contract";
import { assetQueryKeys } from "@/ltx-io/hooks/assetQueryKeys";
import { useAsset } from "@/ltx-io/hooks/useAsset";
import { useExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";
import { useOpenAssetPicker } from "@/ltx-io/screens/Assets/AssetPicker";
import { AssetsLightbox } from "@/ltx-io/screens/Assets/AssetsLightbox";
import {
  AUDIO_OR_VIDEO_ACCEPT,
  AUDIO_OR_VIDEO_EXTENSIONS,
  isSupportedDroppedAudio,
} from "@/ltx-io/screens/Feature/fields/audioAssetInput";
import {
  IMAGE_ACCEPT,
  IMAGE_EXTENSIONS,
  isSupportedDroppedImage,
} from "@/ltx-io/screens/Feature/fields/imageAssetInput";
import { isSupportedDroppedVideo } from "@/ltx-io/screens/Feature/fields/videoAssetInput";

import styles from "./AssistantAttachments.module.scss";
import { assistantMedia } from "./assistant-media";

function Attachment({
  id,
  remove,
  disabled,
}: {
  id: string;
  remove: () => void;
  disabled: boolean;
}) {
  const { data: asset } = useAsset(id);
  const runtime = useExploreRuntime();
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const thumbnail = asset
    ? (runtime.thumbUrlForAsset(asset) ??
      (asset.media_kind === "image" ? runtime.mediaUrlForAsset(asset) : null))
    : null;
  const { Icon, label } = assistantMedia[asset?.media_kind ?? "unknown"];
  const name = asset?.name ?? "Loading asset...";
  const hasThumbnail = thumbnail && failedUrl !== thumbnail;
  return (
    <div className={styles.attachment}>
      <Tooltip content={`${name} (${label})`}>
        <button
          type="button"
          className={styles.thumbnail}
          aria-label={`Preview ${name} (${label})`}
          disabled={!asset}
          onClick={() => setPreviewOpen(true)}
        >
          {hasThumbnail ? (
            <img src={thumbnail} alt="" onError={() => setFailedUrl(thumbnail)} />
          ) : (
            <Icon size={24} aria-hidden="true" />
          )}
          {hasThumbnail && (
            <span className={styles.kind}>
              <Icon size={12} aria-hidden="true" />
            </span>
          )}
        </button>
      </Tooltip>
      <Tooltip content="Remove attachment">
        <Button
          className={styles.remove}
          appearance="overlay"
          hierarchy="secondary"
          size="sm"
          isIconOnly
          leftIcon={<X size={14} />}
          aria-label={`Remove ${name}`}
          disabled={disabled}
          onClick={remove}
        />
      </Tooltip>
      {previewOpen && asset && (
        <AssetsLightbox
          items={[{ ...asset, in_use: true }]}
          activeAssetId={asset.id}
          onClose={() => setPreviewOpen(false)}
        />
      )}
    </div>
  );
}

type AttachmentProps = {
  ids: string[];
  onChange: (ids: string[]) => void;
  disabled: boolean;
};

export function AssistantAttachmentList({ ids, onChange, disabled }: AttachmentProps) {
  if (!ids.length) return null;
  return (
    <div className={styles.attachmentList} role="group" aria-label="Attached assets">
      {ids.map((id) => (
        <Attachment
          key={id}
          id={id}
          disabled={disabled}
          remove={() => onChange(ids.filter((item) => item !== id))}
        />
      ))}
    </div>
  );
}

const ATTACH_REQUEST = {
  title: "Attach media",
  accept: `${IMAGE_ACCEPT},${AUDIO_OR_VIDEO_ACCEPT}`,
  extensions: [...IMAGE_EXTENSIONS, ...AUDIO_OR_VIDEO_EXTENSIONS],
};

function isAttachableFile(file: File): boolean {
  return (
    isSupportedDroppedImage(file) ||
    isSupportedDroppedAudio(file) ||
    isSupportedDroppedVideo(file)
  );
}

export function AssistantAttachments({
  ids,
  onAdd,
  disabled,
  onError,
}: Omit<AttachmentProps, "onChange"> & {
  onAdd: (id: string) => void;
  onError: (message: string) => void;
}) {
  const { mediaInput } = useExploreRuntime();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const portalContainer = useThemedPortalContainer();
  const pickerOpened = useRef(false);
  const [choosing, setChoosing] = useState(false);
  const full = ids.length >= MAX_ASSISTANT_ATTACHMENTS;
  const add = (asset: ExploreAsset) => {
    queryClient.setQueryData(assetQueryKeys.detail(asset.id), asset);
    onAdd(asset.id);
  };
  const reportError = (error: unknown) =>
    onError(error instanceof Error ? error.message : "Could not attach that file.");
  const choose = () => {
    setChoosing(true);
    void mediaInput
      .choose({ ...ATTACH_REQUEST, fileInput: fileInputRef.current })
      .then((asset) => {
        if (asset) add(asset);
      })
      .catch(reportError)
      .finally(() => setChoosing(false));
  };
  const openAssets = useOpenAssetPicker({
    mediaKind: null,
    isEligibleFile: isAttachableFile,
    onSelect: add,
    onUpload: () =>
      mediaInput.choose({ ...ATTACH_REQUEST, fileInput: fileInputRef.current }),
    onDropFile: (file) =>
      void mediaInput.ingestDroppedFile(file).then(add).catch(reportError),
    onDropAssetId: onAdd,
  });
  const items: MenuItem[] = [
    // Each starts in this click so a phone can open its file picker.
    {
      render: (
        <ItemAction
          text="Upload from device"
          leftIcon={<Upload size={16} />}
          onClick={choose}
        />
      ),
    },
    {
      render: (
        <ItemAction
          text="Choose from assets"
          leftIcon={<Images size={16} />}
          onClick={() => {
            pickerOpened.current = true;
            openAssets();
          }}
        />
      ),
    },
  ];
  return (
    <div className={styles.attachments}>
      {/* Mounted so the Remote chooser can drive a real file picker. */}
      <input
        ref={fileInputRef}
        className={styles.hiddenInput}
        type="file"
        accept={ATTACH_REQUEST.accept}
        tabIndex={-1}
        aria-hidden="true"
      />
      <DropdownMenu
        items={items}
        align="start"
        minWidth="200px"
        disabled={disabled || choosing || full}
        portalContainer={portalContainer}
        onCloseAutoFocus={(event) => {
          // Radix would move focus back to + and away from the picker.
          if (!pickerOpened.current) return;
          pickerOpened.current = false;
          event.preventDefault();
        }}
      >
        <span>
          <Tooltip
            content={
              full
                ? `You can attach up to ${MAX_ASSISTANT_ATTACHMENTS} files`
                : "Attach a file"
            }
          >
            <Button
              appearance="neutral"
              hierarchy="plain"
              size="md"
              isIconOnly
              leftIcon={<Plus size={20} />}
              aria-label="Attach a file"
              disabled={disabled || choosing || full}
            />
          </Tooltip>
        </span>
      </DropdownMenu>
    </div>
  );
}
