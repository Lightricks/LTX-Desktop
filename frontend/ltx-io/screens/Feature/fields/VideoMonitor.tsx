import { ActivityCircular } from "@ds/ActivityCircular/ActivityCircular";
import { Button } from "@ds/Button/Button";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import TrimIcon from "@ds/assets/Icons/FullEditor.svg?react";
import MediaReplaceIcon from "@ds/assets/Icons/MediaReplace.svg?react";
import PauseIcon from "@ds/assets/Icons/Pause.svg?react";
import PlayIcon from "@ds/assets/Icons/Play.svg?react";
import RemoveIcon from "@ds/assets/Icons/Remove.svg?react";
import { clsx } from "clsx";
import { type DragEventHandler, type ReactNode, type RefObject } from "react";

import {
  INPUT_PREVIEW_CONTROLS,
  VideoControls,
} from "../../../components/VideoView/VideoControls";
import { useVideoTimeline } from "../../../components/VideoView/useVideoTimeline";
import { useFeatureFormMuteStore } from "../../../stores/featureFormMuteStore";

import styles from "./VideoAssetField.module.scss";

function OverlayButton({
  label,
  ariaLabel,
  icon,
  onClick,
}: {
  label: string;
  ariaLabel: string;
  icon: ReactNode;
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
          aria-label={ariaLabel}
          onClick={(event) => {
            event.stopPropagation();
            onClick();
          }}
        />
      </span>
    </Tooltip>
  );
}

export function VideoMonitor({
  src,
  videoRef,
  isBusy,
  isDragOver,
  cropToFill,
  naturalAspect,
  isPlaying,
  onTogglePlaying,
  onLoadedMetadata,
  onPlay,
  onPause,
  onReplace,
  onTrim,
  onRemove,
  dragHandlers,
}: {
  src: string;
  videoRef: RefObject<HTMLVideoElement>;
  isBusy: boolean;
  isDragOver: boolean;
  cropToFill: boolean;
  naturalAspect: string | null;
  isPlaying: boolean;
  onTogglePlaying: () => void;
  onLoadedMetadata: (video: HTMLVideoElement) => void;
  onPlay: () => void;
  onPause: () => void;
  onReplace: () => void;
  onTrim?: () => void;
  onRemove: () => void;
  dragHandlers: {
    handleDragEnter: DragEventHandler;
    handleDragOver: DragEventHandler;
    handleDragLeave: DragEventHandler;
    handleDrop: DragEventHandler;
  };
}) {
  const isMuted = useFeatureFormMuteStore((state) => state.isMuted);
  const setMuted = useFeatureFormMuteStore((state) => state.setMuted);
  const timeline = useVideoTimeline(videoRef, src);
  const aspectRatio = cropToFill ? (naturalAspect ?? "16 / 9") : "16 / 9";

  return (
    <div
      className={clsx(styles.monitor, isDragOver && styles.dragOver)}
      style={{ aspectRatio }}
      onDragEnter={dragHandlers.handleDragEnter}
      onDragOver={dragHandlers.handleDragOver}
      onDragLeave={dragHandlers.handleDragLeave}
      onDrop={dragHandlers.handleDrop}
    >
      <video
        ref={videoRef}
        className={clsx(styles.monitorVideo, !cropToFill && styles.monitorVideoContain)}
        src={src}
        preload="metadata"
        muted={isMuted}
        playsInline
        loop
        onClick={onTogglePlaying}
        onLoadedMetadata={(event) => onLoadedMetadata(event.currentTarget)}
        onPlay={onPlay}
        onPause={onPause}
      />
      <div className={styles.playOverlay}>
        <button
          type="button"
          className={styles.playButton}
          aria-label={isPlaying ? "Pause" : "Play"}
          onClick={onTogglePlaying}
        >
          {isPlaying ? <PauseIcon /> : <PlayIcon />}
        </button>
      </div>
      <div
        className={clsx(styles.controls, timeline.isDragging && styles.controlsActive)}
        onClick={(event) => event.stopPropagation()}
      >
        <VideoControls
          showElements={INPUT_PREVIEW_CONTROLS}
          currentTime={timeline.currentTime}
          duration={timeline.duration}
          onSeekCallback={timeline.seek}
          onDraggingStarted={timeline.onDraggingStarted}
          onDraggingEnded={timeline.onDraggingEnded}
          onMute={() => setMuted(!isMuted)}
          isDragging={timeline.isDragging}
          isPlaying={isPlaying}
          isMuted={isMuted}
          togglePlaying={onTogglePlaying}
          onFullScreenClick={() => undefined}
          isFullScreen={false}
        />
      </div>
      <div className={styles.overlayActions}>
        <OverlayButton
          label="Replace"
          ariaLabel="Replace video"
          icon={<MediaReplaceIcon />}
          onClick={onReplace}
        />
        {onTrim ? (
          <OverlayButton
            label="Trim"
            ariaLabel="Trim video"
            icon={<TrimIcon />}
            onClick={onTrim}
          />
        ) : null}
        <OverlayButton
          label="Remove"
          ariaLabel="Remove video"
          icon={<RemoveIcon />}
          onClick={onRemove}
        />
      </div>
      {isBusy ? (
        <div className={styles.overlay}>
          <ActivityCircular size={28} />
        </div>
      ) : null}
    </div>
  );
}
