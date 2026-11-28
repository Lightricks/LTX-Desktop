import {
  type DragEventHandler,
  type RefObject,
  useCallback,
  useRef,
  useState,
} from "react";

import { VideoMonitor } from "./VideoMonitor";
import { VideoTrimSection } from "./VideoTrimSection";

export type VideoAssetRangeValue = {
  startSec: number;
  durationSec: number;
  minDurationSec: number;
  maxDurationSec: number;
  label: string;
  onChange: (startSec: number, durationSec: number) => void;
};

export function VideoAssetPreview({
  src,
  isBusy,
  isDragOver,
  durationSeconds,
  range,
  onReplace,
  onTrim,
  onRemove,
  dragHandlers,
}: {
  src: string;
  isBusy: boolean;
  isDragOver: boolean;
  durationSeconds: number | null;
  range?: VideoAssetRangeValue;
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
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [naturalAspect, setNaturalAspect] = useState<string | null>(null);

  const toggleIsPlaying = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      void video.play().catch(() => setIsPlaying(false));
      return;
    }
    video.pause();
  }, []);
  const monitorRef = videoRef as RefObject<HTMLVideoElement>;

  return (
    <>
      <VideoMonitor
        src={src}
        videoRef={monitorRef}
        isBusy={isBusy}
        isDragOver={isDragOver}
        cropToFill={range != null}
        naturalAspect={naturalAspect}
        isPlaying={isPlaying}
        onTogglePlaying={toggleIsPlaying}
        onLoadedMetadata={(video) => {
          if (video.videoWidth > 0 && video.videoHeight > 0) {
            setNaturalAspect(`${video.videoWidth} / ${video.videoHeight}`);
          }
        }}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onReplace={onReplace}
        onTrim={
          onTrim
            ? () => {
                videoRef.current?.pause();
                onTrim();
              }
            : undefined
        }
        onRemove={onRemove}
        dragHandlers={dragHandlers}
      />
      {range != null && durationSeconds != null && durationSeconds > 0 ? (
        <VideoTrimSection
          src={src}
          durationSeconds={durationSeconds}
          range={range}
          videoRef={monitorRef}
          isPlaying={isPlaying}
        />
      ) : null}
    </>
  );
}
