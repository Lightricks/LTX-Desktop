import { clsx } from "clsx";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  VideoDisplay,
  type VideoDisplayRef,
} from "@ds/VideoDisplay/VideoDisplay";
import { VideoMuteButton } from "@ds/VideoMuteButton/VideoMuteButton";
import VideoIcon from "@ds/assets/Icons/VideoCamera.svg?react";

import type { ExploreListedAsset } from "@/lib/explore-contract";
import { isAlphaWebmMime } from "@/ltx-io/screens/Feature/results/cutout/cutoutOutput";
import { useFeatureFormMuteStore } from "@/ltx-io/stores/featureFormMuteStore";

import { AssetVideoMetaBadge } from "./AssetVideoMetaBadge";
import {
  activateHoverVideo,
  deactivateHoverVideo,
} from "./hoverVideoCoordinator";
import backdropStyles from "./cutoutBackdrop.module.scss";
import styles from "./MediaTile.module.scss";

export function VideoPlaceholder({ kind }: { kind: "video" | "missing" }) {
  return (
    <div
      className={styles.placeholder}
      aria-label={kind === "video" ? "Video preview" : "Media unavailable"}
    >
      <VideoIcon />
    </div>
  );
}

export function AssetVideoTile({
  asset,
  mediaUrl,
  thumbnailUrl,
  shouldHoverPlay,
  isHovered,
}: {
  asset: ExploreListedAsset;
  mediaUrl: string | null;
  thumbnailUrl: string | null;
  shouldHoverPlay: boolean;
  isHovered: boolean;
}) {
  const videoRef = useRef<VideoDisplayRef>(null);
  const isHoverPlayingRef = useRef(false);
  const [canPlay, setCanPlay] = useState(false);
  const hasVideo = mediaUrl !== null;
  const isMuted = useFeatureFormMuteStore((state) => state.isMuted);
  const setMuted = useFeatureFormMuteStore((state) => state.setMuted);

  useEffect(() => {
    return () => {
      deactivateHoverVideo(asset.id);
    };
  }, [asset.id]);

  const stopVideo = useCallback(() => {
    isHoverPlayingRef.current = false;
    videoRef.current?.resetVideo();
    setCanPlay(false);
    deactivateHoverVideo(asset.id);
  }, [asset.id]);

  const playVideo = useCallback(() => {
    if (!shouldHoverPlay || !videoRef.current) return;
    isHoverPlayingRef.current = true;
    activateHoverVideo(asset.id, stopVideo);
    void videoRef.current.playVideo().then(() => {
      if (isHoverPlayingRef.current) setCanPlay(true);
    });
  }, [asset.id, shouldHoverPlay, stopVideo]);

  useEffect(() => {
    if (isHovered && shouldHoverPlay) {
      playVideo();
      return;
    }
    stopVideo();
  }, [isHovered, playVideo, shouldHoverPlay, stopVideo]);

  if (!hasVideo) return <VideoPlaceholder kind="missing" />;

  return (
    <div
      className={clsx(
        styles.video,
        isAlphaWebmMime(asset.mime_type) && backdropStyles.cutoutBackdrop,
      )}
    >
      <VideoDisplay
        ref={videoRef}
        videoUrl={mediaUrl}
        fallbackImageUrl={thumbnailUrl ?? undefined}
        title={asset.name}
        preload="none"
        shouldMute={isMuted}
        shouldLoop
        shouldShowControls={false}
        fullWidth
        onPlayStart={() => {
          if (isHoverPlayingRef.current) setCanPlay(true);
        }}
        onCanPlay={() => {
          if (isHoverPlayingRef.current) setCanPlay(true);
        }}
      />
      {thumbnailUrl && !canPlay ? (
        <img
          className={styles.hoverPoster}
          src={thumbnailUrl}
          alt=""
          draggable={false}
        />
      ) : null}
      {!canPlay && !thumbnailUrl ? <VideoPlaceholder kind="video" /> : null}
      <AssetVideoMetaBadge metadata={asset.metadata} />
      <div
        className={styles.mute}
        onClick={(event) => event.stopPropagation()}
      >
        <VideoMuteButton
          isMuted={isMuted}
          onToggle={() => setMuted(!isMuted)}
        />
      </div>
    </div>
  );
}
