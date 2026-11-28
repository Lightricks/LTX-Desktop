import { clsx } from "clsx";
import { Size } from "@ds/lib/infinityCommon";
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";

import { PlaybackProgress } from "@ds/VideoDisplay/PlaybackProgress";
import { Skeleton } from "@ds/Skeleton/Skeleton";
import { logMessage } from "@ds/lib/handleErrors";

import styles from "./VideoDisplay.module.scss";

export type VideoPreload =
  | "on-demand"
  | "none"
  | "metadata"
  | "eager"
  | "autoplay";

export type VideoConfig = {
  /** The video source URL */
  videoUrl: string;
  /** The video id */
  videoId?: string;
  /** The video title */
  title?: string;
  /** The fallback image url */
  fallbackImageUrl?: string;
  /** Whether the video should loop. Default is true. */
  shouldLoop?: boolean;
  /** Whether the video should mute. Default is true. */
  shouldMute?: boolean;
  /** Whether the video should show controls. Default is false. */
  shouldShowControls?: boolean;
  /** The size of the video if passed, otherwise it will take the size of its parent height.
   * If the full width is preferred do not pass any size and use the fullWidth prop instead.
   */
  size?: Size;
  /** Whether the video should take the full width of its parent rather than the full height */
  fullWidth?: boolean;
  /**
   * Controls video loading and autoplay behavior.
   * - `"on-demand"`: No src set until playVideo() is called. For virtualized lists.
   * - `"none"`: Src set, browser won't preload.
   * - `"metadata"`: Only preload metadata.
   * - `"eager"`: Preload full video.
   * - `"autoplay"`: Preload and play immediately.
   */
  preload: VideoPreload;
};

interface VideoDisplayProps extends VideoConfig {
  /** Additional CSS class names to be applied */
  className?: string;
  /** Callback when video starts playing */
  onPlayStart?: () => void;
  /** Callback when video is paused */
  onPause?: () => void;
  /** Callback when video encounters an error */
  onError?: (error: Error) => void;
  /** Callback when video is loaded */
  onLoadMetadata?: (event: React.SyntheticEvent<HTMLVideoElement>) => void;
  /** Callback when the video starts loading */
  onLoadStart?: (event: React.SyntheticEvent<HTMLVideoElement>) => void;
  /** Callback when the video's first frame is ready to display */
  onLoadedData?: (event: React.SyntheticEvent<HTMLVideoElement>) => void;
  /** Callback when the browser can begin playback. */
  onCanPlay?: (event: React.SyntheticEvent<HTMLVideoElement>) => void;
  /** The current frame rate of the video */
  fps?: number;
  /** Whether to show the progress bar. Defaults to false. */
  showProgress?: boolean;
  /** Callback when the current time of the video changes */
  onTimeChange?: (time: number) => void;
  /** How the video/image fits the container. Defaults to "cover".
   * Use "contain" to show full frame with letterboxing.
   * Use "scale-down" to scale the video/image down to fit the container.
   * Use "fill" to fill the container. */
  objectFit?: "cover" | "contain" | "scale-down" | "fill";
}

export type VideoDisplayRef = {
  playVideo: () => Promise<void>;
  pauseVideo: () => void;
  resetVideo: () => void;
  seekTo: (time: number) => void;
};

/**
 * VideoDisplay component that handles video playback with image fallback
 * Supports automatic fallback to image on error, loading states, and hover controls
 *
 * @component
 * @param props - Component props
 * @param props.videoUrl - The video source URL (WebM preferred for smaller file size)
 * @example
 * ```tsx
 * <VideoDisplay videoUrl="path/to/video.webm" fallbackImageUrl="path/to/image.avif" preload="eager" />
 * ```
 */
export const VideoDisplay = forwardRef<VideoDisplayRef, VideoDisplayProps>(
  (
    {
      videoUrl,
      fallbackImageUrl,
      videoId,
      title,
      shouldLoop = true,
      shouldMute = true,
      shouldShowControls = false,
      preload,
      className,
      size,
      fullWidth,
      onPlayStart,
      onPause,
      onError,
      onLoadMetadata,
      onLoadStart,
      onLoadedData,
      onCanPlay,
      fps,
      showProgress = false,
      onTimeChange,
      objectFit = "cover",
    },
    ref,
  ): JSX.Element => {
    let isLazy: boolean;
    let shouldAutoPlay: boolean;
    let htmlPreload: HTMLVideoElement["preload"];

    switch (preload) {
      case "on-demand":
        isLazy = true;
        shouldAutoPlay = false;
        htmlPreload = "none";
        break;
      case "autoplay":
        isLazy = false;
        shouldAutoPlay = true;
        htmlPreload = "auto";
        break;
      case "eager":
        isLazy = false;
        shouldAutoPlay = false;
        htmlPreload = "auto";
        break;
      case "none":
      case "metadata":
        isLazy = false;
        shouldAutoPlay = false;
        htmlPreload = preload;
        break;
    }

    const videoRef = useRef<HTMLVideoElement>(null);
    const lazySrcAssignedRef = useRef(false);
    const [hasVideoError, setHasVideoError] = useState(false);
    const [isVideoLoaded, setIsVideoLoaded] = useState(false);
    const [loadedFallbackImageUrl, setLoadedFallbackImageUrl] = useState<
      string | null
    >(null);
    const isFallbackImageLoaded =
      fallbackImageUrl !== undefined &&
      loadedFallbackImageUrl === fallbackImageUrl;

    const { mediaStyle, lazyMediaStyle } = useMemo(() => {
      const width = fullWidth ? "100%" : (size?.width ?? "auto");
      const height = fullWidth ? (size?.height ?? "100%") : "100%";

      // Only calculate aspect ratio if both dimensions are numbers
      const aspectRatio = size ? size.width / size.height : undefined;
      const base = { width, height, aspectRatio };

      return {
        mediaStyle: base,
        lazyMediaStyle: isLazy
          ? { ...base, transition: "none" as const }
          : base,
      };
    }, [fullWidth, size, isLazy]);

    const handleError = useCallback(
      (errorMessage: string, sourceInfo?: { videoUrl?: string }) => {
        const error = new Error(errorMessage);
        setHasVideoError(true);
        logMessage(
          `Video error, source: ${JSON.stringify(sourceInfo)}`,
          "error",
        );
        onError?.(error);
      },
      [onError],
    );

    useImperativeHandle(ref, () => ({
      playVideo: async () => {
        if (videoRef.current && !hasVideoError) {
          try {
            if (isLazy && videoUrl && !lazySrcAssignedRef.current) {
              videoRef.current.src = videoUrl;
              lazySrcAssignedRef.current = true;
            }

            const playPromise = videoRef.current.play();
            if (playPromise !== undefined) {
              await playPromise;
            }

            setIsVideoLoaded(true);
          } catch (error) {
            if (
              error instanceof DOMException &&
              error.name === "NotAllowedError"
            ) {
              console.warn(
                "Video playback was prevented by browser autoplay restrictions",
              );
            } else if (isPlaybackInterruptionError(error)) {
              console.warn(
                "Video playback was interrupted, this is expected in some cases",
              );
              return;
            } else {
              setHasVideoError(true);
              onError?.(error as Error);
            }
          }
        }
      },
      pauseVideo: () => {
        if (videoRef.current) {
          videoRef.current.pause();
        }
      },
      resetVideo: () => {
        if (videoRef.current) {
          videoRef.current.pause();
          videoRef.current.currentTime = 0;
        }
        setIsVideoLoaded(false);
      },
      seekTo: (time: number) => {
        if (videoRef.current && Number.isFinite(time)) {
          videoRef.current.currentTime = Math.max(0, time);
        }
      },
    }));

    useEffect(() => {
      const video = videoRef.current;
      if (!video) {
        return;
      }

      const handleLoadedData = () => {
        setIsVideoLoaded(true);
      };

      const handlePlay = () => {
        onPlayStart?.();
      };

      const handlePause = () => {
        onPause?.();
      };

      video.addEventListener("loadeddata", handleLoadedData);
      video.addEventListener("play", handlePlay);
      video.addEventListener("pause", handlePause);

      return () => {
        video.removeEventListener("loadeddata", handleLoadedData);
        video.removeEventListener("play", handlePlay);
        video.removeEventListener("pause", handlePause);
      };
    }, [onPlayStart, onPause]);

    useEffect(() => {
      const element = videoRef.current;
      if (!element || !onTimeChange) {
        return;
      }

      const handleTimeUpdate = () => {
        onTimeChange?.(element.currentTime);
      };

      element.addEventListener("timeupdate", handleTimeUpdate);
      return () => {
        element.removeEventListener("timeupdate", handleTimeUpdate);
      };
    }, [onTimeChange]);

    useEffect(() => {
      setIsVideoLoaded(false);
      setHasVideoError(false);
      lazySrcAssignedRef.current = false;

      if (isLazy) {
        if (videoRef.current) {
          videoRef.current.pause();
          videoRef.current.removeAttribute("src");
          videoRef.current.load(); // cancels in-flight network requests
        }
      }
    }, [videoUrl, isLazy]);

    const markFallbackImageLoaded = useCallback(() => {
      if (fallbackImageUrl) {
        setLoadedFallbackImageUrl(fallbackImageUrl);
      }
    }, [fallbackImageUrl]);

    const showFallbackSkeleton =
      Boolean(fallbackImageUrl) && !isFallbackImageLoaded;
    const isVideoVisible = isVideoLoaded && !hasVideoError;

    return (
      <div
        className={clsx(styles.mediaContainer, className)}
        data-testid="media-player"
        style={mediaStyle}
      >
        <div className={styles.media}>
          {showFallbackSkeleton && (
            <Skeleton className={styles.fallbackSkeleton} />
          )}
          {fallbackImageUrl && (
            <img
              data-testid="fallback-image"
              src={fallbackImageUrl}
              alt={title || "Video fallback"}
              className={clsx(styles.fallbackImage, {
                [styles.fallbackImageHidden]: isVideoVisible,
                [styles.objectFitContain]: objectFit === "contain",
                [styles.objectFitScaleDown]: objectFit === "scale-down",
                [styles.objectFitCover]: objectFit === "cover",
                [styles.objectFitFill]: objectFit === "fill",
              })}
              draggable={false}
              style={lazyMediaStyle}
              onLoad={markFallbackImageLoaded}
              onError={markFallbackImageLoaded}
              ref={(img) => {
                // Cached images can complete before onLoad is attached.
                if (img?.complete && img.naturalWidth > 0) {
                  markFallbackImageLoaded();
                }
              }}
            />
          )}
          <video
            data-testid="video"
            ref={videoRef}
            className={clsx(styles.video, {
              [styles.videoLoaded]: isVideoVisible,
              [styles.objectFitContain]: objectFit === "contain",
              [styles.objectFitScaleDown]: objectFit === "scale-down",
              [styles.objectFitCover]: objectFit === "cover",
              [styles.objectFitFill]: objectFit === "fill",
            })}
            autoPlay={shouldAutoPlay}
            muted={shouldMute}
            loop={shouldLoop}
            controls={shouldShowControls}
            playsInline
            preload={htmlPreload}
            onLoadedMetadata={onLoadMetadata}
            onLoadStart={onLoadStart}
            onLoadedData={onLoadedData}
            onCanPlay={onCanPlay}
            id={videoId}
            title={title}
            style={lazyMediaStyle}
            onError={(e) => {
              const videoError = e.currentTarget.error;
              handleError(
                videoError
                  ? `Video failed to load: ${videoError.message || `Code: ${videoError.code}`}`
                  : "Video failed to load",
                { videoUrl },
              );
            }}
            src={isLazy || !videoUrl ? undefined : videoUrl}
          />
          <PlaybackProgress
            videoRef={videoRef}
            fps={fps}
            isVisible={showProgress}
          />
        </div>
      </div>
    );
  },
);

VideoDisplay.displayName = "VideoDisplay";

function isPlaybackInterruptionError(error: unknown): boolean {
  // Check for AbortError DOMException
  if (error instanceof DOMException && error.name === "AbortError") {
    return true;
  }

  // Check for common error message patterns
  if (error instanceof Error) {
    const message = (error.message || "").toLowerCase();
    return (
      message.includes("interrupt") ||
      message.includes("abort") ||
      message.includes("cancel")
    );
  }

  return false;
}
