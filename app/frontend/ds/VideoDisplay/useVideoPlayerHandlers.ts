import { useRef } from "react";

interface VideoPlayerEvents {
  playVideo: () => Promise<void>;
  pauseVideo: () => void;
  resetVideo: () => void;
  seekTo: (time: number) => void;
}

export const useVideoPlayerHandlers = () => {
  const videoPlayerRef = useRef<VideoPlayerEvents | null>(null);

  const handleMouseEnter = () => {
    void videoPlayerRef.current?.playVideo();
  };

  const handleMouseLeave = () => {
    videoPlayerRef.current?.pauseVideo();
  };

  const handleMouseLeaveWithVideoReset = () => {
    videoPlayerRef.current?.pauseVideo();
    videoPlayerRef.current?.resetVideo();
  };

  const handleFocus = () => {
    void videoPlayerRef.current?.playVideo();
  };

  const handleBlur = () => {
    videoPlayerRef.current?.pauseVideo();
  };

  const resetVideo = () => {
    videoPlayerRef.current?.resetVideo();
  };

  return {
    videoPlayerRef,
    handlers: {
      handleMouseEnter,
      handleMouseLeave,
      handleMouseLeaveWithVideoReset,
      handleFocus,
      handleBlur,
      resetVideo,
    },
  };
};
