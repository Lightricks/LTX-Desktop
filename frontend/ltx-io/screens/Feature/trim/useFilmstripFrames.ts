import { useEffect, useState } from "react";

import {
  videoFilmstripFrameCount,
  videoFilmstripTimes,
} from "./filmstripLayout.ts";

function waitForSeek(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeout);
      video.removeEventListener("seeked", finish);
      resolve();
    };
    video.addEventListener("seeked", finish);
    const timeout = window.setTimeout(finish, 500);
    video.currentTime = time;
  });
}

function waitForLoadedData(video: HTMLVideoElement): Promise<void> {
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("loadeddata", onReady);
      video.removeEventListener("error", onError);
    };
    const onReady = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("Video filmstrip failed to load."));
    };
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("Video filmstrip load timed out."));
    }, 10_000);
    video.addEventListener("loadeddata", onReady);
    video.addEventListener("error", onError);
  });
}

async function extractFilmstripFrames(
  src: string,
  durationSeconds: number,
  height: number,
  count: number,
  isCancelled: () => boolean,
): Promise<string[]> {
  const video = document.createElement("video");
  video.muted = true;
  video.preload = "auto";
  video.playsInline = true;
  video.src = src;

  try {
    await waitForLoadedData(video);
    if (isCancelled()) return [];

    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    if (context == null || video.videoWidth <= 0 || video.videoHeight <= 0) {
      return [];
    }

    const aspect = video.videoWidth / video.videoHeight;
    canvas.width = Math.max(1, Math.round(height * aspect));
    canvas.height = Math.max(1, Math.round(height));

    const frames: string[] = [];
    for (const time of videoFilmstripTimes(durationSeconds, count)) {
      await waitForSeek(video, time);
      if (isCancelled()) return [];
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      try {
        frames.push(canvas.toDataURL("image/jpeg", 0.6));
      } catch {
        return [];
      }
    }
    return frames;
  } finally {
    video.removeAttribute("src");
    video.load();
  }
}

export function useFilmstripFrames({
  src,
  width,
  height,
  durationSeconds,
}: {
  src: string;
  width: number;
  height: number;
  durationSeconds: number;
}): string[] {
  const [frames, setFrames] = useState<string[]>([]);
  const count = videoFilmstripFrameCount(width);

  useEffect(() => {
    let cancelled = false;
    setFrames([]);
    void extractFilmstripFrames(
      src,
      durationSeconds,
      height,
      count,
      () => cancelled,
    )
      .then((next) => {
        if (!cancelled) setFrames(next);
      })
      .catch(() => {
        if (!cancelled) setFrames([]);
      });
    return () => {
      cancelled = true;
    };
  }, [count, durationSeconds, height, src]);

  return frames;
}
