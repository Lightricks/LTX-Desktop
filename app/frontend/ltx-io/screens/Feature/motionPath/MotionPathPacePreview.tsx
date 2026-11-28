import {
  MOTION_PATH_PREVIEW_FPS,
  type MotionPath,
  motionPathDotColor,
  sampleMotionPathFrames,
} from "../../../lib/motionPathSampler";
import { useEffect, useRef } from "react";

import styles from "./MotionPathPacePreview.module.scss";
import {
  drawMotionPathStroke,
  resizeCanvasToContainer,
  sampleEditorMotionPathFrames,
} from "./motionPathCanvasUtils";
import { EDITOR_DOT_RADIUS_PX } from "./motionPathEditorConstants";
import {
  MOTION_PATH_MODAL_PREVIEW_MAX_HEIGHT,
  getMotionPathPreviewFrameStyle,
} from "./motionPathPreviewFrameStyle";

export type MotionPathPacePreviewProps = {
  imageUrl?: string;
  /** The frame the pipeline renders for the image, or null until the image loads. */
  imageAspect: number | null;
  /** Every track, each with its own pace. They share the clip length. */
  tracks: MotionPath[];
  durationSec: number;
};

/**
 * Live loop of the colored guide dot moving along the path with the
 * selected timing — same sampler the control-video render uses.
 */
export function MotionPathPacePreview({
  imageUrl,
  imageAspect,
  tracks,
  durationSec,
}: MotionPathPacePreviewProps) {
  const frameRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameIndexRef = useRef(0);
  const rafRef = useRef(0);
  const startedAtRef = useRef(0);
  useEffect(() => {
    const canvas = canvasRef.current;
    const frame = frameRef.current;
    if (!canvas || !frame || tracks.length === 0 || tracks.some((track) => track.points.length < 2)) {
      return;
    }

    const frameCount = Math.max(2, Math.round(durationSec * MOTION_PATH_PREVIEW_FPS));
    const unitTracks = tracks.map((track) =>
      sampleMotionPathFrames(track, durationSec, frameCount),
    );
    let width = 0;
    let height = 0;
    let trackFrames = unitTracks;

    const resizeAndResample = () => {
      ({ width, height } = resizeCanvasToContainer(canvas, frame));
      trackFrames = unitTracks.map((unitFrames) =>
        unitFrames.map(({ x, y }) => ({ x: x * width, y: y * height })),
      );
    };

    resizeAndResample();
    startedAtRef.current = performance.now();

    const drawPaths = (ctx: CanvasRenderingContext2D) => {
      for (const { points } of tracks) {
        drawMotionPathStroke(ctx, sampleEditorMotionPathFrames(points, width, height, 24));
      }
    };

    const tick = (now: number) => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const rect = frame.getBoundingClientRect();
      const nextW = Math.max(1, Math.round(rect.width));
      const nextH = Math.max(1, Math.round(rect.height));
      if (nextW !== width || nextH !== height) {
        resizeAndResample();
      }

      const elapsedSec = (now - startedAtRef.current) / 1000;
      const loopT = ((elapsedSec % durationSec) + durationSec) % durationSec;
      const index = Math.min(
        frameCount - 1,
        Math.floor((loopT / durationSec) * frameCount),
      );
      frameIndexRef.current = index;

      ctx.clearRect(0, 0, width, height);
      drawPaths(ctx);

      for (const [trackIndex, frames] of trackFrames.entries()) {
        const sample = frames[index];
        ctx.beginPath();
        ctx.arc(sample.x, sample.y, EDITOR_DOT_RADIUS_PX + 1, 0, Math.PI * 2);
        ctx.fillStyle = motionPathDotColor(trackIndex);
        ctx.fill();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    const observer = new ResizeObserver(() => resizeAndResample());
    observer.observe(frame);

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(rafRef.current);
      observer.disconnect();
    };
  }, [tracks, durationSec, imageUrl, imageAspect]);

  const frameStyle =
    imageAspect !== null
      ? getMotionPathPreviewFrameStyle(imageAspect, MOTION_PATH_MODAL_PREVIEW_MAX_HEIGHT)
      : undefined;

  return (
    <div className={styles.root} data-testid="motion-path-pace-preview">
      <div
        ref={frameRef}
        className={styles.frame}
        style={frameStyle}
        data-empty={imageAspect === null ? true : undefined}
      >
        {imageUrl ? (
          <img
            src={imageUrl}
            alt=""
            className={styles.image}
            draggable={false}
          />
        ) : null}
        <canvas ref={canvasRef} className={styles.canvas} />
      </div>
    </div>
  );
}
