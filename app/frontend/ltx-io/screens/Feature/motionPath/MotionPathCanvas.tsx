import {
  type MotionPathPoint,
  type MotionPathVec2,
  absoluteHandle,
  motionPathDotColor,
} from "../../../lib/motionPathSampler";
import { useCallback, useEffect, useRef, useState } from "react";

import { Text } from "@ds/Text/Text";

import styles from "./MotionPathCanvas.module.scss";
import {
  drawHandleArm,
  drawMotionPathDots,
  drawMotionPathStroke,
  resizeCanvasToContainer,
  sampleEditorMotionPathFrames,
} from "./motionPathCanvasUtils";
import { type HandleHit } from "./motionPathHitTest";
import {
  MOTION_PATH_MODAL_PREVIEW_MAX_HEIGHT,
  getMotionPathPreviewFrameStyle,
} from "./motionPathPreviewFrameStyle";
import { useMotionPathPointerHandlers } from "./useMotionPathPointerHandlers";

const PATH_SAMPLE_STEPS = 32;
const OTHER_TRACK_ALPHA = 0.4;
const NO_TRACKS: MotionPathCanvasTrack[] = [];

export type MotionPathTool = "pen" | "erase" | "convert" | "freehand";

/** A track the canvas draws but does not edit. `colorIndex` picks its dot color. */
export type MotionPathCanvasTrack = { points: MotionPathPoint[]; colorIndex: number };

export type MotionPathCanvasProps = {
  imageUrl?: string | null;
  /** The frame the pipeline renders for the image, or null until the image loads. */
  imageAspect: number | null;
  points: MotionPathPoint[];
  /** The dot color index of the track being edited. */
  colorIndex: number;
  /** The other tracks. The canvas draws them dimmed and does not edit them. */
  otherTracks?: MotionPathCanvasTrack[];
  tool?: MotionPathTool;
  onPointsChange: (points: MotionPathPoint[], historyMode?: "push" | "replace") => void;
};

export function MotionPathCanvas({
  imageUrl = null,
  imageAspect,
  points,
  colorIndex,
  otherTracks = NO_TRACKS,
  tool = "freehand",
  onPointsChange,
}: MotionPathCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const [selectedPointId, setSelectedPointId] = useState<string | null>(null);
  const [freehandStroke, setFreehandStroke] = useState<MotionPathVec2[]>([]);
  const activeSegmentRef = useRef<number | null>(null);
  const activeHandleRef = useRef<HandleHit | null>(null);
  useEffect(() => {
    if (!selectedPointId) return;
    if (!points.some((point) => point.id === selectedPointId)) {
      setSelectedPointId(null);
    }
  }, [points, selectedPointId]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const frame = frameRef.current;
    if (!canvas || !frame) return;

    const { width, height } = resizeCanvasToContainer(canvas, frame);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    ctx.globalAlpha = OTHER_TRACK_ALPHA;
    for (const { points: trackPoints, colorIndex: trackColorIndex } of otherTracks) {
      if (trackPoints.length >= 2) {
        drawMotionPathStroke(
          ctx,
          sampleEditorMotionPathFrames(trackPoints, width, height, PATH_SAMPLE_STEPS),
          { strokeStyle: "rgba(255, 255, 255, 0.9)" },
        );
      }
      drawMotionPathDots(ctx, trackPoints, {
        width,
        height,
        dotColor: motionPathDotColor(trackColorIndex),
      });
    }
    ctx.globalAlpha = 1;

    if (freehandStroke.length >= 2) {
      ctx.beginPath();
      ctx.moveTo(freehandStroke[0].x * width, freehandStroke[0].y * height);
      for (let i = 1; i < freehandStroke.length; i++) {
        ctx.lineTo(freehandStroke[i].x * width, freehandStroke[i].y * height);
      }
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.lineWidth = 2.5;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.stroke();
    }

    if (points.length >= 2) {
      const allFrames = sampleEditorMotionPathFrames(
        points,
        width,
        height,
        PATH_SAMPLE_STEPS,
      );

      for (let segmentIndex = 0; segmentIndex < points.length - 1; segmentIndex++) {
        const startFrame = segmentIndex * PATH_SAMPLE_STEPS;
        const endFrame = startFrame + PATH_SAMPLE_STEPS;
        const isActive = activeSegmentRef.current === segmentIndex;

        ctx.beginPath();
        ctx.moveTo(allFrames[startFrame].x, allFrames[startFrame].y);
        for (let i = startFrame + 1; i <= endFrame; i++) {
          ctx.lineTo(allFrames[i].x, allFrames[i].y);
        }
        ctx.strokeStyle = isActive
          ? "rgba(255, 255, 255, 0.95)"
          : "rgba(255, 255, 255, 0.55)";
        ctx.lineWidth = isActive ? 3 : 2;
        ctx.stroke();
      }
    }

    const selectedIndex = selectedPointId
      ? points.findIndex((point) => point.id === selectedPointId)
      : -1;
    if (selectedIndex >= 0) {
      const anchor = points[selectedIndex];
      const ax = anchor.x * width;
      const ay = anchor.y * height;
      const active = activeHandleRef.current;

      const outTip = absoluteHandle(anchor, anchor.handleOut);
      if (outTip) {
        drawHandleArm(
          ctx,
          ax,
          ay,
          outTip.x * width,
          outTip.y * height,
          active?.pointIndex === selectedIndex && active.side === "out",
        );
      }

      const inTip = absoluteHandle(anchor, anchor.handleIn);
      if (inTip) {
        drawHandleArm(
          ctx,
          ax,
          ay,
          inTip.x * width,
          inTip.y * height,
          active?.pointIndex === selectedIndex && active.side === "in",
        );
      }
    }

    drawMotionPathDots(ctx, points, {
      width,
      height,
      dotColor: motionPathDotColor(colorIndex),
      selectedPointId,
      selectedRadiusDelta: 1,
    });
  }, [freehandStroke, points, colorIndex, otherTracks, selectedPointId]);

  const { handlePointerDown, handlePointerMove, handlePointerUp, handleDoubleClick } =
    useMotionPathPointerHandlers({
      canvasRef,
      points,
      tool,
      onPointsChange,
      selectedPointId,
      setSelectedPointId,
      setFreehandStroke,
      activeSegmentRef,
      activeHandleRef,
      draw,
    });

  useEffect(() => {
    draw();
  }, [draw, imageAspect]);

  useEffect(() => {
    const frame = frameRef.current;
    if (!frame) return;

    const observer = new ResizeObserver(() => draw());
    observer.observe(frame);
    return () => observer.disconnect();
  }, [draw]);

  const frameStyle =
    imageAspect !== null
      ? getMotionPathPreviewFrameStyle(imageAspect, MOTION_PATH_MODAL_PREVIEW_MAX_HEIGHT)
      : undefined;

  return (
    <div className={styles.root} data-testid="motion-path-canvas">
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
        ) : (
          <div className={styles.empty}>
            <Text as="p" variant="body" size="md">
              Upload a reference image to place path points
            </Text>
          </div>
        )}
        {imageUrl ? (
          <canvas
            ref={canvasRef}
            className={styles.canvas}
            data-tool={tool}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
            onDoubleClick={handleDoubleClick}
          />
        ) : null}
      </div>
    </div>
  );
}
