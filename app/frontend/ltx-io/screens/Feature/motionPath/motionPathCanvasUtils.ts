import {
  MOTION_PATH_DEFAULT_EASING,
  type MotionPathPoint,
  sampleMotionPathFrames,
} from "../../../lib/motionPathSampler";

import {
  EDITOR_DOT_RADIUS_PX,
  EDITOR_HANDLE_RADIUS_PX,
} from "./motionPathEditorConstants";

/** Set backing-store size to CSS size × DPR; ctx draws in logical (CSS) pixels. */
export function resizeCanvasToContainer(
  canvas: HTMLCanvasElement,
  container: HTMLElement,
): { width: number; height: number } {
  const rect = container.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const dpr = window.devicePixelRatio || 1;
  const backingWidth = Math.round(width * dpr);
  const backingHeight = Math.round(height * dpr);

  if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
    canvas.width = backingWidth;
    canvas.height = backingHeight;
  }

  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  return { width, height };
}

export type DrawMotionPathStrokeOptions = {
  offsetX?: number;
  offsetY?: number;
  strokeStyle?: string;
  lineWidth?: number;
  lineCap?: CanvasLineCap;
  lineJoin?: CanvasLineJoin;
};

/** Draw a sampled path polyline in logical pixel space. */
export function drawMotionPathStroke(
  ctx: CanvasRenderingContext2D,
  frames: { x: number; y: number }[],
  options: DrawMotionPathStrokeOptions = {},
): void {
  if (frames.length < 2) return;
  const {
    offsetX = 0,
    offsetY = 0,
    strokeStyle = "rgba(255, 255, 255, 0.55)",
    lineWidth = 2,
    lineCap = "round",
    lineJoin = "round",
  } = options;

  ctx.beginPath();
  ctx.moveTo(offsetX + frames[0].x, offsetY + frames[0].y);
  for (let i = 1; i < frames.length; i++) {
    ctx.lineTo(offsetX + frames[i].x, offsetY + frames[i].y);
  }
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = lineCap;
  ctx.lineJoin = lineJoin;
  ctx.stroke();
}

export type DrawMotionPathDotsOptions = {
  offsetX?: number;
  offsetY?: number;
  width: number;
  height: number;
  dotColor: string;
  selectedPointId?: string | null;
  radius?: number;
  selectedRadiusDelta?: number;
};

/** Draw anchor dots for motion-path points in logical pixel space. */
export function drawMotionPathDots(
  ctx: CanvasRenderingContext2D,
  points: MotionPathPoint[],
  options: DrawMotionPathDotsOptions,
): void {
  const {
    offsetX = 0,
    offsetY = 0,
    width,
    height,
    dotColor,
    selectedPointId = null,
    radius = EDITOR_DOT_RADIUS_PX,
    selectedRadiusDelta = 0,
  } = options;

  for (const point of points) {
    const isSelected = point.id === selectedPointId;
    ctx.beginPath();
    ctx.arc(
      offsetX + point.x * width,
      offsetY + point.y * height,
      radius + (isSelected ? selectedRadiusDelta : 0),
      0,
      Math.PI * 2,
    );
    ctx.fillStyle = dotColor;
    ctx.fill();
    ctx.strokeStyle = isSelected ? "rgba(255, 255, 255, 1)" : "rgba(255, 255, 255, 0.85)";
    ctx.lineWidth = isSelected ? 2 : 1;
    ctx.stroke();
  }
}

/** Sample per-segment frames for editor-style path strokes. */
export function sampleEditorMotionPathFrames(
  points: MotionPathPoint[],
  width: number,
  height: number,
  pathSampleSteps: number,
): { x: number; y: number }[] {
  return sampleMotionPathFrames(
    { points, easing: MOTION_PATH_DEFAULT_EASING, startHoldSec: 0, endHoldSec: 0 },
    1,
    (points.length - 1) * pathSampleSteps + 1,
  ).map(({ x, y }) => ({ x: x * width, y: y * height }));
}

/** Draw a Bézier handle arm (dashed line + tip dot) in logical pixel space. */
export function drawHandleArm(
  ctx: CanvasRenderingContext2D,
  fromX: number,
  fromY: number,
  handleX: number,
  handleY: number,
  active: boolean,
): void {
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(handleX, handleY);
  ctx.strokeStyle = active ? "rgba(255, 255, 255, 0.9)" : "rgba(255, 255, 255, 0.55)";
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.beginPath();
  ctx.arc(handleX, handleY, EDITOR_HANDLE_RADIUS_PX, 0, Math.PI * 2);
  ctx.fillStyle = active ? "rgba(255, 255, 255, 0.95)" : "rgba(255, 255, 255, 0.85)";
  ctx.fill();
  ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
  ctx.lineWidth = 1;
  ctx.stroke();
}

export type CanvasPointerLocal = {
  px: number;
  py: number;
  width: number;
  height: number;
};

/** Map a pointer event to canvas-local logical pixels. */
export function getCanvasPointerLocal(
  canvas: HTMLCanvasElement,
  clientX: number,
  clientY: number,
): CanvasPointerLocal | null {
  const rect = canvas.getBoundingClientRect();
  return {
    px: clientX - rect.left,
    py: clientY - rect.top,
    width: rect.width,
    height: rect.height,
  };
}
