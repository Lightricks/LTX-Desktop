import { type MotionPathEasing, applyMotionPathEasing } from "../../../lib/motionPathSampler";

import styles from "./MotionPathEasingCurveIcon.module.scss";

const SIZE = 28;
const PAD = 4;
const SAMPLES = 24;

/** Small dotted-grid + curve icon for an easing preset (Figma-style). */
export function MotionPathEasingCurveIcon({ easing }: { easing: MotionPathEasing }) {
  const plotW = SIZE - PAD * 2;
  const plotH = SIZE - PAD * 2;

  const points: Array<{ x: number; y: number }> = [];
  let minY = 0;
  let maxY = 1;
  for (let i = 0; i <= SAMPLES; i++) {
    const t = i / SAMPLES;
    const y = applyMotionPathEasing(t, easing);
    points.push({ x: t, y });
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const ySpan = Math.max(1e-6, maxY - minY);

  const toSvg = (p: { x: number; y: number }) => {
    const sx = PAD + p.x * plotW;
    const sy = PAD + plotH - ((p.y - minY) / ySpan) * plotH;
    return `${sx.toFixed(2)},${sy.toFixed(2)}`;
  };

  const d = points.map((p, i) => `${i === 0 ? "M" : "L"}${toSvg(p)}`).join(" ");

  const dots: Array<{ cx: number; cy: number }> = [];
  for (let gx = 0; gx < 4; gx++) {
    for (let gy = 0; gy < 4; gy++) {
      dots.push({
        cx: PAD + (gx / 3) * plotW,
        cy: PAD + (gy / 3) * plotH,
      });
    }
  }

  return (
    <svg
      className={styles.icon}
      width={SIZE}
      height={SIZE}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      aria-hidden
    >
      {dots.map((dot, i) => (
        <circle key={i} cx={dot.cx} cy={dot.cy} r={0.9} className={styles.dot} />
      ))}
      <path d={d} className={styles.curve} fill="none" />
    </svg>
  );
}
