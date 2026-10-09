import { useFilmstripFrames } from "./useFilmstripFrames.ts";
import styles from "./VideoFilmstrip.module.scss";

export function VideoFilmstrip({
  src,
  width,
  height,
  durationSeconds,
}: {
  src: string;
  width: number;
  height: number;
  durationSeconds: number;
}) {
  const frames = useFilmstripFrames({ src, width, height, durationSeconds });

  return (
    <div className={styles.strip} style={{ width, height }}>
      {frames.map((frame, index) => (
        <img key={index} className={styles.frame} src={frame} alt="" />
      ))}
    </div>
  );
}
