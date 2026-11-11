import styles from "./DragGrip.module.scss";

/** The six dots of a drag handle. It takes the color of the text around it. */
export function DragGrip() {
  return <span className={styles.grip} aria-hidden />;
}
