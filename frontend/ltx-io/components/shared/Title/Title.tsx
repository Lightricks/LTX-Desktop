import { clsx } from "clsx";

import styles from "./Title.module.scss";

export function Title({
  text,
  size,
  buttonGroup,
  className,
}: {
  text: string;
  size: "xxs" | "xs" | "sm" | "md" | "lg" | "xl";
  buttonGroup?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={clsx(
        styles.title,
        !!buttonGroup && styles.withButtonGroup,
        styles[size],
        className,
      )}
    >
      {text}
      {buttonGroup}
    </div>
  );
}
