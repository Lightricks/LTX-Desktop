import { aspectRatioToNumber } from "@ds/lib/infinityCommon";
import { PropsWithChildren } from "react";

import styles from "./AspectFill.module.scss";

interface AspectFillProps extends PropsWithChildren {
  ratio: string;
}

export function AspectFill(props: AspectFillProps) {
  const { ratio, children } = props;

  return (
    <div
      style={{
        aspectRatio: aspectRatioToNumber(ratio),
      }}
      className={styles.aspectFill}
    >
      {children}
    </div>
  );
}
