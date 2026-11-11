import { clsx } from "clsx";

import CheckIcon from "@ds/assets/Icons/Accept.svg?react";

import type { MemoryFitsModel } from "@/lib/memory-meter";

import styles from "./MemoryMeter.module.scss";

/** No number yet: the sizes that fit, as pills, and one line for the selected size. No bar. */
export function MemoryFits({ model }: { model: MemoryFitsModel }) {
  return (
    <div className={styles.meter} role="group" aria-label={model.title}>
      <div className={styles.header}>
        <span className={styles.title}>{model.title}</span>
      </div>
      <div className={styles.pills}>
        {model.cells.map((cell) => (
          <span
            key={cell.value}
            className={clsx(
              styles.pill,
              cell.state === "full" ? styles.pillFull : styles.pillShort,
              cell.state === "none" && styles.pillNone,
              cell.selected && styles.pillSelected,
            )}
          >
            {cell.state === "full" ? <CheckIcon className={styles.pillCheck} aria-hidden /> : null}
            {cell.label}
            {cell.note ? <span className={styles.pillNote}>{cell.note}</span> : null}
          </span>
        ))}
      </div>
      <span className={clsx(styles.line, model.line.tone === "over" && styles.lineOver)}>
        {model.line.text}
      </span>
    </div>
  );
}
