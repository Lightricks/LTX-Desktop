import {
  budgetIsStressed,
  budgetLabel,
  budgetNumbers,
  type BudgetReadout,
} from "../lib/budget-figure";

import styles from "./BudgetFraction.module.scss";

export function BudgetFraction({ readout }: { readout: BudgetReadout }) {
  const numbers = budgetNumbers(readout);
  if (numbers == null) {
    return <span className={styles.plain}>{budgetLabel(readout)}</span>;
  }
  return (
    <span className={styles.badge}>
      <span className={styles.label}>{budgetLabel(readout)}</span>
      <span className={budgetIsStressed(readout) ? styles.need : styles.have}>
        {numbers.need}
      </span>
      <span className={styles.slash}>/</span>
      <span className={styles.have}>{numbers.have}</span>
    </span>
  );
}
