import { clsx } from "clsx";
import { forwardRef } from "react";

import { Text } from "@ds/Text/Text";
import { Flex } from "@ds/layout/Flex/Flex";

import styles from "./Stepper.module.scss";

/**
 * A full-width row of steps, each with a title and an optional helper line. It is not
 * `SlidingSegmentControl` (ltx-io): that control is a compact one-line switch. Its selected
 * label turns dark on a white thumb, so a helper line in a nested `Text` would stay light on
 * white. Use `Stepper` for step headers and `SlidingSegmentControl` for option switches.
 */
export type StepItem = {
  title: string;
  /** Optional helper under the title. Omit for title-only segments. */
  description?: string;
  isDisabled?: boolean;
};

export interface StepperProps {
  items: StepItem[];
  selectedIndex: number;
  isDisabled?: boolean;
  onStepClick?: (index: number) => void;
  className?: string;
}

interface StepperStepItemProps {
  item: StepItem;
  index: number;
  isSelected: boolean;
  isDisabled: boolean;
  onStepClick?: (index: number) => void;
}

function StepperStepItem(props: StepperStepItemProps): JSX.Element {
  const { item, index, isSelected, isDisabled, onStepClick } = props;

  const handleClick = () => {
    if (onStepClick) {
      onStepClick(index);
    }
  };

  return (
    <li className={styles.stepItem}>
      <button
        type="button"
        className={clsx(styles.step, {
          [styles.selected]: isSelected,
          [styles.disabled]: isDisabled,
        })}
        onClick={handleClick}
        disabled={isDisabled}
        aria-current={isSelected ? "step" : undefined}
        aria-disabled={isDisabled ? true : undefined}
        data-state={isDisabled ? "disable" : "enable"}
        data-selected={isSelected}
      >
        <Flex direction="column" align="center" justify="center">
          <Text as="span" variant="heading" size="xs" className={styles.title}>
            {item.title}
          </Text>
          {item.description ? (
            <Text as="span" variant="body" size="md" className={styles.description}>
              {item.description}
            </Text>
          ) : null}
        </Flex>
      </button>
    </li>
  );
}

export const Stepper = forwardRef<HTMLOListElement, StepperProps>(
  function Stepper(props, ref): JSX.Element {
    const { items, selectedIndex, isDisabled, onStepClick, className } = props;

    return (
      <ol
        ref={ref}
        role="list"
        aria-label="Steps"
        className={clsx(styles.container, className)}
      >
        {items.map((item, index) => {
          const isItemDisabled = (isDisabled ?? false) || (item.isDisabled ?? false);

          return (
            <StepperStepItem
              key={`${item.title}-${index}`}
              item={item}
              index={index}
              isSelected={index === selectedIndex}
              isDisabled={isItemDisabled}
              onStepClick={onStepClick}
            />
          );
        })}
      </ol>
    );
  },
);

Stepper.displayName = "Stepper";
