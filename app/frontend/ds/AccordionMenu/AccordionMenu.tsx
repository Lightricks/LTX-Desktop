import * as RadixAccordion from "@radix-ui/react-accordion";
import { clsx } from "clsx";

import ArrowLarge from "@ds/assets/Icons/Arrow/Down/Large.svg?react";
import ArrowSmall from "@ds/assets/Icons/Arrow/Down/Small.svg?react";

import styles from "./AccordionMenu.module.scss";

type AccordionMenuProps = {
  trigger: React.ReactNode;
  children: React.ReactNode;
  initialExpand?: boolean;
  /** When set, the accordion is controlled. Prefer this over remounting. */
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
  appearance?: "title";
  headerClassName?: string;
  contentClassName?: string;
};

/** Arrow indicator that rotates when accordion is open. Use inside your trigger. */
export function AccordionArrow({ size = "small" }: { size?: "small" | "large" }) {
  return (
    <div className={styles.arrow}>
      {size === "large" ? <ArrowLarge /> : <ArrowSmall />}
    </div>
  );
}

export function AccordionMenu({
  trigger,
  children,
  initialExpand = false,
  expanded,
  onExpandedChange,
  appearance,
  headerClassName,
  contentClassName,
}: AccordionMenuProps) {
  const controlled = expanded !== undefined;
  return (
    <RadixAccordion.Root
      type="single"
      collapsible
      value={controlled ? (expanded ? "item" : "") : undefined}
      defaultValue={!controlled && initialExpand ? "item" : undefined}
      onValueChange={
        onExpandedChange
          ? (value) => onExpandedChange(value === "item")
          : undefined
      }
      className={clsx(
        styles.expand_menu,
        appearance === "title" && styles.appearance_title,
      )}
    >
      <RadixAccordion.Item value="item">
        <RadixAccordion.Trigger asChild className={headerClassName}>
          {trigger}
        </RadixAccordion.Trigger>
        <RadixAccordion.Content className={clsx(styles.content, contentClassName)}>
          {children}
        </RadixAccordion.Content>
      </RadixAccordion.Item>
    </RadixAccordion.Root>
  );
}
