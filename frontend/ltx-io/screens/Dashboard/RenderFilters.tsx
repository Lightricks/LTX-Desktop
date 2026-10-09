import { useEffect, useRef, useState } from "react";

import { Text } from "@/ds/Text/Text";

import styles from "./DashboardScreen.module.scss";

export function FilterMultiSelect({
  label,
  options,
  selected,
  onChange,
}: {
  label: string;
  options: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const summary = selectionLabel(selected);
  // A saved value this range lacks stays listed, so it can still be unchecked.
  const listed = [...options, ...selected.filter((value) => !options.includes(value))];

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (rootRef.current?.contains(document.activeElement)) triggerRef.current?.focus();
      setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div className={styles.filter}>
      <Text as="span" variant="body" size="md" className={styles.muted}>
        {label}
      </Text>
      <div className={styles.filterMenu} ref={rootRef}>
        <button
          ref={triggerRef}
          type="button"
          className={`${styles.control} ${styles.filterTrigger}`}
          aria-label={`${label}: ${summary}`}
          aria-expanded={open}
          aria-haspopup="true"
          title={summary}
          disabled={listed.length === 0}
          onClick={() => setOpen((current) => !current)}
        >
          <Text as="span" variant="body" size="md" shouldTruncate>
            {summary}
          </Text>
        </button>
        {open ? (
          <div className={styles.filterPopover} role="group" aria-label={label}>
            {listed.map((option) => (
              <label key={option} className={styles.filterOption}>
                <input
                  type="checkbox"
                  checked={selected.includes(option)}
                  onChange={() => onChange(toggleValue(selected, option))}
                />
                <Text as="span" variant="body" size="md">
                  {option}
                </Text>
              </label>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function selectionLabel(selected: string[]): string {
  if (selected.length === 0) return "Any";
  return selected.join(", ");
}

function toggleValue(selected: string[], value: string): string[] {
  return selected.includes(value)
    ? selected.filter((item) => item !== value)
    : [...selected, value];
}
