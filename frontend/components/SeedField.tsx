import { Tooltip } from "@ds/Tooltip/Tooltip";
import { clsx } from "clsx";
import { useState } from "react";

import {
  parseGenerationSeedInput,
  randomGenerationSeed,
} from "../lib/generation-seed";

import styles from "./SeedField.module.scss";

function DiceIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      aria-hidden
    >
      <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16" />
    </svg>
  );
}

function LockIcon({ locked }: { locked: boolean }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      aria-hidden
    >
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
      <path d={locked ? "M7 11V7a5 5 0 0 1 10 0v4" : "M7 11V7a5 5 0 0 1 9 0"} />
    </svg>
  );
}

export function SeedField({
  value,
  onChange,
  ariaLabel,
  locked = false,
  onLockedChange,
  disabled = false,
  className,
}: {
  value: number;
  onChange: (seed: number) => void;
  ariaLabel: string;
  locked?: boolean;
  /** Omit to hide the lock button, e.g. where the lock is a separate toggle. */
  onLockedChange?: (locked: boolean) => void;
  disabled?: boolean;
  className?: string;
}) {
  const [draft, setDraft] = useState<string | null>(null);

  return (
    <div className={clsx(styles.control, className)}>
      <input
        className={styles.input}
        inputMode="numeric"
        autoComplete="off"
        aria-label={ariaLabel}
        disabled={disabled}
        value={draft ?? String(value)}
        onChange={(event) => {
          const text = event.currentTarget.value;
          const parsed = parseGenerationSeedInput(text);
          if (parsed == null) {
            setDraft(text);
            return;
          }
          onChange(parsed);
          setDraft(null);
        }}
        onBlur={() => {
          if (draft == null) return;
          if (draft.trim() === "") onChange(0);
          setDraft(null);
        }}
      />
      <Tooltip
        title="New seed"
        content="Replace this value with a random seed."
      >
        <button
          type="button"
          className={styles.iconButton}
          aria-label="Randomize seed"
          disabled={disabled}
          onClick={() => {
            setDraft(null);
            onChange(randomGenerationSeed(value));
          }}
        >
          <DiceIcon />
        </button>
      </Tooltip>
      {onLockedChange ? (
        <Tooltip
          title={locked ? "Unlock seed" : "Lock seed"}
          content={
            locked
              ? "Pick a new seed after each generation."
              : "Reuse this seed on every generation."
          }
        >
          <button
            type="button"
            className={clsx(styles.iconButton, locked && styles.iconButtonSelected)}
            aria-label={locked ? "Unlock seed" : "Lock seed"}
            aria-pressed={locked}
            disabled={disabled}
            onClick={() => onLockedChange(!locked)}
          >
            <LockIcon locked={locked} />
          </button>
        </Tooltip>
      ) : null}
    </div>
  );
}