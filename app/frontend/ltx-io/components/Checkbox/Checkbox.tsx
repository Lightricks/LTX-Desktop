import { clsx } from "clsx";
import { useLayoutEffect, useRef } from "react";

import CheckMarkIcon from "@/ltx-io/assets/checkmark.svg?react";

import styles from "./Checkbox.module.scss";

type CheckboxProps = {
  checked: boolean;
  indeterminate?: boolean;
  onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  appearance?: "neutral" | "danger" | "brand";
  children?: React.ReactNode;
} & Omit<React.InputHTMLAttributes<HTMLInputElement>, "type" | "children">;

export function Checkbox({
  checked,
  indeterminate = false,
  onChange,
  appearance = "neutral",
  className,
  children,
  ...props
}: CheckboxProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  useLayoutEffect(() => {
    if (inputRef.current) {
      inputRef.current.indeterminate = indeterminate
    }
  }, [indeterminate])

  return (
    <label className={clsx(styles.checkbox, className)}>
      <input
        {...props}
        ref={inputRef}
        type="checkbox"
        checked={checked}
        onChange={onChange}
      />
      <span className={clsx(styles.checkbox_control, styles[appearance])}>
        <CheckMarkIcon />
        <span className={styles.checkbox_dash} aria-hidden />
      </span>
      {children}
    </label>
  );
}
