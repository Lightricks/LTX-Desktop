import { useDebouncedCallback } from "@mantine/hooks";
import { clsx } from "clsx";
import { forwardRef, useEffect, useRef, useState } from "react";

import styles from "./TextField.module.scss";

export type TextFieldProps = {
  placeholder?: string;
  /** For uncontrolled mode */
  defaultValue?: string;
  /** For controlled mode - when provided, component becomes controlled */
  value?: string;
  onInputChange?: (text: string) => void;
  onFocus?: (e: React.FocusEvent<HTMLInputElement>) => void;
  onBlur?: () => void;
  readOnly?: boolean;
  disabled?: boolean;
  maxLength?: number;
  id?: string;
  /** Native input name — helps browser / password-manager autofill. */
  name?: string;
  /**
   * Autofill token (e.g. `name`, `organization`). Defaults to `off`.
   * When enabled, skips the 1Password ignore hint so managers can fill.
   */
  autoComplete?: string;
  debounceDelay?: number;
  className?: string;
  autoFocus?: boolean;
  spacing?: "compact" | "comfortable";
  highlighted?: boolean;
  highlightColor?: string;
  "data-testid"?: string;
};

export const TextField = forwardRef<HTMLInputElement, TextFieldProps>(function TextField(
  {
    placeholder,
    defaultValue = "",
    value: controlledValue,
    onInputChange,
    onFocus,
    onBlur,
    readOnly,
    disabled,
    maxLength,
    id,
    name,
    autoComplete = "off",
    debounceDelay = 0,
    className,
    autoFocus,
    spacing = "compact",
    highlighted,
    highlightColor,
    "data-testid": dataTestId,
  }: TextFieldProps,
  ref,
) {
  const isControlled = controlledValue !== undefined;
  const [uncontrolledValue, setUncontrolledValue] = useState<string>(defaultValue);
  const previousDefaultValueRef = useRef<string>(defaultValue);
  const autofillEnabled = autoComplete !== "off" && autoComplete !== "";

  useEffect(() => {
    if (!isControlled && defaultValue !== previousDefaultValueRef.current) {
      setUncontrolledValue(defaultValue);
      previousDefaultValueRef.current = defaultValue;
    }
  }, [defaultValue, isControlled]);

  const value = isControlled ? controlledValue : uncontrolledValue;

  const onChange = useDebouncedCallback(() => {
    onInputChange?.(value.toString());
  }, debounceDelay);

  return (
    <input
      id={id}
      name={name}
      data-testid={dataTestId}
      className={clsx(
        styles.content,
        highlighted && styles.highlighted,
        styles[spacing],
        className,
      )}
      // Mobile type scale (data-ltxio-type): body-md → body_xl under 768px; applies
      // only under the `.ltx-io` root.
      data-ltxio-type="body-md"
      data-highlight-color={highlightColor}
      type="text"
      placeholder={placeholder}
      value={value}
      onChange={(e) => {
        const newValue = e.target.value;
        if (!isControlled) {
          setUncontrolledValue(newValue);
        }
        if (isControlled) {
          onInputChange?.(newValue);
        } else {
          onChange();
        }
      }}
      onFocus={onFocus}
      onBlur={onBlur}
      maxLength={maxLength}
      ref={ref}
      readOnly={readOnly}
      disabled={disabled}
      autoFocus={autoFocus}
      dir="auto"
      autoComplete={autoComplete}
      // Keep password managers off unless the caller opts into autofill.
      {...(!autofillEnabled ? { "data-1p-ignore": true } : {})}
    />
  );
});
