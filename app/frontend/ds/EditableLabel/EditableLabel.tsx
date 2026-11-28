import { clsx } from "clsx";
import { useEffect, useRef, useState } from "react";

import { useAutoSizeTextArea } from "@ds/hooks/useAutoSizeTextArea";

import styles from "./EditableLabel.module.scss";

export type TextSize = "xs" | "sm" | "md" | "lg" | "xl";

/**
 * Props for the EditableLabel component
 * @interface EditableLabelProps
 */
export type EditableLabelProps = {
  /** Accessible name for the editable field */
  "aria-label"?: string;
  /** Placeholder text when empty */
  placeholder?: string;
  /** Whether the input is disabled */
  disabled?: boolean;
  /** Whether the input is read-only */
  readOnly?: boolean;
  /** Current value */
  value: string;
  /** Callback fired when value changes (on Enter or blur) */
  onInputChange: (text: string) => void;
  /** Text size variant */
  textSize: TextSize;
  /** Additional class name for the input element */
  handleClassName?: string;
  /** Maximum character length */
  maxLength?: number;
  /** Whether to render a textarea instead of an input */
  isMultiline?: boolean;
  /** Visual style variant */
  styleVariant?: "label" | "heading";
  /** Whether to focus the input on mount */
  focusOnMount?: boolean;
};

/**
 * An inline editable text field that supports single-line and multiline modes.
 * Commits on Enter/blur, reverts on Escape.
 *
 * @component
 * @example
 * ```tsx
 * <EditableLabel
 *   value="My Label"
 *   placeholder="Enter name..."
 *   textSize="md"
 *   onInputChange={(text) => console.log(text)}
 * />
 * ```
 */
export function EditableLabel({
  "aria-label": ariaLabel,
  placeholder = "",
  value,
  onInputChange,
  textSize,
  handleClassName,
  maxLength,
  isMultiline = false,
  styleVariant = "label",
  disabled,
  readOnly,
  focusOnMount = false,
}: EditableLabelProps) {
  const [currentValue, setCurrentValue] = useState(value);
  const inputRef = useRef<HTMLInputElement>(null);
  const textAreaRef = useRef<HTMLTextAreaElement>(null);

  useAutoSizeTextArea(textAreaRef, currentValue);

  useEffect(() => {
    setCurrentValue(value);
  }, [value]);

  useEffect(() => {
    if (focusOnMount) {
      const element = isMultiline ? textAreaRef.current : inputRef.current;
      if (element) {
        element.focus();
      }
    }
    // isMultiline is intentionally excluded: this effect must only run when
    // focusOnMount first becomes true. Re-running it on isMultiline changes
    // after mount would cause unexpected focus side effects.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusOnMount]);

  const handleChange = (
    event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    setCurrentValue(event.target.value);
  };

  const handleKeyDown = (
    event: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    const ref = isMultiline ? textAreaRef : inputRef;

    switch (event.key) {
      case "Escape":
        setCurrentValue(value);
        break;
      case "Enter":
        event.preventDefault();
        void onInputChange(currentValue);
        ref.current?.blur();
        break;
    }
  };

  const handleBlur = (allowEmptyValue?: boolean) => {
    if (currentValue.trim() === "" && !allowEmptyValue) {
      setCurrentValue(value);
    } else {
      void onInputChange(currentValue);
    }

    if (isMultiline) {
      textAreaRef.current?.scrollTo(0, 0);
    }
  };

  return isMultiline ? (
    <textarea
      ref={textAreaRef}
      className={clsx(
        styles.textarea,
        styles.content,
        styles[styleVariant],
        styles[textSize],
        handleClassName,
      )}
      value={currentValue}
      aria-label={ariaLabel}
      placeholder={placeholder}
      onChange={handleChange}
      onKeyDown={handleKeyDown}
      onBlur={() => handleBlur(true)}
      disabled={disabled}
      readOnly={readOnly}
    />
  ) : (
    <input
      ref={inputRef}
      size={currentValue.length || placeholder.length}
      type="text"
      className={clsx(
        styles.content,
        styles[styleVariant],
        styles[textSize],
        handleClassName,
      )}
      maxLength={maxLength}
      placeholder={placeholder}
      aria-label={ariaLabel}
      value={currentValue}
      onChange={handleChange}
      onKeyDown={handleKeyDown}
      onBlur={() => handleBlur()}
      disabled={disabled}
      readOnly={readOnly}
    />
  );
}
