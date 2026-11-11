import { clsx } from "clsx";
import { type DragEvent, type ReactNode, useState } from "react";

import styles from "./AddCustomLoraPage.module.scss";

/**
 * The shared drop box for the weights file and the cover image.
 * Clicking it asks the parent to open the file dialog. Dropping a file hands that file back.
 */
export function CustomLoraDropzone({
  disabled,
  className,
  label,
  onActivate,
  onDropFile,
  children,
}: {
  disabled: boolean;
  className?: string;
  label: string;
  onActivate: () => void;
  onDropFile: (file: File) => void;
  children: ReactNode;
}) {
  const [dragOver, setDragOver] = useState(false);

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setDragOver(false);
    if (disabled) return;
    const file = event.dataTransfer.files[0];
    if (file != null) onDropFile(file);
  };

  return (
    <div
      className={clsx(
        styles.dropzone,
        className,
        dragOver && styles.dragOver,
        disabled && styles.dropzoneDisabled,
      )}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      aria-label={label}
      onClick={() => {
        if (!disabled) onActivate();
      }}
      onKeyDown={(event) => {
        if (disabled || (event.key !== "Enter" && event.key !== " ")) return;
        event.preventDefault();
        onActivate();
      }}
      onDragEnter={(event) => {
        event.preventDefault();
        if (!disabled) setDragOver(true);
      }}
      onDragOver={(event) => {
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer)
          event.dataTransfer.dropEffect = disabled ? "none" : "copy";
      }}
      onDragLeave={(event) => {
        event.preventDefault();
        const next = event.relatedTarget;
        if (next instanceof Node && event.currentTarget.contains(next)) return;
        setDragOver(false);
      }}
      onDrop={onDrop}
    >
      {children}
    </div>
  );
}
