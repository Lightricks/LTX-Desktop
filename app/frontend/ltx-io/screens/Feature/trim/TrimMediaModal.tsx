import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { type ReactNode, useState } from "react";

import type { CloseReason } from "../../../components/shared/Modal/ModalContext";

import styles from "./TrimMediaModal.module.scss";
import { MIN_TRIM_DURATION_SECONDS } from "./trimConstants.ts";
import {
  type TrimRange,
  initialTrimRange,
  isApplicableTrimRange,
} from "./trimGeometry.ts";

/**
 * Title, selection state, and Cancel/Done for the audio and video trim modals.
 * The media-specific selector renders into `renderSelector`.
 */
export function TrimMediaModal({
  mediaKind,
  durationSeconds,
  maxDurationSeconds,
  minDurationSeconds = MIN_TRIM_DURATION_SECONDS,
  subtitle,
  initialRange,
  saveErrorMessage,
  onSave,
  dismiss,
  renderSelector,
}: {
  mediaKind: "audio" | "video";
  durationSeconds: number;
  maxDurationSeconds: number;
  minDurationSeconds?: number;
  /** Defaults to the feature's duration-cap sentence. */
  subtitle?: string;
  /** Defaults to the first third of the clip, capped at the max. */
  initialRange?: TrimRange;
  saveErrorMessage: (error: unknown) => string;
  onSave: (startSec: number, endSec: number) => Promise<void>;
  dismiss: (reason: CloseReason) => void;
  renderSelector: (selection: {
    range: TrimRange;
    onRangeChange: (range: TrimRange) => void;
  }) => ReactNode;
}) {
  const [range, setRange] = useState<TrimRange>(
    () =>
      initialRange ??
      initialTrimRange(durationSeconds, maxDurationSeconds, minDurationSeconds),
  );
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const canApply = isApplicableTrimRange(range, {
    durationSeconds,
    maxDurationSeconds,
    minDurationSeconds,
  });

  return (
    <div className={styles.modal}>
      <div className={styles.header}>
        <Text as="h2" variant="heading" size="sm">
          {`Trim ${mediaKind}`}
        </Text>
        <Text as="span" variant="body" size="md">
          {subtitle ??
            `This feature allows input ${mediaKind} not longer than ${maxDurationSeconds} seconds`}
        </Text>
      </div>

      {renderSelector({ range, onRangeChange: setRange })}

      {saveError ? (
        <span className={styles.saveError} role="alert">
          {saveError}
        </span>
      ) : null}

      <div className={styles.actions}>
        <Button
          appearance="neutral"
          hierarchy="plain"
          size="md"
          label="Cancel"
          disabled={isSaving}
          onClick={() => dismiss("cancel")}
        />
        <Button
          appearance="brand"
          hierarchy="primary"
          size="md"
          label="Done"
          disabled={!canApply || isSaving}
          isLoading={isSaving}
          onClick={() => {
            setSaveError(null);
            setIsSaving(true);
            // A failed save keeps the modal open, and reports inside it — the
            // overlay hides the field's own error line.
            void onSave(range.startSec, range.endSec)
              .catch((error: unknown) => {
                setSaveError(saveErrorMessage(error));
              })
              .finally(() => setIsSaving(false));
          }}
        />
      </div>
    </div>
  );
}
