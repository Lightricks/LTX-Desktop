import type { ReactNode, Ref, RefObject } from "react";
import { clsx } from "clsx";

import { Button as DsButton } from "@/ds/Button/Button";
import { Text } from "@/ds/Text/Text";

import { SettingsInputRow } from "./SettingsContent";
import styles from "./SettingsContent.module.scss";

export function ApiKeySettingsBlock({
  title,
  description,
  statusBadge,
  getKeyLink,
  notice,
  input,
  modelSelect,
  replaceDisabled,
  onReplace,
  isSaving = false,
  actionsLocked = false,
  saveError,
  keyConfigured = false,
  onRemove,
  sectionRef,
  className,
}: {
  title: string;
  description: ReactNode;
  statusBadge?: ReactNode;
  getKeyLink?: ReactNode;
  notice?: ReactNode;
  input: ReactNode;
  modelSelect?: ReactNode;
  replaceDisabled: boolean;
  onReplace: () => void | Promise<void>;
  isSaving?: boolean;
  /** True while any key is being saved or removed, so two providers cannot race. */
  actionsLocked?: boolean;
  saveError?: string | null;
  keyConfigured?: boolean;
  onRemove?: () => void | Promise<void>;
  sectionRef?: RefObject<HTMLElement | null>;
  className?: string;
}) {
  return (
    <section
      ref={sectionRef as Ref<HTMLElement> | undefined}
      className={clsx(styles.apiKeysSubsection, className)}
    >
      <div className={styles.subsectionHeaderRow}>
        <div className={styles.subsectionHeading}>
          <Text as="h3" variant="heading" size="xs">
            {title}
          </Text>
          {statusBadge}
        </div>
      </div>
      <div className={styles.apiKeysMeta}>
        <p className={styles.apiKeysDescriptionLine}>
          {description}
          {keyConfigured && onRemove ? (
            <>
              {" "}
              <button
                type="button"
                className={styles.apiKeyInlineLink}
                disabled={actionsLocked}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  void onRemove();
                }}
              >
                Remove key
              </button>
            </>
          ) : getKeyLink ? (
            <> {getKeyLink}</>
          ) : null}
        </p>
        {notice ? <div className={styles.apiKeyNotice}>{notice}</div> : null}
      </div>
      <SettingsInputRow className={styles.apiKeyControlRow}>
        <div className={styles.apiKeyFieldStack}>
          <div className={styles.apiKeyField}>{input}</div>
          {saveError ? (
            <p className={styles.apiKeySaveError} role="alert">
              {saveError}
            </p>
          ) : null}
        </div>
        {modelSelect ? (
          <div className={styles.apiKeyModelSelect}>{modelSelect}</div>
        ) : null}
        <DsButton
          appearance="neutral"
          hierarchy="secondary"
          size="lg"
          label={keyConfigured ? "Replace key" : "Add key"}
          disabled={replaceDisabled || isSaving || actionsLocked}
          isLoading={isSaving}
          onClick={() => void onReplace()}
          className={styles.apiKeyReplaceButton}
        />
      </SettingsInputRow>
    </section>
  );
}
