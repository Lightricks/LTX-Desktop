import { Button } from "@ds/Button/Button";
import { clsx } from "clsx";
import { Check, Copy, X } from "lucide-react";
import { useEffect, useId, useState } from "react";

import { ALL_DEV_FLAGS, DEV_FLAG_SECTIONS, UNLOADED_FLAGS, type DevFlagKey } from "@/lib/dev-flags";

import {
  useFeatureFlags,
  useUpdateFeatureFlags,
  type FeatureFlagsApi,
} from "../../hooks/useFeatureFlags";

import styles from "./DevPanel.module.scss";

type DevPanelProps = {
  api: FeatureFlagsApi;
  open: boolean;
  onClose: () => void;
};

/**
 * Master-detail feature-flag panel shared by Desktop (Ctrl/Cmd+Shift+D) and the
 * phone remote (`?devpanel=1`). Flags live on the backend, so both hosts edit
 * the same values. The body mounts only while open, so each open refetches.
 */
export function DevPanel({ open, ...rest }: DevPanelProps) {
  return open ? <DevPanelBody {...rest} /> : null;
}

function DevPanelBody({ api, onClose }: Omit<DevPanelProps, "open">) {
  const { data, isError: loadFailed, isFetching, refetch } = useFeatureFlags(api);
  const titleId = useId();
  const update = useUpdateFeatureFlags(api);
  // Until `data` arrives the values are placeholders, so the toggle stays disabled.
  const flags = data ?? UNLOADED_FLAGS;
  const errorMessage = loadFailed
    ? "Couldn't load feature flags."
    : update.isError
      ? "Couldn't save the change."
      : null;
  const [selectedKey, setSelectedKey] = useState<DevFlagKey | undefined>(ALL_DEV_FLAGS[0]?.key);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const selected = ALL_DEV_FLAGS.find((flag) => flag.key === selectedKey);
  const isOn = selected ? flags[selected.key] : false;
  // Never show placeholder defaults as if the backend had confirmed them.
  const toggleLabel =
    data === undefined ? (loadFailed ? "Unavailable" : "Loading…") : isOn ? "Enabled" : "Disabled";

  const copyKey = () => {
    if (!selected) return;
    void navigator.clipboard?.writeText(selected.key).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    });
  };

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div
        className={styles.panel}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.header}>
          <div>
            <h2 id={titleId} className={styles.title}>
              Dev Panel
            </h2>
            <p className={styles.subtitle}>Feature toggles shared by Desktop and its paired devices.</p>
          </div>
          <Button
            appearance="neutral"
            hierarchy="plain"
            size="sm"
            isIconOnly
            leftIcon={<X size={16} />}
            aria-label="Close"
            onClick={onClose}
          />
        </div>

        {errorMessage && (
          <div role="alert" className={styles.error}>
            <span>{errorMessage}</span>
            {loadFailed && (
              <Button
                appearance="neutral"
                hierarchy="secondary"
                size="xs"
                label="Retry"
                isLoading={isFetching}
                onClick={() => void refetch()}
              />
            )}
          </div>
        )}

        <div className={styles.body}>
          <div className={styles.list}>
            {DEV_FLAG_SECTIONS.filter((section) => section.flags.length > 0).map((section) => (
              <div key={section.title ?? "new"}>
                {section.title && <div className={styles.sectionTitle}>{section.title}</div>}
                {section.flags.map((spec) => (
                  <button
                    key={spec.key}
                    type="button"
                    className={clsx(
                      styles.flag,
                      flags[spec.key] && styles.on,
                      spec.key === selectedKey && styles.selected,
                    )}
                    onClick={() => setSelectedKey(spec.key)}
                  >
                    {spec.label}
                  </button>
                ))}
              </div>
            ))}
          </div>

          <div className={styles.detail}>
            {selected ? (
              <>
                <div className={styles.detailHeader}>
                  <h3 className={styles.detailTitle}>{selected.label}</h3>
                  <button type="button" className={styles.key} onClick={copyKey} title="Copy flag key">
                    {selected.key}
                    {copied ? <Check size={12} /> : <Copy size={12} />}
                  </button>
                </div>
                <div className={styles.toggle}>
                  <Button
                    role="switch"
                    aria-checked={isOn}
                    appearance={isOn ? "brand" : "neutral"}
                    hierarchy="secondary"
                    size="md"
                    label={toggleLabel}
                    disabled={data === undefined}
                    isLoading={update.isPending}
                    onClick={() => update.mutate({ [selected.key]: !isOn })}
                  />
                </div>
                <p className={styles.description}>{selected.description}</p>
              </>
            ) : (
              <div className={styles.empty}>No feature flags.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
