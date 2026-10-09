import { ActivityCircular } from "@ds/ActivityCircular/ActivityCircular";
import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { useMemo, type ComponentType, type ReactNode } from "react";
import CloseIcon from "@ds/assets/Icons/Close/Normal.svg?react";

import { EnhanceSplitButton } from "@/components/EnhanceSplitButton";
import type { EnhanceProvider } from "@/hooks/use-prompt-enhancer-provider";

import { SettingField } from "./fields/SettingField";
import { AudioAssetField } from "./fields/AudioAssetField";
import { ImageAssetField } from "./fields/ImageAssetField";
import { FormSeedField } from "./fields/FormSeedField";
import { SliderField } from "./fields/SliderField";
import { VideoAssetField } from "./fields/VideoAssetField";
import type { VideoAssetRangeValue } from "./fields/VideoAssetPreview";
import {
  applyOptionFieldOrder,
  groupConsecutiveOptionFields,
  optionControlValue,
} from "./featureFormDisplay";
import { ResultsFeed } from "./results/ResultsFeed";
import { MIN_TRIM_DURATION_SECONDS } from "./trim/trimConstants.ts";
import type { EmptyStateAdapter } from "./results/ResultsFeed";
import type { ResultFrameProps } from "./results/ResultFrame";
import {
  createFeatureFieldChange,
  readAudioAssetFieldValue,
  readImageAssetFieldValue,
  readOptionsFieldValue,
  readSliderFieldValue,
  readVideoAssetFieldValue,
  type FeatureFieldChange,
  type FeatureFieldSchema,
  type FeatureFormModel,
  type FeatureValues,
  type FieldOption,
  type OptionsFieldDef,
  type OptionsFieldKey,
  type SliderFieldKey,
  type TextareaFieldKey,
  type ValidationIssue,
} from "./types";
import type { Generation } from "../../lib/resultsFeedModel";
import styles from "./FeatureFormView.module.scss";

export type FeatureFormPresentationAdapter<
  TSchema extends FeatureFieldSchema,
> = {
  orderOptionFields?: (
    fields: readonly OptionsFieldDef<TSchema>[],
  ) => readonly OptionsFieldDef<TSchema>[];
  getOptionIcon?: (
    field: OptionsFieldDef<TSchema>,
    option: FieldOption,
  ) => ReactNode;
  /** Show image previews at their own aspect ratio, not in a fixed-height box. */
  naturalImagePreview?: boolean;
};

export type VideoAssetRangeBinding<TSchema extends FeatureFieldSchema> = {
  startDataKey: SliderFieldKey<TSchema>;
  durationDataKey: SliderFieldKey<TSchema>;
  minDuration: number;
  maxDuration: number;
  label: string;
};

function assertNever(value: never): never {
  throw new Error(`Unhandled feature form field: ${String(value)}`);
}

function videoAssetRangeValue<TSchema extends FeatureFieldSchema>(
  values: FeatureValues<TSchema>,
  binding: VideoAssetRangeBinding<TSchema>,
  onPatchValues: (patch: Partial<FeatureValues<TSchema>>) => void,
): VideoAssetRangeValue {
  return {
    startSec: readSliderFieldValue(values, binding.startDataKey),
    durationSec: readSliderFieldValue(values, binding.durationDataKey),
    minDurationSec: binding.minDuration,
    maxDurationSec: binding.maxDuration,
    label: binding.label,
    onChange: (startSec, durationSec) => {
      // Computed keys always widen to a string index; the keys themselves are
      // already proven slider keys by `VideoAssetRangeBinding<TSchema>`, so a
      // typo fails at the binding literal instead of silently writing a bogus key.
      onPatchValues({
        [binding.startDataKey]: startSec,
        [binding.durationDataKey]: durationSec,
      } as Partial<FeatureValues<TSchema>>);
    },
  };
}

export function FeatureFormView<TSchema extends FeatureFieldSchema>({
  form,
  values,
  onFieldChange,
  revealedIssues,
  onReset,
  onGenerate,
  generateLabel = "Generate",
  generateHint,
  isDownloadBusy = false,
  onCancelDownload,
  showManualEnhance = false,
  canEnhance = false,
  isEnhancing = false,
  enhanceError = null,
  onEnhance,
  enhanceLabel = "Enhance",
  enhanceProvider = "local",
  canToggleEnhanceProvider = false,
  onEnhanceProviderChange,
  canGenerate,
  isUnavailable,
  isCreating,
  actionError,
  generations,
  isGenerationsLoading,
  generationsError,
  onCancel,
  onRetry,
  onDelete,
  ResultFrameAdapter,
  EmptyStateAdapter,
  presentation,
  audioTrimCapSeconds,
  audioTrimMinSeconds = MIN_TRIM_DURATION_SECONDS,
  videoRange,
  onPatchValues,
}: {
  form: FeatureFormModel<TSchema>;
  values: FeatureValues<TSchema>;
  onFieldChange: (...changes: FeatureFieldChange<TSchema>[]) => void;
  revealedIssues: ValidationIssue[];
  onReset: () => void;
  onGenerate: () => void;
  generateLabel?: string;
  generateHint?: string;
  isDownloadBusy?: boolean;
  onCancelDownload?: () => void;
  showManualEnhance?: boolean;
  canEnhance?: boolean;
  isEnhancing?: boolean;
  enhanceError?: string | null;
  onEnhance?: () => void;
  enhanceLabel?: string;
  enhanceProvider?: EnhanceProvider;
  canToggleEnhanceProvider?: boolean;
  onEnhanceProviderChange?: (provider: EnhanceProvider) => void;
  canGenerate: boolean;
  isUnavailable: boolean;
  isCreating: boolean;
  actionError: ReactNode;
  generations: Generation[];
  isGenerationsLoading: boolean;
  generationsError: string | null;
  onCancel: (id: string) => void;
  onRetry: (id: string) => void;
  onDelete: (id: string) => void;
  ResultFrameAdapter: ComponentType<ResultFrameProps>;
  EmptyStateAdapter?: EmptyStateAdapter;
  presentation?: FeatureFormPresentationAdapter<TSchema>;
  /**
   * Max input length for `audio-asset` fields. `null`/omitted means the cap is
   * unknown (model specs unresolved), which skips the auto-trim prompt.
   */
  audioTrimCapSeconds?: number | null;
  /**
   * Shortest selection the trim modal accepts, so a trimmed clip is still long
   * enough to generate from. Defaults to the Explore floor.
   */
  audioTrimMinSeconds?: number;
  videoRange?: VideoAssetRangeBinding<TSchema>;
  onPatchValues?: (patch: Partial<FeatureValues<TSchema>>) => void;
}) {
  const displayGroups = useMemo(() => {
    const displayFields = presentation?.orderOptionFields
      ? applyOptionFieldOrder(form.fields, presentation.orderOptionFields)
      : form.fields;
    return groupConsecutiveOptionFields(displayFields);
  }, [form.fields, presentation?.orderOptionFields]);

  return (
    <div className={styles.formShell}>
      <div className={styles.formScroll}>
        <div className={styles.formBody}>
          <div className={styles.stepStack}>
            <section className={styles.stepRow}>
              <div className={styles.stepBody}>
                <div className={styles.stepControls}>
                  <div
                    className={styles.stepFields}
                    {...(isDownloadBusy ? { inert: "" } : {})}
                  >
                  {displayGroups.map((group) => {
                    switch (group.kind) {
                      case "textarea": {
                        const field = group.field;
                        const issue = revealedIssues.find(
                          (candidate) => candidate.fieldId === field.id,
                        );
                        return (
                          <section
                            key={field.id}
                            className={styles.stepSection}
                          >
                            <div className={styles.field}>
                              <Text
                                as="span"
                                variant="body"
                                size="md"
                                className={styles.fieldLabel}
                              >
                                {field.label}
                              </Text>
                              <textarea
                                className={styles.textarea}
                                value={String(values[field.dataKey] ?? "")}
                                placeholder={field.placeholder}
                                onChange={(event) =>
                                  onFieldChange(
                                    createFeatureFieldChange<
                                      TSchema,
                                      TextareaFieldKey<TSchema>
                                    >({
                                      kind: "textarea",
                                      fieldId: field.id,
                                      dataKey: field.dataKey,
                                      value: event.currentTarget.value,
                                    }),
                                  )
                                }
                              />
                              {issue ? (
                                <span className={styles.fieldError}>
                                  {issue.message}
                                </span>
                              ) : null}
                              {field.id === "prompt" && showManualEnhance ? (
                                <div className={styles.promptEnhanceRow}>
                                  <EnhanceSplitButton
                                    label={enhanceLabel}
                                    onEnhance={() => onEnhance?.()}
                                    disabled={!canEnhance}
                                    isLoading={isEnhancing}
                                    showProviderMenu={canToggleEnhanceProvider}
                                    provider={enhanceProvider}
                                    onProviderChange={
                                      canToggleEnhanceProvider
                                        ? onEnhanceProviderChange
                                        : undefined
                                    }
                                  />
                                  {enhanceError ? (
                                    <span className={styles.fieldError} role="alert">
                                      {enhanceError}
                                    </span>
                                  ) : null}
                                </div>
                              ) : null}
                            </div>
                          </section>
                        );
                      }
                      case "options":
                        // Hidden when unavailable (the message near the actions covers it).
                        // Defaults are local-valid, so before specs resolve these render the
                        // default combo rather than an unsupported one.
                        if (isUnavailable) return null;
                        return (
                          <div
                            key={group.fields
                              .map((field) => field.id)
                              .join(":")}
                            className={styles.labeledSettings}
                          >
                            {group.fields.map((field) => {
                              const issue = revealedIssues.find(
                                (candidate) => candidate.fieldId === field.id,
                              );
                              return (
                                <SettingField
                                  key={field.id}
                                  field={field}
                                  value={optionControlValue(
                                    readOptionsFieldValue(values, field.dataKey),
                                    field.options,
                                    field.defaultValue,
                                  )}
                                  issue={issue}
                                  getOptionIcon={(option) =>
                                    presentation?.getOptionIcon?.(field, option)
                                  }
                                  onChange={(value) =>
                                    onFieldChange(
                                      createFeatureFieldChange<
                                        TSchema,
                                        OptionsFieldKey<TSchema>
                                      >({
                                        kind: "options",
                                        fieldId: field.id,
                                        dataKey: field.dataKey,
                                        value,
                                      }),
                                    )
                                  }
                                />
                              );
                            })}
                          </div>
                        );
                      case "image-asset": {
                        const field = group.field;
                        const issue = revealedIssues.find(
                          (candidate) => candidate.fieldId === field.id,
                        );
                        return (
                          <section
                            key={field.id}
                            className={styles.stepSection}
                          >
                            <ImageAssetField
                              field={field}
                              naturalPreview={presentation?.naturalImagePreview}
                              value={
                                readImageAssetFieldValue(
                                  values,
                                  field.dataKey,
                                ) ?? null
                              }
                              issue={issue}
                              onChange={(nextValue) =>
                                onFieldChange(
                                  createFeatureFieldChange<
                                    TSchema,
                                    typeof field.dataKey
                                  >({
                                    kind: "image-asset",
                                    fieldId: field.id,
                                    dataKey: field.dataKey,
                                    value: nextValue,
                                  }),
                                )
                              }
                            />
                          </section>
                        );
                      }
                      case "slider": {
                        const field = group.field;
                        const issue = revealedIssues.find(
                          (candidate) => candidate.fieldId === field.id,
                        );
                        return (
                          <section
                            key={field.id}
                            className={styles.stepSection}
                          >
                            <SliderField
                              field={field}
                              value={readSliderFieldValue(
                                values,
                                field.dataKey,
                              )}
                              issue={issue}
                              onChange={(nextValue) =>
                                onFieldChange(
                                  createFeatureFieldChange<
                                    TSchema,
                                    SliderFieldKey<TSchema>
                                  >({
                                    kind: "slider",
                                    fieldId: field.id,
                                    dataKey: field.dataKey,
                                    value: nextValue,
                                  }),
                                )
                              }
                            />
                          </section>
                        );
                      }
                      case "audio-asset": {
                        const field = group.field;
                        const issue = revealedIssues.find(
                          (candidate) => candidate.fieldId === field.id,
                        );
                        return (
                          <section
                            key={field.id}
                            className={styles.stepSection}
                          >
                            <AudioAssetField
                              field={field}
                              value={
                                readAudioAssetFieldValue(
                                  values,
                                  field.dataKey,
                                ) ?? null
                              }
                              issue={issue}
                              trimCapSeconds={audioTrimCapSeconds ?? null}
                              trimMinSeconds={audioTrimMinSeconds}
                              onChange={(nextValue) =>
                                onFieldChange(
                                  createFeatureFieldChange<
                                    TSchema,
                                    typeof field.dataKey
                                  >({
                                    kind: "audio-asset",
                                    fieldId: field.id,
                                    dataKey: field.dataKey,
                                    value: nextValue,
                                  }),
                                )
                              }
                            />
                          </section>
                        );
                      }
                      case "video-asset": {
                        const field = group.field;
                        const issue = revealedIssues.find(
                          (candidate) => candidate.fieldId === field.id,
                        );
                        return (
                          <section
                            key={field.id}
                            className={styles.stepSection}
                          >
                            <VideoAssetField
                              field={field}
                              value={
                                readVideoAssetFieldValue(
                                  values,
                                  field.dataKey,
                                ) ?? null
                              }
                              issue={issue}
                              maxDurationSeconds={field.maxDurationSeconds}
                              range={
                                videoRange && onPatchValues
                                  ? videoAssetRangeValue(
                                      values,
                                      videoRange,
                                      onPatchValues,
                                    )
                                  : undefined
                              }
                              onChange={(nextValue) =>
                                onFieldChange(
                                  createFeatureFieldChange<
                                    TSchema,
                                    typeof field.dataKey
                                  >({
                                    kind: "video-asset",
                                    fieldId: field.id,
                                    dataKey: field.dataKey,
                                    value: nextValue,
                                  }),
                                )
                              }
                            />
                          </section>
                        );
                      }
                      default:
                        return assertNever(group);
                    }
                  })}

                  <FormSeedField />
                  </div>

                  <div className={styles.stepActions}>
                    <div className={styles.stepActionsRow}>
                      <Button
                        appearance="neutral"
                        hierarchy="secondary"
                        size="lg"
                        label="Reset"
                        onClick={onReset}
                        disabled={isDownloadBusy}
                      />
                      <div className={styles.generateButtonWrap}>
                        <Button
                          appearance="brand"
                          hierarchy="primary"
                          size="lg"
                          label={generateLabel}
                          isLoading={isCreating && !isDownloadBusy}
                          leftIcon={
                            isDownloadBusy ? (
                              <ActivityCircular
                                appearance="over-background"
                                size={13.5}
                              />
                            ) : undefined
                          }
                          rightIcon={isDownloadBusy ? <CloseIcon /> : undefined}
                          disabled={!canGenerate && !isDownloadBusy}
                          onClick={
                            isDownloadBusy ? onCancelDownload : onGenerate
                          }
                          aria-label={
                            isDownloadBusy ? "Cancel download" : undefined
                          }
                          className={styles.generateButton}
                        />
                        {generateHint ? (
                          <Text
                            as="span"
                            variant="body"
                            size="sm"
                            className={styles.generateHint}
                          >
                            {generateHint}
                          </Text>
                        ) : null}
                      </div>
                    </div>
                    {isUnavailable ? (
                      <span className={styles.fieldError} role="alert">
                        This machine can&apos;t run any supported settings for
                        this feature.
                      </span>
                    ) : actionError ? (
                      <div className={styles.fieldError} role="alert">
                        {actionError}
                      </div>
                    ) : null}
                  </div>
                </div>

                <div className={styles.stepOutput}>
                  <ResultsFeed
                    generations={generations}
                    isLoading={isGenerationsLoading}
                    loadError={generationsError}
                    onCancel={onCancel}
                    onRetry={onRetry}
                    onDelete={onDelete}
                    ResultFrameAdapter={ResultFrameAdapter}
                    EmptyStateAdapter={EmptyStateAdapter}
                  />
                </div>
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
