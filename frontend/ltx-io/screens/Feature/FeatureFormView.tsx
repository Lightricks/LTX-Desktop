import { ActivityCircular } from "@ds/ActivityCircular/ActivityCircular";
import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { useMemo, type ComponentType, type ReactNode } from "react";
import CloseIcon from "@ds/assets/Icons/Close/Normal.svg?react";

import { EnhanceSplitButton } from "@/components/EnhanceSplitButton";
import type { EnhanceProvider } from "@/hooks/use-prompt-enhancer-provider";
import type { ExploreAsset } from "@/lib/explore-contract";

import { MemoryFits } from "../../components/MemoryMeter/MemoryMeter";

import { SettingField } from "./fields/SettingField";
import { AudioAssetField } from "./fields/AudioAssetField";
import { ImageAssetField } from "./fields/ImageAssetField";
import { ImageAssetsField } from "./fields/ImageAssetsField";
import { KeyframesField } from "./fields/keyframes/KeyframesField";
import { MotionPathField } from "./fields/MotionPathField";
import { AdvancedFields } from "./fields/AdvancedFields";
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
import { Info } from "../../components/shared/Info/Info";
import type { EmptyStateAdapter } from "./results/ResultsFeed";
import type { ResultFrameProps } from "./results/ResultFrame";
import {
  createFeatureFieldChange,
  readAudioAssetFieldValue,
  readImageAssetFieldValue,
  readImageAssetsFieldValue,
  readKeyframesFieldValue,
  readMotionPathFieldValue,
  readOptionsFieldValue,
  readSliderFieldValue,
  readVideoAssetFieldValue,
  type FeatureFieldChange,
  type FeatureFieldSchema,
  type FeatureFormModel,
  type FeatureValues,
  type FieldOption,
  type ImageAssetFieldKey,
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
  /** An action beside an `image-asset` field's label, such as making the image with AI. */
  imageAssetAction?: (input: {
    fieldId: string;
    label: string;
    values: FeatureValues<TSchema>;
    setAsset: (asset: ExploreAsset) => void;
  }) => ReactNode;
};

export type VideoAssetRangeBinding<TSchema extends FeatureFieldSchema> = {
  startDataKey: SliderFieldKey<TSchema>;
  durationDataKey: SliderFieldKey<TSchema>;
  minDuration: number;
  maxDuration: number;
  label: string;
  /** Preflight longest-passing selection at the current tier, when known. */
  maxDurationSeconds?: number;
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
    // Product cap tightened by the preflight longest-passing selection when known.
    maxDurationSec:
      binding.maxDurationSeconds == null
        ? binding.maxDuration
        : Math.min(binding.maxDuration, binding.maxDurationSeconds),
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
  generateLabel,
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
  pageId,
  generations,
  isGenerationsLoading,
  isGenerationsFetching,
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
  negativePromptAccessory,
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
  /** Names this page for the remembered results scroll position. */
  pageId: string;
  generations: Generation[];
  isGenerationsLoading: boolean;
  isGenerationsFetching: boolean;
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
  /** Shown under the negative prompt field, e.g. the NAG strength row. */
  negativePromptAccessory?: ReactNode;
}) {
  const fieldIssue = (fieldId: string) =>
    revealedIssues.find(
      (candidate) => candidate.fieldId === fieldId && candidate.tone !== "risk",
    );
  const memory = isUnavailable ? null : (form.memory ?? null);
  const memoryFits = isUnavailable || memory != null ? null : (form.memoryFits ?? null);
  const resolvedGenerateLabel =
    generateLabel ?? (memory?.over ? "Generate anyway" : "Generate");
  const resolvedGenerateHint =
    generateHint ??
    (memory?.over
      ? undefined
      : (memory?.estimatedTimeText ?? form.estimatedTimeText ?? undefined));
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
                        const issue = fieldIssue(field.id);
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
                                {field.helpText ? (
                                  <Info
                                    content={field.helpText}
                                    side="top"
                                    align="start"
                                    maxWidth={280}
                                    size="sm"
                                  />
                                ) : null}
                              </Text>
                              <textarea
                                className={
                                  field.compact
                                    ? `${styles.textarea} ${styles.negativePromptTextarea}`
                                    : styles.textarea
                                }
                                value={String(values[field.dataKey] ?? "")}
                                placeholder={field.placeholder}
                                disabled={field.disabledReason != null}
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
                              ) : field.disabledReason ? (
                                <span className={styles.fieldNotice} role="status">
                                  {field.disabledReason}
                                </span>
                              ) : null}
                              {field.id === "negativePrompt" ? negativePromptAccessory : null}
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
                              const issue = fieldIssue(field.id);
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
                        const issue = fieldIssue(field.id);
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
                              action={
                                presentation?.imageAssetAction
                                  ? ({ setAsset }) =>
                                      presentation.imageAssetAction?.({
                                        fieldId: field.id,
                                        label: field.label,
                                        values,
                                        setAsset,
                                      })
                                  : undefined
                              }
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
                      case "image-assets": {
                        const field = group.field;
                        return (
                          <section
                            key={field.id}
                            className={styles.stepSection}
                          >
                            <ImageAssetsField
                              field={field}
                              maxCount={field.maxCount}
                              value={readImageAssetsFieldValue(values, field.dataKey)}
                              issue={revealedIssues.find(
                                (candidate) => candidate.fieldId === field.id,
                              )}
                              onChange={(nextValue) =>
                                onFieldChange(
                                  createFeatureFieldChange<
                                    TSchema,
                                    typeof field.dataKey
                                  >({
                                    kind: "image-assets",
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
                      case "keyframes": {
                        const field = group.field;
                        return (
                          <section
                            key={field.id}
                            className={styles.stepSection}
                          >
                            <KeyframesField
                              field={field}
                              maxCount={field.maxCount}
                              value={readKeyframesFieldValue(values, field.dataKey)}
                              duration={Number(readOptionsFieldValue(values, field.durationKey))}
                              fps={Number(readOptionsFieldValue(values, field.fpsKey))}
                              issue={fieldIssue(field.id)}
                              onChange={(nextValue) =>
                                onFieldChange(
                                  createFeatureFieldChange<
                                    TSchema,
                                    typeof field.dataKey
                                  >({
                                    kind: "keyframes",
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
                        const issue = fieldIssue(field.id);
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
                        const issue = fieldIssue(field.id);
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
                        const issue = fieldIssue(field.id);
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
                      case "motion-path": {
                        const field = group.field;
                        return (
                          <section
                            key={field.id}
                            className={styles.stepSection}
                          >
                            <MotionPathField
                              field={field}
                              image={readImageAssetFieldValue(values, field.imageKey) ?? null}
                              paths={readMotionPathFieldValue(values, field.dataKey)}
                              duration={Number(readOptionsFieldValue(values, field.durationKey))}
                              issue={fieldIssue(field.id)}
                              action={
                                presentation?.imageAssetAction
                                  ? ({ setAsset }) =>
                                      presentation.imageAssetAction?.({
                                        fieldId: field.imageKey,
                                        label: field.label,
                                        values,
                                        setAsset,
                                      })
                                  : undefined
                              }
                              onChange={(edit) =>
                                onFieldChange(
                                  ...("image" in edit
                                    ? [
                                        createFeatureFieldChange<
                                          TSchema,
                                          ImageAssetFieldKey<TSchema>
                                        >({
                                          kind: "image-asset",
                                          fieldId: field.id,
                                          dataKey: field.imageKey,
                                          value: edit.image,
                                        }),
                                      ]
                                    : [
                                        createFeatureFieldChange<
                                          TSchema,
                                          typeof field.dataKey
                                        >({
                                          kind: "motion-path",
                                          fieldId: field.id,
                                          dataKey: field.dataKey,
                                          value: edit.paths,
                                        }),
                                        createFeatureFieldChange<
                                          TSchema,
                                          typeof field.durationKey
                                        >({
                                          kind: "options",
                                          fieldId: field.durationKey,
                                          dataKey: field.durationKey,
                                          value: edit.duration,
                                        }),
                                      ]),
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

                  {isUnavailable ? null : (
                    <AdvancedFields
                      fields={form.advancedFields ?? []}
                      values={values}
                      revealedIssues={revealedIssues}
                      onFieldChange={onFieldChange}
                    />
                  )}
                  <FormSeedField />
                  </div>

                  <div className={styles.stepActions}>
                    {memoryFits ? (
                      <MemoryFits model={memoryFits} />
                    ) : null}
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
                          label={resolvedGenerateLabel}
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
                        {resolvedGenerateHint ? (
                          <Text
                            as="span"
                            variant="body"
                            size="sm"
                            className={styles.generateHint}
                          >
                            {resolvedGenerateHint}
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
                    // One feed for each page. Saved custom LoRAs share one route, so this feed
                    // stays mounted between them. The key restarts its scroll and result state.
                    key={pageId}
                    pageId={pageId}
                    generations={generations}
                    isLoading={isGenerationsLoading}
                    isFetching={isGenerationsFetching}
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
