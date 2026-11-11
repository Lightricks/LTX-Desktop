import type { UseMutationResult } from "@tanstack/react-query";
import {
  type ComponentType,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { confirmableBudgetIssue, generateHintForTier } from "@/lib/budget-risk";
import { enhanceActionLabel } from "@/lib/enhance-ui-copy";
import type { HomeFeatureId } from "@/lib/home-features";
import {
  type PromptHistoryEntry,
  parsePromptProvenance,
  restorePromptHistory,
} from "@/lib/prompt-provenance";

import { useExploreRuntime } from "../../runtime/ExploreRuntime";
import { DEFAULT_NAG_STRENGTH } from "../../lib/nagStrength";
import { NagStrengthControl } from "./fields/NagStrengthControl";
import { type PromptValues as ManualEnhancePromptValues, useManualEnhance } from "../../hooks/useManualEnhance";
import { type Generation, readGenerationParams } from "../../lib/resultsFeedModel";
import {
  type FeatureFormKey,
  type FeatureSchemaMap,
  useFeatureFormStore,
} from "../../stores/featureFormStore";

import { type FeatureFormPresentationAdapter, FeatureFormView, type VideoAssetRangeBinding } from "./FeatureFormView";
import { RiskyGenerateDialog } from "./RiskyGenerateDialog";
import type { ResultFrameProps } from "./results/ResultFrame";
import type { EmptyStateAdapter } from "./results/ResultsFeed";
import type { FeatureDefinition, FeatureFieldChange } from "./types";
import { useFeatureForm } from "./useFeatureForm";

export function FeatureFormScreen<TId extends HomeFeatureId, TCreateBody, TContext>({
  definition,
  context,
  contextReady,
  resourceError,
  createGeneration,
  ResultFrameAdapter,
  EmptyStateAdapter,
  presentation,
  audioTrimCapSeconds,
  audioTrimMinSeconds,
  videoRange,
  generateLabel,
  generateHint,
  generateBusy = false,
  generateEnabled = true,
  actionError: actionErrorOverride,
  interceptGenerate,
  onCancelGenerate,
  generationFilter,
  storeKey,
  customLoraId,
}: {
  definition: FeatureDefinition<FeatureSchemaMap[TId], TCreateBody, TContext, TId>;
  context: TContext;
  contextReady: boolean;
  resourceError: unknown;
  createGeneration: UseMutationResult<Generation, Error, TCreateBody>;
  ResultFrameAdapter: ComponentType<ResultFrameProps>;
  EmptyStateAdapter?: EmptyStateAdapter;
  presentation?: FeatureFormPresentationAdapter<FeatureSchemaMap[TId]>;
  audioTrimCapSeconds?: number | null;
  audioTrimMinSeconds?: number;
  videoRange?: VideoAssetRangeBinding<FeatureSchemaMap[TId]>;
  generateLabel?: string;
  generateHint?: string;
  generateBusy?: boolean;
  generateEnabled?: boolean;
  actionError?: ReactNode;
  interceptGenerate?: (proceed: () => void) => void;
  onCancelGenerate?: () => void;
  generationFilter?: (generation: Generation) => boolean;
  storeKey?: FeatureFormKey;
  customLoraId?: string;
}) {
  const resolvedStoreKey = storeKey ?? definition.id;
  const form = useFeatureForm(
    definition,
    context,
    createGeneration,
    contextReady,
    resourceError,
    generationFilter,
    resolvedStoreKey,
  );

  const storedHistory = useFeatureFormStore(
    (state) => state.promptHistoryByFeature[resolvedStoreKey],
  );
  const setPromptHistoryState = useFeatureFormStore((state) => state.setPromptHistory);
  const nagStrength = useFeatureFormStore(
    (state) => state.nagStrengthByFeature[resolvedStoreKey] ?? DEFAULT_NAG_STRENGTH,
  );
  const setNagStrength = useFeatureFormStore((state) => state.setNagStrength);

  const params = readGenerationParams(form.generations[0]?.spec);
  const promptText =
    "prompt" in form.values && typeof form.values.prompt === "string"
      ? form.values.prompt
      : "";
  const negativePromptText =
    "negativePrompt" in form.values && typeof form.values.negativePrompt === "string"
      ? form.values.negativePrompt
      : undefined;
  const { history: promptHistory, index: historyIndex } = restorePromptHistory({
    stored: storedHistory,
    prompt: promptText,
    negativePrompt: negativePromptText,
    lastGenerationPrompt: typeof params?.prompt === "string" ? params.prompt : undefined,
    lastGenerationNegativePrompt:
      typeof params?.negativePrompt === "string" ? params.negativePrompt : undefined,
    lastGenerationProvenance: parsePromptProvenance(params?.promptProvenance),
  });

  const onPromptHistoryChange = useCallback(
    (history: PromptHistoryEntry[], index: number) => {
      setPromptHistoryState(resolvedStoreKey, { history, index });
      const entry = history[index];
      if (!entry) return;
      const changes: FeatureFieldChange<FeatureSchemaMap[TId]>[] = [];
      if ("prompt" in form.values) {
        changes.push({
          kind: "textarea",
          fieldId: "prompt",
          dataKey: "prompt",
          value: entry.text,
        } as FeatureFieldChange<FeatureSchemaMap[TId]>);
      }
      if ("negativePrompt" in form.values && entry.negativeText !== undefined) {
        changes.push({
          kind: "textarea",
          fieldId: "negativePrompt",
          dataKey: "negativePrompt",
          value: entry.negativeText,
        } as FeatureFieldChange<FeatureSchemaMap[TId]>);
      }
      if (changes.length > 0) {
        form.updateField(...changes);
      }
    },
    [form, resolvedStoreKey, setPromptHistoryState],
  );

  const enhance = useManualEnhance({
    featureId: definition.id,
    storeKey: resolvedStoreKey,
    customLoraId,
    values: form.values as ManualEnhancePromptValues,
    promptHistory,
    historyIndex,
    onPromptHistoryChange,
    isCreating: form.isCreating,
  });

  const onReset = useCallback(() => {
    setPromptHistoryState(resolvedStoreKey, { history: [], index: -1 });
    setNagStrength(resolvedStoreKey, DEFAULT_NAG_STRENGTH);
    form.reset();
  }, [form, resolvedStoreKey, setNagStrength, setPromptHistoryState]);

  // The row shows only while a negative prompt is typed and NAG can run: otherwise the knobs
  // do nothing. What it shows is what Generate sends.
  const showNagStrength =
    enhance.showNagStrengthControls &&
    enhance.nagUnavailableReason == null &&
    (negativePromptText ?? "").trim() !== "";
  const generateNow = useCallback(() => {
    return form.generateWithProvenance(
      enhance.promptProvenance,
      showNagStrength ? nagStrength : null,
    );
  }, [enhance.promptProvenance, form, nagStrength, showNagStrength]);
  const generateNowRef = useRef(generateNow);
  generateNowRef.current = generateNow;
  const generateEnabledRef = useRef(generateEnabled);
  generateEnabledRef.current = generateEnabled;
  const proceed = useCallback(() => {
    const run = () => {
      if (!generateEnabledRef.current) return;
      generateNowRef.current();
    };
    if (!generateEnabledRef.current) return;
    if (interceptGenerate) {
      interceptGenerate(run);
      return;
    }
    run();
  }, [interceptGenerate]);
  const { budgetRiskConfirm } = useExploreRuntime();
  const budgetIssue = confirmableBudgetIssue(form.issues);
  const [pendingRisk, setPendingRisk] = useState<typeof budgetIssue>(null);

  useEffect(() => {
    if (generateEnabled) return;
    setPendingRisk(null);
  }, [generateEnabled]);

  const onGenerate = useCallback(() => {
    if (!generateEnabled || enhance.isEnhancing) return;
    if (budgetIssue != null && !(budgetRiskConfirm?.skip ?? false)) {
      setPendingRisk(budgetIssue);
      return;
    }
    proceed();
  }, [budgetIssue, budgetRiskConfirm, enhance.isEnhancing, generateEnabled, proceed]);

  const formModel = useMemo(
    () =>
      enhance.nagUnavailableReason == null
        ? form.form
        : {
            ...form.form,
            fields: form.form.fields.map((field) =>
              field.kind === "textarea" && field.id === "negativePrompt"
                ? { ...field, disabledReason: enhance.nagUnavailableReason }
                : field,
            ),
          },
    [enhance.nagUnavailableReason, form.form],
  );

  const negativePromptAccessory = showNagStrength ? (
      <NagStrengthControl
        value={nagStrength}
        onChange={(next) => setNagStrength(resolvedStoreKey, next)}
      />
    ) : null;

  return (
    <>
    <FeatureFormView
      form={formModel}
      values={form.values}
      onFieldChange={form.updateField}
      revealedIssues={form.revealedIssues}
      onReset={onReset}
      onGenerate={onGenerate}
      generateLabel={generateLabel}
      generateHint={generateHint ?? generateHintForTier(budgetIssue?.tier)}
      isDownloadBusy={generateBusy}
      onCancelDownload={onCancelGenerate}
      showManualEnhance={enhance.showManualEnhance}
      canEnhance={enhance.canEnhance}
      isEnhancing={enhance.isEnhancing}
      enhanceError={enhance.enhanceError}
      onEnhance={enhance.runEnhance}
      enhanceLabel={enhanceActionLabel(enhance.enhanceProvider, enhance.isEnhancing)}
      enhanceProvider={enhance.enhanceProvider}
      canToggleEnhanceProvider={enhance.canToggleEnhanceProvider}
      onEnhanceProviderChange={enhance.setEnhanceProvider}
      canGenerate={form.canGenerate && !enhance.isEnhancing && generateEnabled}
      isUnavailable={form.isUnavailable}
      isCreating={form.isCreating}
      actionError={actionErrorOverride ?? form.actionError}
      pageId={resolvedStoreKey}
      generations={form.generations}
      isGenerationsLoading={form.isGenerationsLoading}
      isGenerationsFetching={form.isGenerationsFetching}
      generationsError={form.generationsError}
      onCancel={form.cancel}
      onRetry={form.retry}
      onDelete={form.remove}
      ResultFrameAdapter={ResultFrameAdapter}
      EmptyStateAdapter={EmptyStateAdapter}
      presentation={presentation}
      audioTrimCapSeconds={audioTrimCapSeconds}
      audioTrimMinSeconds={audioTrimMinSeconds}
      videoRange={videoRange}
      onPatchValues={form.patchValues}
      negativePromptAccessory={negativePromptAccessory}
    />
    {pendingRisk != null ? (
      <RiskyGenerateDialog
        tier={pendingRisk.tier}
        message={pendingRisk.message}
        allowDontAskAgain={budgetRiskConfirm != null}
        onAdjust={() => setPendingRisk(null)}
        onConfirm={({ dontAskAgain }) => {
          if (dontAskAgain) budgetRiskConfirm?.setSkip(true);
          setPendingRisk(null);
          proceed();
        }}
      />
    ) : null}
    </>
  );
}
