import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";
import { shallow } from "zustand/vanilla/shallow";

import { useGenerationSeed } from "@/hooks/use-generation-seed";
import { shouldRollSeedAfterSuccess } from "@/lib/generation-seed";
import { readCachedGenerationSeed } from "@/lib/generation-seed-sync";
import type { HomeFeatureId } from "@/lib/home-features";
import type { PromptProvenance } from "@/lib/prompt-provenance";
import type { UseMutationResult } from "@tanstack/react-query";
import { useAdvancedControls } from "../../hooks/useAdvancedControls";
import { useCancelGeneration } from "../../hooks/useCancelGeneration";
import { useDeleteGeneration } from "../../hooks/useDeleteGeneration";
import { useGenerations } from "../../hooks/useGenerations";
import { useRetryGeneration } from "../../hooks/useRetryGeneration";
import { injectGenerationSeed } from "../../lib/injectGenerationSeed";
import { injectPromptProvenance } from "../../lib/injectPromptProvenance";
import { injectNagStrength, type NagStrength } from "../../lib/nagStrength";
import { useNagUnavailableReason } from "../../hooks/useNagUnavailableReason";
import type { Generation } from "../../lib/resultsFeedModel";
import {
  type FeatureFormKey,
  type FeatureSchemaMap,
  useFeatureFormStore,
} from "../../stores/featureFormStore";
import { useExploreRuntime } from "../../runtime/ExploreRuntime";
import {
  resolveFeatureFormValues,
  resolveFeatureSeedValues,
  type FeatureDefinition,
  type FeatureFieldChange,
  type FeatureValues,
} from "./types";
import { formWithAdvancedFields, valuesForRun } from "./advancedFields";

function errorMessage(error: unknown): string | null {
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message;
  }
  return null;
}

export function useFeatureForm<
  TId extends HomeFeatureId,
  TCreateBody,
  TContext,
>(
  definition: FeatureDefinition<FeatureSchemaMap[TId], TCreateBody, TContext, TId>,
  context: TContext,
  createGeneration: UseMutationResult<Generation, Error, TCreateBody>,
  contextReady: boolean,
  resourceError: unknown,
  /**
   * Keep only some of the feature's generations. A page for one saved custom LoRA shows only the
   * runs of that LoRA, though every custom LoRA run has the same feature id. Must be stable.
   */
  generationFilter?: (generation: Generation) => boolean,
  storeKey: FeatureFormKey = definition.id,
) {
  const storedValues = useFeatureFormStore((state) =>
    state.getValues(storeKey),
  ) as FeatureValues<FeatureSchemaMap[TId]> | undefined;
  const hasStoredValues = useFeatureFormStore((state) =>
    state.hasValues(storeKey),
  );
  const setStoreValues = useFeatureFormStore((state) => state.setValues);
  // The request scale wins over Settings, so a strength sent while NAG is off for this feature
  // (Settings strength 0, API text encoding, Torch Compile) would turn NAG back on. The screen
  // only passes one while its row shows; this keeps it out even if a caller passes one anyway.
  const nagUnavailableReason = useNagUnavailableReason(definition.id);

  const { api } = useExploreRuntime();
  const advancedControls = useAdvancedControls();
  const queryClient = useQueryClient();
  const { randomize: rollGenerationSeed, ready: seedReady } =
    useGenerationSeed(api);

  const [keepPollingId, setKeepPollingId] = useState<string | null>(null);
  const generationsQuery = useGenerations(definition.id, {
    keepPolling: keepPollingId != null,
  });
  const cancelGeneration = useCancelGeneration();
  const retryGeneration = useRetryGeneration();
  const deleteGeneration = useDeleteGeneration();

  const [revealedIssueIds, setRevealedIssueIds] = useState<Set<string>>(
    () => new Set(),
  );

  const scopedGenerations = useMemo(
    () =>
      generationFilter && generationsQuery.data
        ? generationsQuery.data.filter(generationFilter)
        : generationsQuery.data,
    [generationFilter, generationsQuery.data],
  );

  const values = resolveFeatureFormValues(definition, {
    context,
    contextReady,
    storedValues,
    generationsFetched: generationsQuery.isFetched,
    lastGenerationSpec: scopedGenerations?.[0]?.spec,
  });

  useEffect(() => {
    if (!generationsQuery.data || keepPollingId == null) return;
    if (generationsQuery.data.some((row) => row.id === keepPollingId)) {
      setKeepPollingId(null);
    }
  }, [generationsQuery.data, keepPollingId]);

  useEffect(() => {
    if (!contextReady) return;
    if (!hasStoredValues && !generationsQuery.isFetched) return;

    const lastGeneration = scopedGenerations?.[0];
    const next = resolveFeatureSeedValues(definition, {
      context,
      hasStoredValues,
      storedValues,
      lastGenerationSpec: lastGeneration?.spec,
    });
    if (!shallow(storedValues, next)) {
      setStoreValues(storeKey, next);
    }
  }, [
    definition,
    context,
    contextReady,
    scopedGenerations,
    generationsQuery.isFetched,
    hasStoredValues,
    setStoreValues,
    storedValues,
    storeKey,
  ]);

  const fullForm = useMemo(
    () => definition.form(values, context),
    [context, definition, values],
  );
  const form = useMemo(
    () => formWithAdvancedFields(fullForm, advancedControls),
    [advancedControls, fullForm],
  );

  const issues = useMemo(
    () => definition.validate(values, context),
    [context, definition, values],
  );

  const revealedIssues = useMemo(
    () =>
      issues.filter(
        (issue) => issue.alwaysRevealed === true || revealedIssueIds.has(issue.id),
      ),
    [issues, revealedIssueIds],
  );

  // The Seed field is disabled until the shared seed loads, so generating earlier would use a
  // seed the user never saw.
  const canGenerate =
    contextReady &&
    seedReady &&
    generationsQuery.isFetched &&
    (definition.isReady?.(values, context) ?? true);
  const isUnavailable =
    contextReady && (definition.isUnavailable?.(values, context) ?? false);

  const setValues = useCallback(
    (next: FeatureValues<FeatureSchemaMap[TId]>) => {
      const normalized = definition.normalize(next, context);
      setStoreValues(storeKey, normalized);
      setRevealedIssueIds((current) => {
        const remaining = definition.validate(normalized, context);
        const remainingIds = new Set(remaining.map((issue) => issue.id));
        const nextIds = new Set(
          [...current].filter((id) => remainingIds.has(id)),
        );
        return nextIds.size === current.size ? current : nextIds;
      });
    },
    [context, definition, setStoreValues, storeKey],
  );

  const updateField = useCallback(
    (...changes: FeatureFieldChange<FeatureSchemaMap[TId]>[]) => {
      if (changes.length === 0) return;
      let next = values;
      for (const change of changes) {
        next = definition.applyChange(next, change, context);
      }
      setValues(next);
    },
    [context, definition, setValues, values],
  );

  const patchValues = useCallback(
    (patch: Partial<FeatureValues<FeatureSchemaMap[TId]>>) => {
      setValues({ ...values, ...patch });
    },
    [setValues, values],
  );

  const reset = useCallback(() => {
    setValues(definition.resetValues?.(context) ?? definition.defaults);
    setRevealedIssueIds(new Set());
  }, [context, definition, setValues]);

  const generateWithProvenance = useCallback(
    // nagStrength: the strength row the screen shows, or null when it shows none. Null sends
    // the body as the definition built it.
    (promptProvenance: PromptProvenance, nagStrength: NagStrength | null = null): boolean => {
      const submittedSeed = readCachedGenerationSeed(queryClient);
      if (!canGenerate || submittedSeed == null) return false;
      const currentIssues = definition.validate(values, context);
      const blockingIssues = currentIssues.filter(
        (issue) => issue.blocksGenerate !== false,
      );
      if (blockingIssues.length > 0) {
        setRevealedIssueIds(new Set(currentIssues.map((issue) => issue.id)));
        return false;
      }
      setRevealedIssueIds(new Set());
      createGeneration.reset();
      cancelGeneration.reset();
      retryGeneration.reset();
      deleteGeneration.reset();
      const built = injectGenerationSeed(
        injectPromptProvenance(
          definition.toCreateBody(
            valuesForRun(fullForm, values, definition.defaults, advancedControls),
            context,
          ) as {
            params: Record<string, unknown>;
          },
          promptProvenance,
        ),
        submittedSeed.seed,
      );
      const body =
        nagStrength && nagUnavailableReason == null
          ? injectNagStrength(built, nagStrength)
          : built;
      createGeneration.mutate(body as TCreateBody, {
        onSuccess: (generation) => {
          setKeepPollingId(generation.id);
          const currentSeed = readCachedGenerationSeed(queryClient)?.seed;
          if (
            currentSeed != null &&
            shouldRollSeedAfterSuccess(submittedSeed, currentSeed)
          ) {
            rollGenerationSeed();
          }
        },
      });
      return true;
    },
    [
      canGenerate,
      cancelGeneration,
      context,
      createGeneration,
      definition,
      deleteGeneration,
      fullForm,
      advancedControls,
      nagUnavailableReason,
      queryClient,
      retryGeneration,
      rollGenerationSeed,
      values,
    ],
  );

  // The feed is virtualized, so it takes every generation. A stable empty list keeps its memo.
  const allGenerations = useMemo(
    () => scopedGenerations ?? [],
    [scopedGenerations],
  );

  const generationsError =
    generationsQuery.isError && allGenerations.length === 0
      ? errorMessage(generationsQuery.error)
      : null;

  const actionError =
    errorMessage(createGeneration.error) ??
    errorMessage(cancelGeneration.error) ??
    errorMessage(retryGeneration.error) ??
    errorMessage(deleteGeneration.error) ??
    errorMessage(resourceError) ??
    (generationsError ? null : errorMessage(generationsQuery.error));

  return {
    values,
    form,
    issues,
    revealedIssues,
    actionError,
    updateField,
    patchValues,
    reset,
    generateWithProvenance,
    canGenerate,
    isUnavailable,
    isCreating: createGeneration.isPending,
    generations: allGenerations,
    isGenerationsLoading: generationsQuery.isPending,
    isGenerationsFetching: generationsQuery.isFetching,
    generationsError,
    cancel: (generationId: string) => {
      createGeneration.reset();
      retryGeneration.reset();
      deleteGeneration.reset();
      cancelGeneration.mutate(generationId);
    },
    retry: (generationId: string) => {
      createGeneration.reset();
      cancelGeneration.reset();
      deleteGeneration.reset();
      retryGeneration.mutate(generationId, {
        onSuccess: (generation) => setKeepPollingId(generation.id),
      });
    },
    remove: (generationId: string) => {
      createGeneration.reset();
      cancelGeneration.reset();
      retryGeneration.reset();
      deleteGeneration.mutate(generationId);
    },
  };
}
