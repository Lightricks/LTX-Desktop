import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";
import { shallow } from "zustand/vanilla/shallow";

import { useGenerationSeed } from "@/hooks/use-generation-seed";
import { shouldRollSeedAfterSuccess } from "@/lib/generation-seed";
import { readCachedGenerationSeed } from "@/lib/generation-seed-sync";
import type { HomeFeatureId } from "@/lib/home-features";
import type { PromptProvenance } from "@/lib/prompt-provenance";
import type { UseMutationResult } from "@tanstack/react-query";
import { useCancelGeneration } from "../../hooks/useCancelGeneration";
import { useDeleteGeneration } from "../../hooks/useDeleteGeneration";
import { useGenerations } from "../../hooks/useGenerations";
import { useRetryGeneration } from "../../hooks/useRetryGeneration";
import { injectGenerationSeed } from "../../lib/injectGenerationSeed";
import { injectPromptProvenance } from "../../lib/injectPromptProvenance";
import {
  getVisibleGenerations,
  type Generation,
} from "../../lib/resultsFeedModel";
import {
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
) {
  const storedValues = useFeatureFormStore((state) =>
    state.getValues(definition.id),
  );
  const hasStoredValues = useFeatureFormStore((state) =>
    state.hasValues(definition.id),
  );
  const setStoreValues = useFeatureFormStore((state) => state.setValues);

  const { api } = useExploreRuntime();
  const queryClient = useQueryClient();
  const { randomize: rollGenerationSeed, ready: seedReady } =
    useGenerationSeed(api);

  const [keepPollingId, setKeepPollingId] = useState<string | null>(null);
  const generationsQuery = useGenerations(definition.id, {
    keepPolling: keepPollingId != null,
  });
  const cancelGeneration = useCancelGeneration(definition.id);
  const retryGeneration = useRetryGeneration(definition.id);
  const deleteGeneration = useDeleteGeneration(definition.id);

  const [revealedIssueIds, setRevealedIssueIds] = useState<Set<string>>(
    () => new Set(),
  );

  const values = resolveFeatureFormValues(definition, {
    context,
    contextReady,
    storedValues,
    generationsFetched: generationsQuery.isFetched,
    lastGenerationSpec: generationsQuery.data?.[0]?.spec,
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

    const lastGeneration = generationsQuery.data?.[0];
    const next = resolveFeatureSeedValues(definition, {
      context,
      hasStoredValues,
      storedValues,
      lastGenerationSpec: lastGeneration?.spec,
    });
    if (!shallow(storedValues, next)) {
      setStoreValues(definition.id, next);
    }
  }, [
    definition,
    context,
    contextReady,
    generationsQuery.data,
    generationsQuery.isFetched,
    hasStoredValues,
    setStoreValues,
    storedValues,
  ]);

  const form = useMemo(
    () => definition.form(values, context),
    [context, definition, values],
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
      setStoreValues(definition.id, normalized);
      setRevealedIssueIds((current) => {
        const remaining = definition.validate(normalized, context);
        const remainingIds = new Set(remaining.map((issue) => issue.id));
        const nextIds = new Set(
          [...current].filter((id) => remainingIds.has(id)),
        );
        return nextIds.size === current.size ? current : nextIds;
      });
    },
    [context, definition, setStoreValues],
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
    (promptProvenance: PromptProvenance) => {
      const submittedSeed = readCachedGenerationSeed(queryClient);
      if (!canGenerate || submittedSeed == null) return;
      const currentIssues = definition.validate(values, context);
      const blockingIssues = currentIssues.filter(
        (issue) => issue.blocksGenerate !== false,
      );
      if (blockingIssues.length > 0) {
        setRevealedIssueIds(new Set(currentIssues.map((issue) => issue.id)));
        return;
      }
      setRevealedIssueIds(new Set());
      createGeneration.reset();
      cancelGeneration.reset();
      retryGeneration.reset();
      deleteGeneration.reset();
      const body = injectGenerationSeed(
        injectPromptProvenance(
          definition.toCreateBody(values, context) as {
            params: Record<string, unknown>;
          },
          promptProvenance,
        ),
        submittedSeed.seed,
      );
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
    },
    [
      canGenerate,
      cancelGeneration,
      context,
      createGeneration,
      definition,
      deleteGeneration,
      queryClient,
      retryGeneration,
      rollGenerationSeed,
      values,
    ],
  );

  const visibleGenerations = useMemo(
    () => getVisibleGenerations(generationsQuery.data ?? []),
    [generationsQuery.data],
  );

  const generationsError =
    generationsQuery.isError && visibleGenerations.length === 0
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
    generations: visibleGenerations,
    isGenerationsLoading: generationsQuery.isPending,
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
