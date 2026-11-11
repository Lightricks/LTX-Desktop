import { useEffect, useMemo, type ComponentType } from "react";
import type { UseMutationResult, UseQueryResult } from "@tanstack/react-query";

import {
  selectFreshPreflight,
  selectTrimMaxSeconds,
} from "../../lib/editPreflight";
import type { IcLoraVideoRecipeId } from "@/lib/ic-lora-recipes";

import { useAsset } from "../../hooks/useAsset";
import { useFeatureFlags } from "../../hooks/useFeatureFlags";
import { useExploreRuntime } from "../../runtime/ExploreRuntime";
import { useEditPreflight } from "../../hooks/useEditPreflight";
import { useGenerations } from "../../hooks/useGenerations";
import { useVideoGenerationModelSpecs } from "../../hooks/useVideoGenerationModelSpecs";
import type { Generation } from "../../lib/resultsFeedModel";
import {
  type FeatureSchemaMap,
  useFeatureFormStore,
} from "../../stores/featureFormStore";
import type { AssetRef, FeatureDefinition, FeatureValues } from "./types";
import {
  editResolutionPreflightCandidates,
  type VideoInputContext,
} from "./definitions/videoFeature";
import { assetDurationSeconds } from "./fields/prepareDurationLimitedMediaImport.ts";
import { FeatureFormScreen } from "./FeatureFormScreen";
import type {
  FeatureFormPresentationAdapter,
  VideoAssetRangeBinding,
} from "./FeatureFormView";
import { ResultFrame } from "./results/ResultFrame";

type VideoFeatureId = "retake" | "extend" | IcLoraVideoRecipeId;

type VideoToVideoSchema = FeatureSchemaMap[VideoFeatureId];

/** Selection slice the preflight matrix needs. Durations include the minimum. */
export type VideoPreflightSelection = {
  startTime: number;
  mode: "start" | "end";
  durations: number[];
};

type SeedPlanInput = {
  hasStoredValues: boolean;
  generationsReady: boolean;
  generationsFailed: boolean;
  lastGenerationSpec: unknown;
  hydratedVideoName: string | null;
  hydratedVideoPending: boolean;
};

const EMPTY_VIDEO_CONTEXT: VideoInputContext = {
  specs: null,
  seed: null,
  videoDurationSeconds: null,
  videoDurationPending: false,
  videoWidth: null,
  videoHeight: null,
  videoFps: null,
};

export function useVideoToVideoFeature<
  TSchema extends VideoToVideoSchema,
  TRequest,
>({
  featureId,
  definition,
  presentation,
  EmptyStateAdapter,
  createGeneration,
  useSeed,
  planSeed,
  isPackagedSeedName,
  writeAppliedRevision,
  seedValues,
  videoRange,
  selectPreflightInput,
}: {
  featureId: VideoFeatureId;
  definition: FeatureDefinition<TSchema, TRequest, VideoInputContext, VideoFeatureId>;
  presentation: FeatureFormPresentationAdapter<TSchema>;
  EmptyStateAdapter: ComponentType;
  createGeneration: UseMutationResult<Generation, Error, TRequest>;
  useSeed: (options?: { enabled?: boolean }) => UseQueryResult<AssetRef>;
  planSeed: (input: SeedPlanInput) => "wait" | "skip" | "ingest";
  isPackagedSeedName: (name: string | null | undefined) => boolean;
  writeAppliedRevision: () => void;
  seedValues: (context: VideoInputContext) => FeatureValues<TSchema>;
  videoRange?: VideoAssetRangeBinding<TSchema>;
  // Edit-only: retake/extend pass a selector so the hook fetches advisory
  // verdicts; IC-LoRA recipe screens omit it and preflight stays null.
  selectPreflightInput?: (
    values: FeatureValues<TSchema> | undefined,
  ) => VideoPreflightSelection;
}) {
  const specsQuery = useVideoGenerationModelSpecs();
  const { api } = useExploreRuntime();
  const flagsQuery = useFeatureFlags(api);
  // Only retake and extend read this flag. IC-LoRA recipes ignore it.
  const editsFeature = featureId === "retake" || featureId === "extend";
  const highResVideoToVideo = editsFeature && flagsQuery.data?.highResVideoToVideo === true;
  const hasStoredValues = useFeatureFormStore((state) =>
    state.hasValues(featureId),
  );
  const getValues = useFeatureFormStore((state) => state.getValues);
  const setStoreValues = useFeatureFormStore((state) => state.setValues);
  const generationsQuery = useGenerations(featureId);
  const lastGenerationSpec = generationsQuery.data?.[0]?.spec;
  const storedVideoId = useFeatureFormStore(
    (state) => state.getValues(featureId)?.video?.assetId ?? null,
  );
  const generationVideoId = hasStoredValues
    ? null
    : definition.fromGeneration(lastGenerationSpec, EMPTY_VIDEO_CONTEXT)
        ?.video?.assetId ?? null;
  const videoAssetId = storedVideoId ?? generationVideoId;
  const videoAssetQuery = useAsset(videoAssetId);
  const videoDuration = videoAssetQuery.data
    ? assetDurationSeconds(videoAssetQuery.data)
    : null;
  const videoMeta =
    videoAssetQuery.data?.metadata.mediaType === "video"
      ? videoAssetQuery.data.metadata.metadata
      : null;
  const videoWidth = videoMeta != null && videoMeta.width > 0 ? videoMeta.width : null;
  const videoHeight = videoMeta != null && videoMeta.height > 0 ? videoMeta.height : null;
  const videoFps = videoMeta != null && videoMeta.fps != null && videoMeta.fps > 0 ? videoMeta.fps : null;
  const videoDurationPending = videoAssetId != null && videoAssetQuery.isPending;
  const seedPlan = planSeed({
    hasStoredValues,
    generationsReady: generationsQuery.isSuccess,
    generationsFailed: generationsQuery.isError,
    lastGenerationSpec,
    hydratedVideoName: videoAssetQuery.data?.name ?? null,
    hydratedVideoPending: videoAssetId != null && videoAssetQuery.isPending,
  });
  const seedQuery = useSeed({ enabled: seedPlan === "ingest" });

  useEffect(() => {
    if (seedPlan !== "ingest" || seedQuery.data == null) return;
    const current = getValues(featureId);
    if (
      current?.video != null &&
      current.video.assetId !== seedQuery.data.assetId &&
      !isPackagedSeedName(videoAssetQuery.data?.name)
    ) {
      return;
    }
    const next = {
      ...(current ??
        seedValues({
          specs: specsQuery.data ?? null,
          seed: seedQuery.data,
          videoDurationSeconds: videoDuration,
          videoDurationPending,
          videoWidth,
          videoHeight,
          videoFps,
        })),
      video: seedQuery.data,
    };
    if (current?.video?.assetId !== next.video.assetId) {
      setStoreValues(featureId, next);
    }
    writeAppliedRevision();
  }, [
    featureId,
    getValues,
    isPackagedSeedName,
    seedPlan,
    seedQuery.data,
    seedValues,
    setStoreValues,
    specsQuery.data,
    videoAssetQuery.data?.name,
    videoDuration,
    videoDurationPending,
    videoFps,
    videoHeight,
    videoWidth,
    writeAppliedRevision,
  ]);

  const storedValues = useFeatureFormStore((state) =>
    state.getValues(featureId),
  );
  const preflightSelection = selectPreflightInput?.(
    storedValues as FeatureValues<TSchema> | undefined,
  );
  const preflightCandidates = useMemo(
    () =>
      videoWidth == null || videoHeight == null
        ? []
        : editResolutionPreflightCandidates(videoWidth, videoHeight, highResVideoToVideo),
    [videoWidth, videoHeight, highResVideoToVideo],
  );
  const preflightResult = useEditPreflight({
    // Without a selector the request stays disabled (assetId null, no
    // durations), so the fallback feature value below never fires.
    feature:
      featureId === "retake" || featureId === "extend" ? featureId : "retake",
    assetId: preflightSelection == null ? null : videoAssetId,
    startTime: preflightSelection?.startTime ?? 0,
    mode: preflightSelection?.mode ?? "end",
    candidates: preflightCandidates,
    durations: preflightSelection?.durations ?? [],
  });
  const preflight = selectFreshPreflight(preflightResult, videoAssetId);

  const context = useMemo<VideoInputContext>(
    () => ({
      specs: specsQuery.data,
      seed: seedQuery.data ?? null,
      videoDurationSeconds: videoDuration,
      videoDurationPending,
      videoWidth,
      videoHeight,
      preflight,
      videoFps,
      highResVideoToVideo,
    }),
    [
      highResVideoToVideo,
      seedQuery.data,
      specsQuery.data,
      videoDuration,
      videoDurationPending,
      videoFps,
      videoHeight,
      videoWidth,
      preflight,
    ],
  );

  // The form normalizes a stored resolution against the offered cells. It must not do that
  // before the flag is known, or a saved 2160p source would fall back on every start.
  const storedResolution = storedValues?.resolution ?? null;
  const effectiveVideoRange = useMemo(() => {
    if (videoRange == null || storedResolution == null) return videoRange;
    const max = selectTrimMaxSeconds(
      preflight,
      storedResolution,
      videoRange.minDuration,
    );
    if (max == null) return videoRange;
    return { ...videoRange, maxDurationSeconds: max };
  }, [videoRange, storedResolution, preflight]);

  const contextReady =
    specsQuery.isSuccess &&
    (!editsFeature || flagsQuery.isSuccess) &&
    (seedPlan === "skip" ||
      (seedPlan === "wait" && (hasStoredValues || seedQuery.isSuccess)) ||
      (seedPlan === "ingest" && seedQuery.isSuccess));
  const resourceError =
    specsQuery.error ??
    (editsFeature ? flagsQuery.error : null) ??
    (seedPlan === "ingest" ? seedQuery.error : null) ??
    (hasStoredValues ? null : generationsQuery.error);

  return {
    // Bridge: this wrapper is generic over one video schema, the shared
    // screen is typed for the retake|extend|IC-LoRA recipe union. TSchema is
    // always instantiated as one union member at each call site.
    definition: definition as unknown as FeatureDefinition<
      VideoToVideoSchema,
      TRequest,
      VideoInputContext,
      VideoFeatureId
    >,
    context,
    contextReady,
    generateEnabled: specsQuery.budgetSettingsReady,
    resourceError,
    createGeneration,
    EmptyStateAdapter,
    presentation: presentation as unknown as FeatureFormPresentationAdapter<VideoToVideoSchema>,
    videoRange: effectiveVideoRange as unknown as
      | VideoAssetRangeBinding<VideoToVideoSchema>
      | undefined,
  };
}

export function VideoToVideoFeatureScreen<
  TSchema extends VideoToVideoSchema,
  TRequest,
>(
  props: Parameters<typeof useVideoToVideoFeature<TSchema, TRequest>>[0],
) {
  const screen = useVideoToVideoFeature(props);
  if (!screen.contextReady && screen.resourceError == null) {
    return null;
  }
  return <FeatureFormScreen {...screen} ResultFrameAdapter={ResultFrame} />;
}
