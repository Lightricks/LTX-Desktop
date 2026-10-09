import { useEffect, useMemo, type ComponentType } from "react";
import type { UseMutationResult, UseQueryResult } from "@tanstack/react-query";

import type { IcLoraRecipeId } from "@/lib/ic-lora-recipes";

import { useAsset } from "../../hooks/useAsset";
import { useGenerations } from "../../hooks/useGenerations";
import { useVideoGenerationModelSpecs } from "../../hooks/useVideoGenerationModelSpecs";
import type { Generation } from "../../lib/resultsFeedModel";
import {
  type FeatureSchemaMap,
  useFeatureFormStore,
} from "../../stores/featureFormStore";
import type { AssetRef, FeatureDefinition, FeatureValues } from "./types";
import type { VideoInputContext } from "./definitions/videoFeature";
import { assetDurationSeconds } from "./fields/prepareDurationLimitedMediaImport.ts";
import { FeatureFormScreen } from "./FeatureFormScreen";
import type {
  FeatureFormPresentationAdapter,
  VideoAssetRangeBinding,
} from "./FeatureFormView";
import { ResultFrame } from "./results/ResultFrame";

type VideoFeatureId = "retake" | "extend" | IcLoraRecipeId;

type VideoToVideoSchema = FeatureSchemaMap[VideoFeatureId];

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
}) {
  const specsQuery = useVideoGenerationModelSpecs();
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

  const context = useMemo<VideoInputContext>(
    () => ({
      specs: specsQuery.data,
      seed: seedQuery.data ?? null,
      videoDurationSeconds: videoDuration,
      videoDurationPending,
      videoWidth,
      videoHeight,
      videoFps,
    }),
    [
      seedQuery.data,
      specsQuery.data,
      videoDuration,
      videoDurationPending,
      videoFps,
      videoHeight,
      videoWidth,
    ],
  );

  const contextReady =
    specsQuery.isSuccess &&
    (seedPlan === "skip" ||
      (seedPlan === "wait" && (hasStoredValues || seedQuery.isSuccess)) ||
      (seedPlan === "ingest" && seedQuery.isSuccess));
  const resourceError =
    specsQuery.error ??
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
    resourceError,
    createGeneration,
    EmptyStateAdapter,
    presentation: presentation as unknown as FeatureFormPresentationAdapter<VideoToVideoSchema>,
    videoRange: videoRange as unknown as
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
