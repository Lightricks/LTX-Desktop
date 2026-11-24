import { useEffect, useLayoutEffect, useRef, useState } from "react";

import type { ExploreAsset } from "@/lib/explore-contract";

import { useCreateTextToImage } from "../../hooks/useCreateTextToImage";
import { useGenerationDetail } from "../../hooks/useGenerationDetail";
import { cancelGenerationQuietly } from "../../lib/cancelGenerationQuietly";
import { isInFlightGeneration } from "../../lib/resultsFeedModel";
import { useExploreRuntime } from "../../runtime/ExploreRuntime";

import type { CreateTextToImageRequest } from "./imageGeneratorRequest";

/**
 * The image job this generator owns.
 *
 * Closing the generator stops a job that is still queued or running. A job nobody waits for
 * would block the next video and, with fal, spend API credit. A finished image stays in Assets.
 * The live job id sits in a ref that a layout effect updates, so a discarded render cannot
 * cancel the wrong job.
 */
export function useOwnedGeneration() {
  const { api } = useExploreRuntime();
  const create = useCreateTextToImage();
  const [generationId, setGenerationId] = useState<string | null>(null);
  const generation = useGenerationDetail(generationId);

  const generationInFlight =
    generationId !== null &&
    (generation === undefined || isInFlightGeneration(generation));
  const inFlight = create.isPending || generationInFlight;

  const succeeded = generation?.status === "succeeded" ? generation.outputs[0] : undefined;
  // The last image stays until a new one succeeds. A failed or cancelled regeneration leaves
  // the shown image available to use. State adjusted during render, not a ref read.
  const [lastOutput, setLastOutput] = useState<ExploreAsset | undefined>(undefined);
  if (succeeded && succeeded.id !== lastOutput?.id) setLastOutput(succeeded);
  const output = succeeded ?? lastOutput;

  const liveGenerationId = useRef<string | null>(null);
  useLayoutEffect(() => {
    liveGenerationId.current = generationInFlight ? generationId : null;
  }, [generationId, generationInFlight]);

  const unmounted = useRef(false);
  useEffect(() => {
    unmounted.current = false;
    return () => {
      unmounted.current = true;
      const id = liveGenerationId.current;
      if (id !== null) cancelGenerationQuietly(api, id);
    };
  }, [api]);

  const start = (body: CreateTextToImageRequest) => {
    // mutateAsync, because mutate callbacks do not run after the modal unmounts.
    // The create error shows through create.isError.
    create
      .mutateAsync(body)
      .then((created) => {
        // The modal closed while the job was being created.
        if (unmounted.current) cancelGenerationQuietly(api, created.id);
        else setGenerationId(created.id);
      })
      .catch(() => undefined);
  };

  return {
    createError: create.isError ? create.error : null,
    generation,
    generationId,
    // True from the first Generate click. A failed create counts, so its error shows.
    hasStarted: !create.isIdle,
    inFlight,
    output,
    start,
  };
}
