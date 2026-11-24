import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback } from "react";

import {
  GENERATION_SEED_POLL_MS,
  GENERATION_SEED_QUERY_KEY,
  GENERATION_SEED_RETRY_MS,
  fetchGenerationSeed,
  readCachedGenerationSeed,
  saveGenerationSeed,
  type GenerationSeedApi,
  type GenerationSeedState,
} from "../lib/generation-seed-sync";
import { randomGenerationSeed } from "../lib/generation-seed";

export function useGenerationSeed(api: GenerationSeedApi): GenerationSeedState & {
  ready: boolean;
  setSeed: (seed: number) => void;
  setLocked: (locked: boolean) => void;
  randomize: () => void;
} {
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: GENERATION_SEED_QUERY_KEY,
    queryFn: () => fetchGenerationSeed(queryClient, api),
    refetchInterval: (query) =>
      query.state.data ? GENERATION_SEED_POLL_MS : GENERATION_SEED_RETRY_MS,
    refetchOnWindowFocus: true,
  });

  const setSeed = useCallback(
    (seed: number) => saveGenerationSeed(queryClient, api, { seed }),
    [api, queryClient],
  );
  const setLocked = useCallback(
    (locked: boolean) => saveGenerationSeed(queryClient, api, { locked }),
    [api, queryClient],
  );
  const randomize = useCallback(() => {
    const current = readCachedGenerationSeed(queryClient);
    saveGenerationSeed(queryClient, api, { seed: randomGenerationSeed(current?.seed) });
  }, [api, queryClient]);

  return {
    seed: query.data?.seed ?? 0,
    locked: query.data?.locked ?? false,
    ready: query.isSuccess,
    setSeed,
    setLocked,
    randomize,
  };
}
