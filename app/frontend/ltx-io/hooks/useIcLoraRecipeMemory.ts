import { useQuery } from "@tanstack/react-query";
import { useDeferredValue, useMemo } from "react";

import type { components } from "@/generated/backend-openapi";
import { useExploreRuntime } from "../runtime/ExploreRuntime";
import { unwrapApiResult } from "../lib/unwrapApiResult";
import type { IcLoraRecipeMemory } from "../screens/Feature/definitions/icLoraMemory";

export type IcLoraRecipeMemoryInput = components["schemas"]["IcLoraRecipeMemoryRequest"];

/**
 * Memory advice for a Home video recipe: which cells fit before a clip is picked, and the
 * selected cell's ask once it is. Deferred and sticky like the edit preflight, so a cell change
 * keeps the last answer up until the next one lands. Null while pending, errored, or disabled.
 */
export function useIcLoraRecipeMemory(
  recipeId: string,
  input: IcLoraRecipeMemoryInput | null,
): IcLoraRecipeMemory | null {
  const { api, budgetMode } = useExploreRuntime();
  const deferred = useDeferredValue(input);
  const queryKey = useMemo(
    () => ["ic-lora-recipe-memory", budgetMode ?? "server", recipeId, JSON.stringify(deferred)],
    [budgetMode, deferred, recipeId],
  );
  const enabled = deferred != null && deferred.cells.length > 0;
  const query = useQuery({
    queryKey,
    queryFn: async (): Promise<IcLoraRecipeMemory> =>
      unwrapApiResult(
        await api.icLoraRecipeMemory(recipeId, deferred as IcLoraRecipeMemoryInput),
      ),
    enabled,
    retry: false,
    staleTime: 30_000,
    placeholderData: (previous) => previous,
  });
  if (!enabled) return null;
  return query.data ?? null;
}
