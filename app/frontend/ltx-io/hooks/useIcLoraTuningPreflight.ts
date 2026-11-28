import { useQuery } from "@tanstack/react-query";
import { useDeferredValue } from "react";

import type { components } from "@/generated/backend-openapi";
import type { IcLoraRecipeId } from "@/lib/ic-lora-recipes";

import { unwrapApiResult } from "../lib/unwrapApiResult";
import { useExploreRuntime } from "../runtime/ExploreRuntime";

type CreateBody = components["schemas"]["CreateIcLoraRecipeRequest"];
type TuningPreflight = components["schemas"]["IcLoraTilePreflightResponse"];
type PreflightRequest = components["schemas"]["IcLoraTilePreflightRequest"];
type TilePreset = components["schemas"]["IcLoraTilePreflightCell"]["tileSize"];

const TILE_CANDIDATES: TilePreset[] = ["standard", "large", "extra-large"];

/**
 * Advisory verdicts for each tile size, chunk length and carry of the current form. The prompt
 * and the overlap do not change a verdict, so the request sends neither and typing or sliding
 * does not refetch.
 * A new request never shows the verdicts of the old one: a stale refusal would reset a valid pick.
 * Null while pending, errored or disabled. Unknown always means allowed.
 */
export function useIcLoraTuningPreflight(
  recipeId: IcLoraRecipeId,
  body: CreateBody | null,
  windows: readonly number[],
  carries: readonly number[],
): TuningPreflight | null {
  const { api, budgetMode } = useExploreRuntime();
  const deferred = useDeferredValue(body);
  const request: PreflightRequest | null =
    deferred == null
      ? null
      : {
          // The budget prices the tile size, window and carry, never the overlap, so a slider
          // step must not send a new request.
          params: { ...deferred.params, prompt: "", tilingOverlap: undefined },
          inputs: deferred.inputs,
          tileSizes: [...TILE_CANDIDATES],
          maxWindows: [...windows],
          carries: [...carries],
        };
  const query = useQuery({
    queryKey: [
      "ic-lora-tuning-preflight",
      recipeId,
      budgetMode ?? "server",
      JSON.stringify(request),
    ],
    queryFn: async () => {
      if (request == null) {
        throw new Error("preflight request is missing");
      }
      return unwrapApiResult(await api.preflightIcLoraRecipe(recipeId, request));
    },
    enabled: request != null,
    retry: false,
    staleTime: 30_000,
  });
  return request == null ? null : (query.data ?? null);
}
