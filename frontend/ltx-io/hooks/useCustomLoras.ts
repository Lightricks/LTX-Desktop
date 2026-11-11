import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { components } from "@/generated/backend-openapi";
import { CUSTOM_LORA_HOME_FEATURE } from "@/lib/custom-lora";

import { unwrapApiResult } from "../lib/unwrapApiResult";
import { type ExploreApi, useExploreRuntime } from "../runtime/ExploreRuntime";
import { recentCustomLoraIds } from "../screens/Feature/definitions/customLora";

import { fetchGenerations, generationQueryKeys } from "./generationQueryKeys";

type CustomLoraImportRequest = components["schemas"]["CustomLoraImportRequest"];

export const CUSTOM_LORAS_QUERY_KEY = ["custom-loras"] as const;

const IMPORT_POLL_INTERVAL_MS = 1000;
// Three failed polls in a row means the backend is gone. Stop and show an error.
const IMPORT_POLL_MAX_FAILURES = 3;

type ListApi = Pick<ExploreApi, "listCustomLoras">;
type RemoveApi = Pick<ExploreApi, "deleteCustomLora" | "deleteCustomLoraWeights">;

/**
 * The saved LoRAs. Takes the bound client, so Settings (outside the Explore runtime) and the Home
 * screens share one cache entry.
 */
export function useCustomLorasWith(api: ListApi, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: CUSTOM_LORAS_QUERY_KEY,
    queryFn: async () => unwrapApiResult(await api.listCustomLoras()).loras,
    enabled: options?.enabled ?? true,
  });
}

export function useCustomLoras(options?: { enabled?: boolean }) {
  const { api } = useExploreRuntime();
  return useCustomLorasWith(api, options);
}

/**
 * The saved LoRAs the user ran, newest run first. It reads the list of Custom LoRA runs, the same
 * list the results feed uses, so a new run moves its LoRA to the top without a second request.
 */
export function useRecentCustomLoraIds(): readonly string[] {
  const { api } = useExploreRuntime();
  const query = useQuery({
    queryKey: generationQueryKeys.list(CUSTOM_LORA_HOME_FEATURE.id),
    queryFn: () => fetchGenerations(api, CUSTOM_LORA_HOME_FEATURE.id),
  });
  const generations = query.data;
  return useMemo(() => recentCustomLoraIds(generations ?? []), [generations]);
}

/** Remove the whole LoRA: weights, entry, and its place in the sidebar. */
export function useRemoveCustomLoraWith(api: RemoveApi) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (customLoraId: string) =>
      unwrapApiResult(await api.deleteCustomLora(customLoraId)),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: CUSTOM_LORAS_QUERY_KEY });
    },
  });
}

export function useRemoveCustomLora() {
  const { api } = useExploreRuntime();
  return useRemoveCustomLoraWith(api);
}

/** Delete the weights only. The LoRA stays listed and asks for the weights again. */
export function useDeleteCustomLoraWeightsWith(api: RemoveApi) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (customLoraId: string) =>
      unwrapApiResult(await api.deleteCustomLoraWeights(customLoraId)),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: CUSTOM_LORAS_QUERY_KEY });
    },
  });
}

export type CustomLoraImportState =
  | { status: "idle" }
  | { status: "running"; customLoraId: string; progress: number }
  | { status: "error"; message: string; code?: string; customLoraId?: string };

type CustomLoraRestoreRequest = components["schemas"]["CustomLoraRestoreRequest"];

type ImportApi = Pick<
  ExploreApi,
  | "importCustomLora"
  | "restoreCustomLoraWeights"
  | "getCustomLoraImportProgress"
  | "getCustomLoraImportActive"
>;

/**
 * One import (or one weights restore) at a time, the same as the backend. The state lives with
 * the screen. On mount the hook picks up an import that is already running (for example after a
 * screen change).
 */
export function useCustomLoraImport(
  api: ImportApi,
  onImported: (customLoraId: string) => void,
) {
  const queryClient = useQueryClient();
  const [state, setState] = useState<CustomLoraImportState>({ status: "idle" });
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const generationRef = useRef(0);
  const onImportedRef = useRef(onImported);
  onImportedRef.current = onImported;

  const stopPolling = useCallback(() => {
    generationRef.current += 1;
    if (pollRef.current != null) clearInterval(pollRef.current);
    pollRef.current = null;
  }, []);

  const poll = useCallback(
    (sessionId: string, customLoraId: string, initialProgress: number) => {
      stopPolling();
      const generation = generationRef.current;
      setState({ status: "running", customLoraId, progress: initialProgress });
      let inFlight = false;
      let failures = 0;
      const tick = async () => {
        if (inFlight || generation !== generationRef.current) return;
        inFlight = true;
        try {
          const result = await api.getCustomLoraImportProgress({ sessionId });
          if (generation !== generationRef.current) return;
          if (!result.ok) {
            failures += 1;
            if (failures >= IMPORT_POLL_MAX_FAILURES) {
              stopPolling();
              setState({
                status: "error",
                message: "Lost contact with the import.",
                code: "IMPORT_POLL_LOST",
                customLoraId,
              });
            }
            return;
          }
          failures = 0;
          const { status, progress, error } = result.data;
          if (status === "complete") {
            stopPolling();
            setState({ status: "idle" });
            await queryClient.invalidateQueries({ queryKey: CUSTOM_LORAS_QUERY_KEY });
            onImportedRef.current(result.data.custom_lora_id ?? customLoraId);
            return;
          }
          if (status === "error") {
            stopPolling();
            setState({
              status: "error",
              message: error ?? "The LoRA could not be added.",
              customLoraId,
            });
            return;
          }
          setState({ status: "running", customLoraId, progress });
        } finally {
          inFlight = false;
        }
      };
      pollRef.current = setInterval(() => {
        void tick();
      }, IMPORT_POLL_INTERVAL_MS);
      void tick();
    },
    [api, queryClient, stopPolling],
  );

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const result = await api.getCustomLoraImportActive();
      if (cancelled || !result.ok) return;
      const { session_id, custom_lora_id, progress } = result.data;
      if (session_id != null && custom_lora_id != null) {
        poll(session_id, custom_lora_id, progress ?? 0);
      }
    })();
    return () => {
      cancelled = true;
      stopPolling();
    };
  }, [api, poll, stopPolling]);

  const start = useCallback(
    async (request: CustomLoraImportRequest): Promise<string | null> => {
      setState({ status: "running", customLoraId: "", progress: 0 });
      const result = await api.importCustomLora(request);
      if (!result.ok) {
        setState({
          status: "error",
          message: result.error.message,
          code: result.error.code,
        });
        return null;
      }
      poll(result.data.sessionId, result.data.custom_lora_id, 0);
      return result.data.custom_lora_id;
    },
    [api, poll],
  );

  /** Put the weights of a saved LoRA back: from its saved link, or from `source_path`. */
  const restore = useCallback(
    async (customLoraId: string, request: CustomLoraRestoreRequest) => {
      setState({ status: "running", customLoraId, progress: 0 });
      const result = await api.restoreCustomLoraWeights(customLoraId, request);
      if (!result.ok) {
        setState({
          status: "error",
          message: result.error.message,
          code: result.error.code,
          customLoraId,
        });
        return;
      }
      poll(result.data.sessionId, result.data.custom_lora_id, 0);
    },
    [api, poll],
  );

  const dismissError = useCallback(() => {
    setState((current) => (current.status === "error" ? { status: "idle" } : current));
  }, []);

  return { state, start, restore, dismissError };
}
