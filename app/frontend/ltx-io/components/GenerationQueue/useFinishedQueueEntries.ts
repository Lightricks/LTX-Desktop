import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useMemo } from "react";

import {
  generationQueryKeys,
  invalidateAffectedGenerationQueries,
  normalizeQueueSnapshot,
  replaceGenerationQueueCache,
  type ExploreQueueSnapshot,
} from "../../hooks/generationQueryKeys.ts";
import { unwrapApiResult, type ApiResult } from "../../lib/unwrapApiResult.ts";
import type { ExploreApi } from "../../runtime/ExploreRuntime.tsx";

type FinishedQueueApi = Pick<
  ExploreApi,
  | "markGenerationQueueDoneSeen"
  | "dismissGenerationQueueDone"
  | "clearGenerationQueueDone"
  | "clearGenerationQueueFailed"
>;

type FinishedQueueMutation = Promise<
  ApiResult<Parameters<typeof normalizeQueueSnapshot>[0]>
>;

export function useFinishedQueueEntries({
  snapshot,
  api,
}: {
  snapshot: ExploreQueueSnapshot | undefined;
  api: FinishedQueueApi;
}) {
  const queryClient = useQueryClient();
  const doneEntries = snapshot?.done ?? [];
  const failedEntries = snapshot?.failed ?? [];
  const unreadDoneIds = useMemo(
    () => new Set(snapshot?.unseen_ids ?? []),
    [snapshot?.unseen_ids],
  );

  const commit = useCallback(
    (request: FinishedQueueMutation) => {
      void request
        .then((result) => {
          if (!result.ok) {
            invalidateAffectedGenerationQueries(queryClient);
            return;
          }
          replaceGenerationQueueCache(
            queryClient,
            normalizeQueueSnapshot(unwrapApiResult(result)),
          );
        })
        .catch(() => {
          invalidateAffectedGenerationQueries(queryClient);
        });
    },
    [queryClient],
  );

  const writeOptimistic = useCallback(
    async (update: (snapshot: ExploreQueueSnapshot) => ExploreQueueSnapshot) => {
      await queryClient.cancelQueries({ queryKey: generationQueryKeys.queue });
      const current = queryClient.getQueryData<ExploreQueueSnapshot>(
        generationQueryKeys.queue,
      );
      if (current == null) return;
      replaceGenerationQueueCache(
        queryClient,
        update(normalizeQueueSnapshot(current)),
      );
    },
    [queryClient],
  );

  const markDoneRead = useCallback(
    (id: string) => {
      void (async () => {
        await writeOptimistic((snapshot) => ({
          ...snapshot,
          unseen_ids: snapshot.unseen_ids.filter((item) => item !== id),
        }));
        commit(api.markGenerationQueueDoneSeen(id));
      })();
    },
    [api, commit, writeOptimistic],
  );

  const removeDoneEntry = useCallback(
    (id: string) => {
      void (async () => {
        await writeOptimistic((snapshot) => ({
          ...snapshot,
          done: snapshot.done.filter((entry) => entry.generation.id !== id),
          unseen_ids: snapshot.unseen_ids.filter((item) => item !== id),
        }));
        commit(api.dismissGenerationQueueDone(id));
      })();
    },
    [api, commit, writeOptimistic],
  );

  const clearDone = useCallback(() => {
    void (async () => {
      await writeOptimistic((snapshot) => ({
        ...snapshot,
        done: [],
        unseen_ids: [],
      }));
      commit(api.clearGenerationQueueDone());
    })();
  }, [api, commit, writeOptimistic]);

  const clearFailed = useCallback(() => {
    void (async () => {
      await writeOptimistic((snapshot) => ({
        ...snapshot,
        failed: [],
      }));
      commit(api.clearGenerationQueueFailed());
    })();
  }, [api, commit, writeOptimistic]);

  return {
    doneEntries,
    failedEntries,
    unreadDoneIds,
    clearDone,
    clearFailed,
    removeDoneEntry,
    markDoneRead,
  };
}
