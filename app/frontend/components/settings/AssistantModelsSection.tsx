import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Fragment, useEffect, useLayoutEffect } from "react";

import { useAppSettings } from "@/contexts/AppSettingsContext";
import { Button } from "@/ds/Button/Button";
import {
  assistantModelsQueryKey,
  useAssistantModels,
} from "@/hooks/use-assistant-models";
import { ApiClient, type ApiSuccessOf } from "@/lib/api-client";
import { formatBytes } from "@/lib/format";
import { confirmCatalogModelDelete, matchesCatalogSearch } from "@/lib/model-catalog";

import {
  ModelCatalogActionGroup,
  ModelCatalogCellText,
  ModelCatalogDetailRow,
  ModelCatalogEmpty,
  ModelCatalogInstalledCheck,
  ModelCatalogNameCell,
  ModelCatalogSizeCell,
  ModelCatalogSizeText,
  ModelCatalogTable,
  modelCatalogActionButtonWideClassName,
  modelCatalogDownloadButtonLabel,
  modelCatalogRowClassNames,
} from "./ModelCatalogTable";
import { SettingsHint, SettingsProgressBar, SettingsSubsection } from "./SettingsContent";

type Model = ApiSuccessOf<"listAssistantModels">["models"][number]["model"];

export function AssistantModelsSection({
  active,
  searchQuery = "",
  onSearchHasMatches,
}: {
  active: boolean;
  searchQuery?: string;
  onSearchHasMatches?: (hasMatches: boolean) => void;
}) {
  const query = useAssistantModels(active);
  const { notifyModelsChanged } = useAppSettings();
  const queryClient = useQueryClient();
  useEffect(() => {
    if (query.data?.download?.status === "complete") notifyModelsChanged();
  }, [query.data?.download?.status, notifyModelsChanged]);
  const download = useMutation({
    mutationFn: async (model: Model) => {
      const result = await ApiClient.downloadAssistantModel({ model });
      if (!result.ok) throw new Error(result.error.message);
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: assistantModelsQueryKey }),
  });
  const remove = useMutation({
    mutationFn: async (model: Model) => {
      const result = await ApiClient.deleteAssistantModel({ model });
      if (!result.ok) throw new Error(result.error.message);
    },
    onSuccess: () => notifyModelsChanged(),
    onSettled: () => queryClient.invalidateQueries({ queryKey: assistantModelsQueryKey }),
  });
  const matches = matchesCatalogSearch(
    "assistant local qwen 3.5 9b vision multimodal",
    searchQuery,
  );
  useLayoutEffect(() => onSearchHasMatches?.(matches), [matches, onSearchHasMatches]);
  if (!active || !matches) return null;

  const progress = query.data?.download;
  const busy =
    download.isPending || remove.isPending || progress?.status === "downloading";
  const error =
    remove.error?.message ??
    download.error?.message ??
    query.error?.message ??
    query.data?.error;

  return (
    <SettingsSubsection title="Assistant models">
      {error && <SettingsHint variant="warning">{error}</SettingsHint>}
      {query.isPending ? <ModelCatalogEmpty message="Loading models..." /> : null}
      <ModelCatalogTable>
        {query.data?.models.map((model) => {
          const downloading =
            progress?.model === model.model && progress.status === "downloading";
          const failed = progress?.model === model.model && progress.status === "error";
          return (
            <Fragment key={model.model}>
              <tr className={modelCatalogRowClassNames()}>
                <ModelCatalogNameCell>
                  <div className="flex min-w-0 flex-col">
                    <div className="flex items-center gap-1.5">
                      <ModelCatalogCellText>{model.name}</ModelCatalogCellText>
                      {model.installed && <ModelCatalogInstalledCheck />}
                    </div>
                  </div>
                </ModelCatalogNameCell>
                <ModelCatalogSizeCell>
                  <ModelCatalogSizeText>
                    {formatBytes(model.size_bytes)}
                  </ModelCatalogSizeText>
                </ModelCatalogSizeCell>
                <ModelCatalogActionGroup
                  action={
                    !model.installed ? (
                      <Button
                        appearance="neutral"
                        hierarchy="secondary"
                        size="md"
                        className={modelCatalogActionButtonWideClassName()}
                        label={modelCatalogDownloadButtonLabel(
                          downloading,
                          progress?.progress ?? 0,
                        )}
                        aria-label={
                          downloading
                            ? `Downloading ${model.name}`
                            : `Download ${model.name}`
                        }
                        disabled={busy || !query.data?.available || query.isError}
                        onClick={() => download.mutate(model.model)}
                      />
                    ) : (
                      <Button
                        appearance="neutral"
                        hierarchy="secondary"
                        size="md"
                        className={modelCatalogActionButtonWideClassName()}
                        label="Delete"
                        aria-label={`Delete ${model.name}`}
                        disabled={busy || query.isError}
                        onClick={() => {
                          if (confirmCatalogModelDelete(model.name))
                            remove.mutate(model.model);
                        }}
                      />
                    )
                  }
                />
              </tr>
              {downloading && (
                <ModelCatalogDetailRow>
                  <SettingsProgressBar percent={progress.progress} />
                </ModelCatalogDetailRow>
              )}
              {failed && (
                <ModelCatalogDetailRow>
                  <SettingsHint variant="warning">{progress.error}</SettingsHint>
                </ModelCatalogDetailRow>
              )}
            </Fragment>
          );
        })}
      </ModelCatalogTable>
    </SettingsSubsection>
  );
}
