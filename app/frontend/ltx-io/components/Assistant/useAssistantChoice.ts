import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { unwrapApiResult } from "@/ltx-io/lib/unwrapApiResult";
import {
  type ExploreAssistantLocalModel,
  type ExploreAssistantProvider,
  type ExploreAssistantSettings,
  useExploreRuntime,
} from "@/ltx-io/runtime/ExploreRuntime";

type AssistantChoice = {
  provider: ExploreAssistantProvider;
  localModel: string;
  geminiModel: string;
  hasGeminiApiKey: boolean;
  loadError: string | null;
  loaded: boolean;
  select: ExploreAssistantSettings["select"];
  switching: boolean;
  openSettings: ExploreAssistantSettings["openSettings"] | null;
  settingsName: string;
};

const ASSISTANT_SELECTION_QUERY_KEY = ["assistant", "selection"] as const;
// Desktop's choice is kept current by AppSettingsContext; Remote polls it here.
const SELECTION_REFRESH_MS = 10_000;

export function useAssistantChoice(active: boolean): AssistantChoice {
  const { api, assistantSettings } = useExploreRuntime();
  const queryClient = useQueryClient();
  const remote = useQuery({
    queryKey: ASSISTANT_SELECTION_QUERY_KEY,
    enabled: active && assistantSettings == null,
    queryFn: async () => unwrapApiResult(await api.getAssistantSelection()),
    refetchInterval: SELECTION_REFRESH_MS,
  });
  const remoteSelect = useMutation({
    mutationFn: async ({
      provider,
      localModel,
    }: {
      provider: ExploreAssistantProvider;
      localModel?: ExploreAssistantLocalModel;
    }) => {
      // A read in flight across the save could land after it with the old choice.
      await queryClient.cancelQueries({ queryKey: ASSISTANT_SELECTION_QUERY_KEY });
      const selection = unwrapApiResult(
        await api.setAssistantSelection({ provider, local_model: localModel ?? null }),
      );
      await queryClient.cancelQueries({ queryKey: ASSISTANT_SELECTION_QUERY_KEY });
      queryClient.setQueryData(ASSISTANT_SELECTION_QUERY_KEY, selection);
    },
  });
  if (assistantSettings) {
    return { ...assistantSettings, loadError: null, loaded: true, settingsName: "Settings" };
  }
  return {
    provider: remote.data?.provider ?? "gemini",
    localModel: remote.data?.local_model ?? "",
    geminiModel: remote.data?.gemini_model ?? "",
    hasGeminiApiKey: remote.data?.has_gemini_api_key ?? false,
    loadError: remote.error ? remote.error.message : null,
    loaded: remote.data != null,
    select: (provider, localModel) => remoteSelect.mutateAsync({ provider, localModel }),
    switching: remoteSelect.isPending,
    openSettings: null,
    settingsName: "LTX Desktop Settings",
  };
}
