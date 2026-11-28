import { H3_VARIANTS } from "../../lib/h3-models";
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import type { VideoGenerationModelSpecsResponse } from "../../lib/video-generation-model-specs";
import { hasManagedVideoModels, type AmdPickerContext } from "../lib/amdVideoPicker";
import { unwrapApiResult } from "../lib/unwrapApiResult";
import { useExploreRuntime } from "../runtime/ExploreRuntime";

export function useAmdVideoPicker(specs?: VideoGenerationModelSpecsResponse) {
  const { api, host, preferredLocalVideoRenderer, pairedAmdModels } = useExploreRuntime();
  const enabled = hasManagedVideoModels(specs);
  const desktopEnabled = enabled && host === "desktop";
  const paired = useQuery({
    queryKey: ["paired-amd-models"], enabled: enabled && host === "remote" && !!pairedAmdModels,
    queryFn: pairedAmdModels, refetchInterval: enabled && host === "remote" ? 10000 : false, retry: false,
  });
  const versions = useQuery({
    queryKey: ["ltx-versions"], enabled: desktopEnabled,
    queryFn: async () => unwrapApiResult(await api.getLtxVersions()).versions,
    retry: false,
  });
  const needsComfyModels = desktopEnabled && !!(specs?.comfy_local_model || specs?.kandinsky_local_model);
  const status = useQuery({
    queryKey: ["comfy-models"], enabled: needsComfyModels,
    queryFn: async () => unwrapApiResult(await api.getComfyModels()),
    refetchInterval: needsComfyModels ? 5000 : false,
    retry: false,
  });
  const h3Supported = !!specs?.h3_fl2va_local_model || !!specs?.h3_fl2va_distill_local_model || !!specs?.h3_ref2va_local_model;
  const h3Models = useQuery({
    queryKey: ["h3-models"], enabled: h3Supported && host === "desktop",
    queryFn: async () => unwrapApiResult(await api.getH3Models()),
    refetchInterval: h3Supported && host === "desktop" ? 5000 : false, retry: false,
  });
  const native = useMemo<AmdPickerContext["native"]>(() => {
    if (!enabled) return undefined;
    if (!versions.data) {
      const state = versions.isError || paired.isError ? "error" : "checking";
      return { "ltx-2.3-fast": state, "ltx-2.5-fast": state };
    }
    return {
      "ltx-2.3-fast": versions.data.some(row =>
        (row.model_id === "ltx-2.3-22b-distilled-1.1" || row.model_id === "ltx-2.3-22b-distilled") && row.installed,
      ) ? "ready" : "missing",
      "ltx-2.5-fast": versions.data.some(row => row.model_id === "ltx-2.5-22b-distilled" && row.installed)
        ? "ready" : "missing",
    };
  }, [enabled, versions.data, versions.isError, paired.isError]);
  const activeId = versions.data?.find(row => row.active)?.model_id;
  const amd = useMemo<AmdPickerContext | undefined>(() => enabled ? {
    preferred: preferredLocalVideoRenderer ?? "native",
    desktop: host === "desktop",
    native,
    activeNative: activeId == null ? undefined : activeId.startsWith("ltx-2.3-") ? "ltx-2.3-fast" : "ltx-2.5-fast",
    kandinskyDownloading: status.data?.kandinskyStatus === "downloading",
    status: status.data?.status ?? (status.isError || paired.isError ? "error" : "checking"),
    message: status.data?.message ?? "",
    h3: h3Models.data ? Object.fromEntries(H3_VARIANTS.map(id => [id, h3Models.data!.licenseAcknowledged
      ? h3Models.data!.variants[id] : { status: "missing", message: "Review local model rights in setup." }])) : undefined,
    kandinskyReady: status.data?.kandinskyReady === true,
    kandinskyMessage: status.data?.kandinskyMessage ?? "",
  } : undefined, [enabled, host, native, activeId, preferredLocalVideoRenderer, h3Models.data, status.data, status.isError, paired.isError]);
  return { amd: host === "remote" ? paired.data ?? amd : amd,
    ready: !enabled || (host === "remote" ? !paired.isPending : preferredLocalVideoRenderer != null) };
}
