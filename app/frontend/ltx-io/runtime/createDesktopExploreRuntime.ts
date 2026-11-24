import {
  EXPLORE_LORA_RECIPE_SEED_FILES,
  type ExploreLoraRecipeSeedId,
  PACKAGED_SEED_KINDS,
  type PackagedSeedKind,
} from "../../../shared/explore-seed-filenames.ts";
import type { EnhanceProvider } from "../../hooks/use-prompt-enhancer-provider";
import { ApiClient } from "../../lib/api-client.ts";
import { type BackendFetch, backendFetch, backendWsUrl } from "../../lib/backend.ts";
import type { BudgetMode } from "../../lib/budget-risk.ts";
import { pathToFileUrl } from "../../lib/file-url.ts";
import type { PackagedExploreSeedId } from "../assets/packaged-explore-assets";
import { ApiResultError, unwrapApiResult } from "../lib/unwrapApiResult.ts";
import { createLiveMicrophoneAccess } from "../media/microphone/liveAccess.ts";
import type { OsMicrophoneStatus } from "../media/microphone/types.ts";
import { AUDIO_FILE_PATH_UNREADABLE_MESSAGE } from "../screens/Feature/fields/audioAssetInput.ts";
import {
  DROPPED_FILE_PATH_UNREADABLE_MESSAGE,
  IMAGE_EXTENSIONS,
} from "../screens/Feature/fields/imageAssetInput.ts";
import { isSupportedDroppedVideo } from "../screens/Feature/fields/videoAssetInput.ts";

import type {
  ExploreApi,
  ExploreAsset,
  ExploreAudioSource,
  ExploreMediaChooseRequest,
  ExploreMediaInput,
  ExploreRuntime,
} from "./ExploreRuntime";
import { DESKTOP_GENERATION_POLLING_POLICY } from "./generationPollingPolicy.ts";
import {
  UNSUPPORTED_MEDIA_MESSAGE,
  describeMediaFilter,
  isAcceptedMediaFile,
  requestExtractsAudioFromVideo,
} from "./mediaInput.ts";

export type DesktopExploreHost = {
  showOpenFileDialog: (input: {
    title?: string;
    filters?: { name: string; extensions: string[] }[];
    properties?: string[];
  }) => Promise<string[] | null>;
  getPathForFile: (file: File) => string;
  getLocalFileSize: (input: { filePath: string }) => Promise<number>;
  showItemInFolder: (input: { filePath: string }) => void;
  getExploreLoraRecipeSeedPath: (input: {
    seedId: ExploreLoraRecipeSeedId;
  }) => Promise<string>;
  getExploreAudioToVideoSeedPaths: () => Promise<{
    audio: string;
    startFrame: string;
  }>;
  getPackagedSeedPath: (input: { kind: PackagedSeedKind }) => Promise<string>;
  writeTempFile: (input: { suffix: ".wav"; data: ArrayBuffer }) => Promise<string>;
  diagnoseOsMicrophone?: () => Promise<OsMicrophoneStatus>;
  openMicrophoneSettings?: (() => Promise<void>) | null;
  openHuggingFaceAuth: (params: {
    clientId: string;
    redirectUri: string;
    scope: string;
    state: string;
    codeChallenge: string;
    codeChallengeMethod: string;
  }) => Promise<boolean>;
};

function desktopHost(): DesktopExploreHost {
  return {
    showOpenFileDialog: (input) => window.electronAPI.showOpenFileDialog(input),
    getPathForFile: (file) => window.electronAPI.getPathForFile(file),
    getLocalFileSize: (input) => window.electronAPI.getLocalFileSize(input),
    showItemInFolder: (input) => {
      void window.electronAPI.showItemInFolder(input);
    },
    getExploreLoraRecipeSeedPath: (input) =>
      window.electronAPI.getExploreLoraRecipeSeedPath(input),
    getExploreAudioToVideoSeedPaths: () =>
      window.electronAPI.getExploreAudioToVideoSeedPaths(),
    getPackagedSeedPath: (input) => window.electronAPI.getPackagedSeedPath(input),
    writeTempFile: async (input) => {
      const result = await window.electronAPI.writeTempFile(input);
      if (!result.success) {
        throw new Error(result.error);
      }
      return result.path;
    },
    diagnoseOsMicrophone: async () => {
      const result = await window.electronAPI.diagnoseMicrophone();
      if (!result.success) {
        return "unknown";
      }
      return result.osMicrophone;
    },
    openMicrophoneSettings:
      window.electronAPI.platform === "win32" || window.electronAPI.platform === "darwin"
        ? async () => {
            const result = await window.electronAPI.openMicrophoneSettings();
            if (!result.success) {
              throw new Error(result.error);
            }
          }
        : null,
    openHuggingFaceAuth: (params) => window.electronAPI.openHuggingFaceAuth(params),
  };
}

function pathForDroppedFile(host: DesktopExploreHost, file: File): string | null {
  try {
    const filePath = host.getPathForFile(file);
    return filePath && filePath.length > 0 ? filePath : null;
  } catch {
    return null;
  }
}

function isPackagedSeedKind(id: string): id is PackagedSeedKind {
  return (PACKAGED_SEED_KINDS as readonly string[]).includes(id);
}

function isExploreLoraRecipeSeedId(id: string): id is ExploreLoraRecipeSeedId {
  return Object.prototype.hasOwnProperty.call(EXPLORE_LORA_RECIPE_SEED_FILES, id);
}

async function desktopPackagedAssetPath(
  id: PackagedExploreSeedId,
  host: DesktopExploreHost,
): Promise<string> {
  switch (id) {
    case "audio-to-video-audio":
      return (await host.getExploreAudioToVideoSeedPaths()).audio;
    case "audio-to-video-start-frame":
      return (await host.getExploreAudioToVideoSeedPaths()).startFrame;
    default:
      if (isPackagedSeedKind(id)) {
        return host.getPackagedSeedPath({ kind: id });
      }
      if (isExploreLoraRecipeSeedId(id)) {
        return host.getExploreLoraRecipeSeedPath({ seedId: id });
      }
      throw new Error(
        `Desktop path ingest is not wired for packaged seed "${id satisfies never}"`,
      );
  }
}

export function createDesktopMediaInput(
  api: ExploreApi,
  host: DesktopExploreHost,
): ExploreMediaInput {
  const ingestPath = async (filePath: string) =>
    unwrapApiResult(await api.ingestAsset({ path: filePath }));
  const openDialog = (request: ExploreMediaChooseRequest, properties?: string[]) =>
    host.showOpenFileDialog({
      title: request.title,
      filters: [
        {
          name: describeMediaFilter(request),
          extensions: [...request.extensions],
        },
      ],
      ...(properties ? { properties } : {}),
    });
  const ingestAcceptedPath = async (
    filePath: string,
    request: ExploreMediaChooseRequest,
  ) => {
    if (!isAcceptedMediaFile({ name: filePath, type: "" }, request)) {
      throw new ApiResultError(UNSUPPORTED_MEDIA_MESSAGE, {
        code: "UNSUPPORTED_MEDIA",
      });
    }
    return ingestPath(filePath);
  };

  return {
    choose: async (request) => {
      const paths = await openDialog(request);
      const selected = paths?.[0];
      if (!selected) {
        return null;
      }
      const picked = { name: selected, type: "" };
      if (isSupportedDroppedVideo(picked) && requestExtractsAudioFromVideo(request)) {
        // A2V import affordance: a video pick on an audio field that also
        // accepts video resolves to its extracted audio track.
        const video = await ingestPath(selected);
        return unwrapApiResult(await api.extractAudio(video.id));
      }
      return ingestAcceptedPath(selected, request);
    },
    chooseMany: async (request) => {
      const paths = await openDialog(request, ["openFile", "multiSelections"]);
      return (paths ?? []).map((selected) => () => ingestAcceptedPath(selected, request));
    },
    ingestDroppedFile: async (file) => {
      const filePath = pathForDroppedFile(host, file);
      if (!filePath) {
        throw new Error(DROPPED_FILE_PATH_UNREADABLE_MESSAGE);
      }
      return ingestPath(filePath);
    },
  };
}

function createDesktopRecordedFileIngest(
  api: ExploreApi,
  host: DesktopExploreHost,
): (file: File) => Promise<ExploreAsset> {
  return async (file) => {
    const data = await file.arrayBuffer();
    const filePath = await host.writeTempFile({ suffix: ".wav", data });
    return unwrapApiResult(await api.ingestAsset({ path: filePath }));
  };
}

function createDesktopDroppedVideoAudio(
  api: ExploreApi,
  host: DesktopExploreHost,
): (file: File) => Promise<ExploreAsset> {
  // Dropping a video on an audio field is a valid source: ingest it, then let
  // the backend pull the audio track out.
  return async (file) => {
    const filePath = pathForDroppedFile(host, file);
    if (!filePath) {
      throw new Error(AUDIO_FILE_PATH_UNREADABLE_MESSAGE);
    }
    const video = unwrapApiResult(await api.ingestAsset({ path: filePath }));
    return unwrapApiResult(await api.extractAudio(video.id));
  };
}

export function createDesktopAudioSourceUrl(
  fetchBackend: BackendFetch,
): (asset: { id: string }) => Promise<ExploreAudioSource> {
  return async (asset) => {
    // The waveform decodes bytes with `fetch`, and CSP's connect-src does not
    // cover `file:`. Stream them off the backend's asset-bytes route — the same
    // URL Remote plays — rather than base64-ing a whole track through IPC.
    const response = await fetchBackend(`/api/assets/${asset.id}/bytes`);
    if (!response.ok) {
      throw new ApiResultError("could not read this audio", {
        code: response.status === 404 ? "ASSET_NOT_FOUND" : undefined,
        status: response.status,
      });
    }
    const url = URL.createObjectURL(await response.blob());
    return { url, release: () => URL.revokeObjectURL(url) };
  };
}

export function createDesktopExploreRuntime(
  modelsVersion: number,
  options?: {
    api?: ExploreApi;
    host?: DesktopExploreHost;
    wsUrl?: (path: string) => Promise<string>;
    fetchBackend?: BackendFetch;
    persistEnhanceProviderPreference?: (provider: EnhanceProvider) => void;
    autoEnhancePrompts?: boolean;
    onGeminiKeyRequired?: () => void;
    usesLtxApiTextEncoding?: boolean;
    nagUnavailableReason?: string | null;
    showNagStrengthControls?: boolean;
    openLtxApiKeySettings?: () => void;
    openTextEncodingSettings?: () => void;
    budgetMode?: BudgetMode;
    budgetModeSaving?: boolean;
    advancedControls?: boolean;
    neverSplitIntoParts?: boolean;
    neverSplitSaving?: boolean;
    budgetRiskConfirm?: ExploreRuntime["budgetRiskConfirm"];
  },
): ExploreRuntime & { modelsVersion: number } {
  const api = options?.api ?? ApiClient;
  const host = options?.host ?? desktopHost();
  return {
    modelsVersion,
    budgetMode: options?.budgetMode ?? "brave",
    budgetModeSaving: options?.budgetModeSaving ?? false,
    advancedControls: options?.advancedControls ?? false,
    neverSplitIntoParts: options?.neverSplitIntoParts ?? false,
    neverSplitSaving: options?.neverSplitSaving ?? false,
    budgetRiskConfirm: options?.budgetRiskConfirm,
    api,
    wsUrl: options?.wsUrl ?? backendWsUrl,
    mediaInput: createDesktopMediaInput(api, host),
    loadPackagedAsset: async (id) => {
      const seedPath = await desktopPackagedAssetPath(id, host);
      return unwrapApiResult(await api.ingestAsset({ path: seedPath }));
    },
    ingestRecordedFile: createDesktopRecordedFileIngest(api, host),
    microphoneAccess: createLiveMicrophoneAccess({
      diagnoseOsMicrophone: host.diagnoseOsMicrophone ?? (async () => "unknown"),
      openMicrophoneSettings: host.openMicrophoneSettings ?? null,
    }),
    ingestDroppedVideoAudio: createDesktopDroppedVideoAudio(api, host),
    catalogInstall: {
      openHuggingFaceAuth: (params) => host.openHuggingFaceAuth(params),
    },
    customLoraImport: {
      pickWeightsFile: async () => {
        const paths = await host.showOpenFileDialog({
          title: "Choose LoRA weights",
          filters: [{ name: "LoRA weights", extensions: ["safetensors"] }],
          properties: ["openFile"],
        });
        return paths?.[0] ?? null;
      },
      pickCoverFile: async () => {
        const paths = await host.showOpenFileDialog({
          title: "Choose a cover image",
          filters: [{ name: "Images", extensions: [...IMAGE_EXTENSIONS] }],
          properties: ["openFile"],
        });
        return paths?.[0] ?? null;
      },
      pathForDroppedFile: (file) => pathForDroppedFile(host, file),
      fileSize: async (filePath) => {
        try {
          return await host.getLocalFileSize({ filePath });
        } catch {
          return null;
        }
      },
    },
    audioSourceUrl: createDesktopAudioSourceUrl(options?.fetchBackend ?? backendFetch),
    mediaUrlForAsset: (asset) =>
      asset.path && asset.path.length > 0 ? pathToFileUrl(asset.path) : null,
    thumbUrlForAsset: (asset) =>
      asset.thumbnail_path && asset.thumbnail_path.length > 0
        ? pathToFileUrl(asset.thumbnail_path)
        : null,
    waveformUrlForAsset: async (asset) => {
      const response = await backendFetch(`/api/assets/${asset.id}/bytes`);
      if (!response.ok) return null;
      const blob = await response.blob();
      return URL.createObjectURL(blob);
    },
    host: "desktop",
    revealInFolder: (filePath) => {
      host.showItemInFolder({ filePath });
    },
    hoverPreview: true,
    deleteAsset: async (assetId) => {
      unwrapApiResult(await api.deleteAsset(assetId));
    },
    generationPolling: DESKTOP_GENERATION_POLLING_POLICY,
    persistEnhanceProviderPreference: options?.persistEnhanceProviderPreference,
    autoEnhancePrompts: options?.autoEnhancePrompts,
    onGeminiKeyRequired: options?.onGeminiKeyRequired,
    usesLtxApiTextEncoding: options?.usesLtxApiTextEncoding,
    nagUnavailableReason: options?.nagUnavailableReason,
    showNagStrengthControls: options?.showNagStrengthControls,
    openLtxApiKeySettings: options?.openLtxApiKeySettings,
    openTextEncodingSettings: options?.openTextEncodingSettings,
  };
}
