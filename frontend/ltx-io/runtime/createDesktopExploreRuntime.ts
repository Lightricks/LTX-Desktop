import type { EnhanceProvider } from "../../hooks/use-prompt-enhancer-provider";
import { ApiClient } from "../../lib/api-client.ts";
import { type BackendFetch, backendFetch, backendWsUrl } from "../../lib/backend.ts";
import { pathToFileUrl } from "../../lib/file-url.ts";
import type { PackagedExploreSeedId } from "../assets/packaged-explore-assets";
import {
  EXPLORE_LORA_RECIPE_SEED_FILES,
  PACKAGED_SEED_KINDS,
  type ExploreLoraRecipeSeedId,
  type PackagedSeedKind,
} from "../../../shared/explore-seed-filenames.ts";
import { ApiResultError, unwrapApiResult } from "../lib/unwrapApiResult.ts";
import { AUDIO_FILE_PATH_UNREADABLE_MESSAGE } from "../screens/Feature/fields/audioAssetInput.ts";
import { isSupportedDroppedVideo } from "../screens/Feature/fields/videoAssetInput.ts";
import { DROPPED_FILE_PATH_UNREADABLE_MESSAGE } from "../screens/Feature/fields/imageAssetInput.ts";

import { createLiveMicrophoneAccess } from "../media/microphone/liveAccess.ts";
import type { OsMicrophoneStatus } from "../media/microphone/types.ts";
import type {
  ExploreApi,
  ExploreAsset,
  ExploreAudioSource,
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
  }) => Promise<string[] | null>;
  getPathForFile: (file: File) => string;
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
      window.electronAPI.platform === "win32" ||
      window.electronAPI.platform === "darwin"
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

  return {
    choose: async (request) => {
      const paths = await host.showOpenFileDialog({
        title: request.title,
        filters: [
          {
            name: describeMediaFilter(request),
            extensions: [...request.extensions],
          },
        ],
      });
      const selected = paths?.[0];
      if (!selected) {
        return null;
      }
      const picked = { name: selected, type: "" };
      if (
        isSupportedDroppedVideo(picked) &&
        requestExtractsAudioFromVideo(request)
      ) {
        // A2V import affordance: a video pick on an audio field that also
        // accepts video resolves to its extracted audio track.
        const video = await ingestPath(selected);
        return unwrapApiResult(await api.extractAudio(video.id));
      }
      if (!isAcceptedMediaFile(picked, request)) {
        throw new ApiResultError(UNSUPPORTED_MEDIA_MESSAGE, {
          code: "UNSUPPORTED_MEDIA",
        });
      }
      return ingestPath(selected);
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
    openLtxApiKeySettings?: () => void;
    openTextEncodingSettings?: () => void;
  },
): ExploreRuntime & { modelsVersion: number } {
  const api = options?.api ?? ApiClient;
  const host = options?.host ?? desktopHost();
  return {
    modelsVersion,
    api,
    wsUrl: options?.wsUrl ?? backendWsUrl,
    mediaInput: createDesktopMediaInput(api, host),
    loadPackagedAsset: async (id) => {
      const seedPath = await desktopPackagedAssetPath(id, host);
      return unwrapApiResult(await api.ingestAsset({ path: seedPath }));
    },
    ingestRecordedFile: createDesktopRecordedFileIngest(api, host),
    microphoneAccess: createLiveMicrophoneAccess({
      diagnoseOsMicrophone:
        host.diagnoseOsMicrophone ?? (async () => "unknown"),
      openMicrophoneSettings: host.openMicrophoneSettings ?? null,
    }),
    ingestDroppedVideoAudio: createDesktopDroppedVideoAudio(api, host),
    catalogInstall: {
      openHuggingFaceAuth: (params) => host.openHuggingFaceAuth(params),
    },
    audioSourceUrl: createDesktopAudioSourceUrl(
      options?.fetchBackend ?? backendFetch,
    ),
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
    openLtxApiKeySettings: options?.openLtxApiKeySettings,
    openTextEncodingSettings: options?.openTextEncodingSettings,
  };
}
