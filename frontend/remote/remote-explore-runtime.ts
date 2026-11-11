import type { EnhanceProvider } from "../hooks/use-prompt-enhancer-provider.ts";
import { createApiClient } from "../lib/api-client.ts";
import { createBackendWsUrl } from "../lib/backend.ts";
import type { PackagedExploreSeedId } from "../ltx-io/assets/packaged-explore-assets.ts";
import { ApiResultError, unwrapApiResult } from "../ltx-io/lib/unwrapApiResult.ts";
import type {
  ExploreApi,
  ExploreMediaChooseRequest,
  ExploreMediaInput,
  ExploreRuntime,
} from "../ltx-io/runtime/ExploreRuntime.tsx";
import { REMOTE_GENERATION_POLLING_POLICY } from "../ltx-io/runtime/generationPollingPolicy.ts";
import {
  UNSUPPORTED_MEDIA_MESSAGE,
  isAcceptedMediaFile,
} from "../ltx-io/runtime/mediaInput.ts";

import { api, getBearerToken } from "./api.ts";

const REMOTE_ENHANCE_PROVIDER_KEY = "ltx-remote-enhance-provider";

export function readRemoteEnhanceProviderPreference(): EnhanceProvider | null {
  try {
    const value = globalThis.localStorage.getItem(REMOTE_ENHANCE_PROVIDER_KEY);
    return value === "local" || value === "api" ? value : null;
  } catch {
    return null;
  }
}

export function persistRemoteEnhanceProviderPreference(provider: EnhanceProvider): void {
  try {
    globalThis.localStorage.setItem(REMOTE_ENHANCE_PROVIDER_KEY, provider);
  } catch {
    // private mode / blocked storage
  }
}

function waitForMountedInputFiles(input: HTMLInputElement): Promise<File[]> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (files: File[]) => {
      if (settled) return;
      settled = true;
      input.removeEventListener("change", onChange);
      input.removeEventListener("cancel", onCancel);
      input.value = "";
      resolve(files);
    };
    // Copy the list first: clearing the input empties the live FileList.
    const onChange = () => finish(Array.from(input.files ?? []));
    const onCancel = () => finish([]);
    input.addEventListener("change", onChange);
    input.addEventListener("cancel", onCancel);
    input.click();
  });
}

async function waitForMountedInputFile(input: HTMLInputElement): Promise<File | null> {
  return (await waitForMountedInputFiles(input))[0] ?? null;
}

function remoteWsUrl() {
  return createBackendWsUrl(async () => ({
    url: globalThis.location?.origin ?? "http://127.0.0.1",
    token: getBearerToken() ?? "",
  }));
}

function signedMediaUrl(value: string | null | undefined): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

export function createRemoteMediaInput(
  exploreApi: ExploreApi,
  waitForFile: (
    input: HTMLInputElement,
  ) => Promise<File | null> = waitForMountedInputFile,
  waitForFiles: (
    input: HTMLInputElement,
  ) => Promise<File[]> = waitForMountedInputFiles,
): ExploreMediaInput {
  const uploadAccepted = async (file: File, request: ExploreMediaChooseRequest) => {
    if (!isAcceptedMediaFile(file, request)) {
      throw new ApiResultError(UNSUPPORTED_MEDIA_MESSAGE, {
        code: "UNSUPPORTED_MEDIA",
      });
    }
    return unwrapApiResult(await exploreApi.uploadAsset(file));
  };
  return {
    choose: async (request) => {
      if (request.fileInput == null) {
        throw new Error("Remote choose requires a mounted file input");
      }
      const file = await waitForFile(request.fileInput);
      return file ? uploadAccepted(file, request) : null;
    },
    chooseMany: async (request) => {
      if (request.fileInput == null) {
        throw new Error("Remote choose requires a mounted file input");
      }
      const files = await waitForFiles(request.fileInput);
      return files.map((file) => () => uploadAccepted(file, request));
    },
    ingestDroppedFile: async (file) =>
      unwrapApiResult(await exploreApi.uploadAsset(file)),
  };
}

export function createRemoteExploreRuntime(options: {
  loadPackagedFile: (id: PackagedExploreSeedId) => Promise<File>;
  api?: ExploreApi;
  waitForFile?: (input: HTMLInputElement) => Promise<File | null>;
  waitForFiles?: (input: HTMLInputElement) => Promise<File[]>;
  wsUrl?: (path: string) => Promise<string>;
}): ExploreRuntime & { modelsVersion: null } {
  const exploreApi = options.api ?? createApiClient(api);

  return {
    modelsVersion: null,
    api: exploreApi,
    wsUrl: options.wsUrl ?? remoteWsUrl(),
    mediaInput: createRemoteMediaInput(
      exploreApi,
      options.waitForFile,
      options.waitForFiles,
    ),
    loadPackagedAsset: async (id) =>
      unwrapApiResult(await exploreApi.uploadAsset(await options.loadPackagedFile(id))),
    ingestRecordedFile: async (file) =>
      unwrapApiResult(await exploreApi.uploadAsset(file)),
    microphoneAccess: null,
    // Remote uploads only accept the audio allowlist, so a dropped video has no
    // path to an extracted track — the field reports it as unsupported.
    ingestDroppedVideoAudio: null,
    catalogInstall: null,
    // The host signs every remote asset's bytes URL (media elements can't send
    // the Bearer token), so the trimmer plays it directly — nothing to revoke.
    audioSourceUrl: async (asset) => {
      const url = signedMediaUrl(asset.bytes_url);
      if (url != null) {
        return { url };
      }
      throw new Error("Remote audio asset is missing its media URL");
    },
    mediaUrlForAsset: (asset) => signedMediaUrl(asset.bytes_url),
    thumbUrlForAsset: (asset) => signedMediaUrl(asset.thumbnail_url),
    waveformUrlForAsset: async (asset) => {
      const url = signedMediaUrl(asset.bytes_url);
      if (url == null) return null;
      const response = await fetch(url);
      if (!response.ok) return null;
      const blob = await response.blob();
      return URL.createObjectURL(blob);
    },
    host: "remote",
    revealInFolder: null,
    // Tiles still require a fine pointer, so phones do not hover-play.
    hoverPreview: true,
    deleteAsset: null,
    generationPolling: REMOTE_GENERATION_POLLING_POLICY,
    persistEnhanceProviderPreference: persistRemoteEnhanceProviderPreference,
    readEnhanceProviderPreference: readRemoteEnhanceProviderPreference,
  };
}
