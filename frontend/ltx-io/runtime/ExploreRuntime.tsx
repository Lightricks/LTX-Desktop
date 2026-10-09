import { type ReactNode, createContext, useContext } from "react";

import type { EnhanceProvider } from "@/hooks/use-prompt-enhancer-provider";
import type { BoundApiClient } from "@/lib/api-client";
import type { ExploreAsset, ExploreListedAsset } from "@/lib/explore-contract";

import type { MicrophoneAccess } from "@/ltx-io/media/microphone/types";

import type { PackagedExploreSeedId } from "../assets/packaged-explore-assets";
import { LtxioToastHost } from "../components/shared/Toast/LtxioToastHost";

import type { ExploreGenerationPollingPolicy } from "./generationPollingPolicy";

export type { ExploreAsset, ExploreListedAsset, ExploreGenerationPollingPolicy };
export { folderRevealPath } from "./folderRevealPath";

export type ExploreMediaAsset = {
  id: string;
  path?: string | null;
  bytes_url?: string | null;
};

export type ExploreApi = Pick<
  BoundApiClient,
  | "getGenerateVideoModelSpecs"
  | "getFeatureFlags"
  | "updateFeatureFlags"
  | "listGenerations"
  | "listRecentFeatures"
  | "getDashboard"
  | "getDashboardSelection"
  | "updateDashboardSelection"
  | "getGenerationSeed"
  | "updateGenerationSeed"
  | "getGenerationQueue"
  | "reorderGenerationQueue"
  | "clearGenerationQueueDone"
  | "clearGenerationQueueFailed"
  | "markGenerationQueueDoneSeen"
  | "dismissGenerationQueueDone"
  | "createTextToVideo"
  | "createImageToVideo"
  | "createLoraRecipe"
  | "createAudioToVideo"
  | "createRetake"
  | "createExtend"
  | "createIcLoraRecipe"
  | "listIcLoras"
  | "startIcLoraDownload"
  | "getIcLoraDownloadProgress"
  | "getIcLoraDownloadActive"
  | "retryGeneration"
  | "cancelQueuedGeneration"
  | "deleteGeneration"
  | "getAsset"
  | "ingestAsset"
  | "uploadAsset"
  | "listAssets"
  | "deleteAsset"
  // LoRA-recipe preflight (works on both hosts via the runtime's bound client).
  | "listLoras"
  | "startLoraDownload"
  | "getLoraDownloadProgress"
  | "getLoraDownloadActive"
  | "getHuggingFaceAuthStatus"
  | "startHuggingFaceLogin"
  | "huggingFaceLogout"
  | "trimAudio"
  | "trimVideo"
  | "extractAudio"
  | "getPromptEnhancer"
  | "enhancePrompt"
>;

export type ExploreMediaChooseRequest = {
  title: string;
  accept: string;
  extensions: readonly string[];
  fileInput: HTMLInputElement | null;
};

export type ExploreMediaInput = {
  choose: (request: ExploreMediaChooseRequest) => Promise<ExploreAsset | null>;
  ingestDroppedFile: (file: File) => Promise<ExploreAsset>;
};

/**
 * A URL the trimmer can both play and `fetch` for waveform decoding. Desktop
 * streams backend bytes into a `blob:` URL (CSP blocks fetching `file:`) and
 * releases it via `release`; Remote returns a token-bearing bytes URL with
 * nothing to release.
 */
export type ExploreAudioSource = {
  url: string;
  release?: () => void;
};

/** Desktop increments this when local models change. Remote is always `null`. */
export type ExploreModelsVersion = number | null;

/**
 * Hugging Face OAuth handoff for gated catalog installs. Desktop opens the
 * system browser through Electron; Remote has no equivalent.
 */
export type HuggingFaceAuthParams = {
  clientId: string;
  redirectUri: string;
  scope: string;
  state: string;
  codeChallenge: string;
  codeChallengeMethod: string;
};

export type ExploreCatalogInstall = {
  openHuggingFaceAuth: (params: HuggingFaceAuthParams) => Promise<boolean>;
};

export type ExploreRuntime = {
  modelsVersion: ExploreModelsVersion;
  api: ExploreApi;
  wsUrl: (path: string) => Promise<string>;
  mediaInput: ExploreMediaInput;
  loadPackagedAsset: (id: PackagedExploreSeedId) => Promise<ExploreAsset>;
  /**
   * Ingest a recorded blob (no filesystem path). Desktop writes a temp wav
   * then path-ingests; Remote uploads multipart.
   */
  ingestRecordedFile: (file: File) => Promise<ExploreAsset>;
  /**
   * How this host captures a recording. Desktop is live `getUserMedia`.
   * Remote has no in-app Record (`null`); Import is the file path.
   */
  microphoneAccess: MicrophoneAccess | null;
  /**
   * Ingest a dropped video and return its extracted audio track. `null` where
   * the host cannot extract audio (Remote), so the field can show an
   * unsupported message instead of a failed request.
   */
  ingestDroppedVideoAudio: ((file: File) => Promise<ExploreAsset>) | null;
  /**
   * Start Hugging Face OAuth and LoRA download from this host. `null` on Remote:
   * recipe preflight lists catalog status but must not offer Sign in / Download.
   */
  catalogInstall: ExploreCatalogInstall | null;
  audioSourceUrl: (asset: ExploreAsset) => Promise<ExploreAudioSource>;
  mediaUrlForAsset: (asset: ExploreMediaAsset) => string | null;
  thumbUrlForAsset: (asset: ExploreAsset) => string | null;
  waveformUrlForAsset: (asset: ExploreMediaAsset) => Promise<string | null>;
  revealInFolder: ((filePath: string) => void) | null;
  hoverPreview: boolean;
  deleteAsset: ((assetId: string) => Promise<void>) | null;
  generationPolling: ExploreGenerationPollingPolicy;
  persistEnhanceProviderPreference?: (provider: EnhanceProvider) => void;
  readEnhanceProviderPreference?: () => EnhanceProvider | null;
  /** Desktop live setting. Remote omits this and follows GET /api/prompt-enhancer. */
  autoEnhancePrompts?: boolean;
  onGeminiKeyRequired?: () => void;
  /**
   * Desktop: Text encoding is currently LTX API (`useLocalTextEncoder === false`).
   * Remote omits this — the phone cannot read Settings.
   */
  usesLtxApiTextEncoding?: boolean;
  /**
   * Desktop only. Opens Settings on the API Keys tab and focuses the LTX key.
   * Remote has no Settings modal, so this stays unset.
   */
  openLtxApiKeySettings?: () => void;
  /**
   * Desktop only. Opens Settings on General, scrolled to Text encoding.
   * Remote has no Settings modal, so this stays unset.
   */
  openTextEncodingSettings?: () => void;
};

const ExploreRuntimeContext = createContext<ExploreRuntime | null>(null);

export function ExploreRuntimeProvider({
  value,
  children,
}: {
  value: ExploreRuntime;
  children: ReactNode;
}) {
  return (
    <ExploreRuntimeContext.Provider value={value}>
      {children}
      <LtxioToastHost />
    </ExploreRuntimeContext.Provider>
  );
}

export function useExploreRuntime(): ExploreRuntime {
  const value = useContext(ExploreRuntimeContext);
  if (value == null) {
    throw new Error("useExploreRuntime must be used within ExploreRuntimeProvider");
  }
  return value;
}
