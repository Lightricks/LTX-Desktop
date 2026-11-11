import { type ReactNode, createContext, useContext } from "react";

import type { EnhanceProvider } from "@/hooks/use-prompt-enhancer-provider";
import type { ApiSuccessOf, BoundApiClient } from "@/lib/api-client";
import type { BudgetMode } from "@/lib/budget-risk";
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
  | "getUiPreferences"
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
  | "createTextToImage"
  | "getImageGenerationStatus"
  | "createImageToVideo"
  | "createKeyframes"
  | "createLoraRecipe"
  | "createCustomLora"
  // Custom LoRA. List works on both hosts. Import, delete, and putting weights back exist on Desktop only.
  | "listCustomLoras"
  | "importCustomLora"
  | "getCustomLoraLinkInfo"
  | "getCustomLoraImportProgress"
  | "getCustomLoraImportActive"
  | "deleteCustomLora"
  | "deleteCustomLoraWeights"
  | "restoreCustomLoraWeights"
  | "createAudioToVideo"
  | "createRetake"
  | "createExtend"
  | "preflightEdit"
  | "icLoraRecipeMemory"
  | "createIcLoraRecipe"
  | "preflightIcLoraRecipe"
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
  | "getGeneration"
  | "assistantTurn"
  | "listAssistantConversations"
  | "getAssistantConversation"
  | "deleteAssistantConversation"
  | "continueAssistant"
  | "stopAssistant"
  | "streamAssistantEvents"
  | "getAssistantSelection"
  | "setAssistantSelection"
  | "enqueueAssistantMessage"
  | "removeQueuedAssistantMessage"
  | "releaseAssistantQueue"
  | "listAssistantModels"
>;

export type ExploreAssistantProvider = "gemini" | "local";
export type ExploreAssistantLocalModel =
  ApiSuccessOf<"getAssistantSelection">["local_model"];

/**
 * Desktop's live assistant settings. Remote omits this and reads and sets the
 * same shared choice through /api/assistant/selection.
 */
export type ExploreAssistantSettings = {
  provider: ExploreAssistantProvider;
  localModel: ExploreAssistantLocalModel;
  geminiModel: string;
  hasGeminiApiKey: boolean;
  select: (
    provider: ExploreAssistantProvider,
    localModel?: ExploreAssistantLocalModel,
  ) => Promise<void>;
  /** A select is saving, from Settings or the composer. */
  switching: boolean;
  /** With a fix, opens where that fix is made; otherwise the Assistant section. */
  openSettings: (fix?: "geminiKey" | "localModel") => void;
};

export type ExploreMediaChooseRequest = {
  title: string;
  accept: string;
  extensions: readonly string[];
  fileInput: HTMLInputElement | null;
};

/** One picked file that is not ingested yet. Call it to ingest the file. */
export type PendingImport = () => Promise<ExploreAsset>;

export type ExploreMediaInput = {
  choose: (request: ExploreMediaChooseRequest) => Promise<ExploreAsset | null>;
  /**
   * Multi-select picker. It returns one pending import for each picked file, in pick
   * order. The caller decides how many to ingest, so a limit wastes no upload.
   */
  chooseMany: (request: ExploreMediaChooseRequest) => Promise<PendingImport[]>;
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

/**
 * Add and delete custom LoRAs. Desktop only: the weights are copied or downloaded into the
 * models folder on this computer. Remote leaves it unset, so the phone can run a saved LoRA
 * and cannot add or delete one.
 */
export type ExploreCustomLoraImport = {
  /** Show the file dialog. Returns the picked path, or `null` when the user cancels. */
  pickWeightsFile: () => Promise<string | null>;
  /** Show the image dialog. Returns the picked path, or `null` when the user cancels. */
  pickCoverFile: () => Promise<string | null>;
  /** The path of a file the user dropped, or `null` when this computer cannot read it. */
  pathForDroppedFile: (file: File) => string | null;
  /** The size of a file the user just picked, or `null` when it cannot be read. */
  fileSize: (filePath: string) => Promise<number | null>;
};

export type ExploreRuntime = {
  askAssistant?: (assetId: string) => void;
  assistantSettings?: ExploreAssistantSettings;
  modelsVersion: ExploreModelsVersion;
  /**
   * Desktop settings value, so a mode toggle refetches specs. Remote omits it
   * and renders whatever the specs payload echoes.
   */
  budgetMode?: BudgetMode;
  /** True while Desktop is saving the budget mode. Generate stays off until it lands. */
  budgetModeSaving?: boolean;
  /** Desktop setting: show the Advanced section of Refine Details. Remote omits it and reads it from the host. */
  advancedControls?: boolean;
  /** Desktop's "Never split into parts", like `budgetMode`: specs are priced with it. */
  neverSplitIntoParts?: boolean;
  neverSplitSaving?: boolean;
  /**
   * Desktop's persisted "don't ask again" for risky generates. Remote omits
   * it, so the phone always confirms and offers no opt-out.
   */
  budgetRiskConfirm?: {
    skip: boolean;
    setSkip: (skip: boolean) => void;
  };
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
  /** Desktop only. `undefined` on Remote. */
  customLoraImport?: ExploreCustomLoraImport;
  audioSourceUrl: (asset: ExploreAsset) => Promise<ExploreAudioSource>;
  mediaUrlForAsset: (asset: ExploreMediaAsset) => string | null;
  thumbUrlForAsset: (asset: ExploreAsset) => string | null;
  waveformUrlForAsset: (asset: ExploreMediaAsset) => Promise<string | null>;
  /** Which app runs this screen. Copy that names a place to fix a problem follows it. */
  host: "desktop" | "remote";
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
   * Desktop: non-null when NAG negative prompt guidance will be skipped
   * (API text encoding active, Torch Compile enabled, or NAG scale set to 0).
   */
  nagUnavailableReason?: string | null;
  /**
   * Desktop: the live "show negative prompt strength" setting. Remote omits it and reads
   * the same setting from the prompt enhancer status.
   */
  showNagStrengthControls?: boolean;
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
