import { AlertCircle } from "lucide-react";
import { Fragment, useCallback, useEffect, useLayoutEffect, useMemo, useState } from "react";

import { Button as DsButton } from "@/ds/Button/Button";
import { Text } from "@/ds/Text/Text";

import { useAppSettings } from "../../contexts/AppSettingsContext";
import { useHfAuth } from "../../hooks/use-hf-auth";
import { useHfModelAccess } from "../../hooks/use-hf-model-access";
import { ApiClient, type ApiSuccessOf } from "../../lib/api-client";
import { formatBytes } from "../../lib/format";
import { logger } from "../../lib/logger";
import {
  ENCODING_MODEL_SEARCH_HAYSTACK,
  matchesCatalogSearch,
} from "../../lib/model-catalog";
import { HfModelAccessGate } from "../HfModelAccessGate";
import { Tooltip, TooltipProvider } from "@/ds/Tooltip/Tooltip";

import {
  LOCAL_TEXT_ENCODER_REQUIRED_TOOLTIP,
  ModelCatalogActionGroup,
  ModelCatalogCellText,
  ModelCatalogDetailRow,
  ModelCatalogNameCell,
  ModelCatalogSizeCell,
  ModelCatalogSizeText,
  ModelCatalogInstalledCheck,
  ModelCatalogTable,
  confirmCatalogModelDelete,
  modelCatalogActionButtonWideClassName,
  modelCatalogDownloadButtonLabel,
  modelCatalogRowClassNames,
  modelCatalogSettingsTooltipPortal,
} from "./ModelCatalogTable";
import modelCatalogTableStyles from "./ModelCatalogTable.module.scss";
import { SettingsStack, SettingsSubsection } from "./SettingsContent";
import contentStyles from "./SettingsContent.module.scss";

type EncodingCp = NonNullable<
  ApiSuccessOf<"getTextEncoderRecommendation">["cp_to_download"]
>;
type EnhancerCp = NonNullable<
  ApiSuccessOf<"getTextEncoderRecommendation">["local_enhancer_cp"]
>;

const DOWNLOAD_POLL_MS = 1000;

function gbToBytes(gb: number): number {
  return Math.round(gb * 1_000_000_000);
}

// Maps a raw checkpoint ID (e.g. "gemma4-12b-with-proj-ltx-2.5") to the short
// friendly name shown in Settings → Models (e.g. "Gemma 4"). Unknown IDs fall
// through unchanged so the row still shows something meaningful.
function friendlyEncodingModelName(cp: string): string {
  if (cp.includes("gemma-4-e2b")) return "Gemma 4 E2B";
  if (cp.includes("gemma4-12b") || cp.includes("gemma-4-12b")) return "Gemma 4";
  if (cp.includes("gemma-3")) return "Gemma 3";
  return cp;
}

function downloadPercent(
  progress: ApiSuccessOf<"getModelDownloadProgress"> | null,
): number {
  if (progress?.status !== "downloading") return 0;
  return Math.round(progress.total_progress ?? progress.current_file_progress);
}

export function EncodingEnhanceModelsSection({
  active,
  searchQuery = "",
  onSearchHasMatches,
}: {
  active: boolean;
  searchQuery?: string;
  onSearchHasMatches?: (hasMatches: boolean) => void;
}) {
  const { notifyModelsChanged } = useAppSettings();
  const [recommendation, setRecommendation] = useState<
    ApiSuccessOf<"getTextEncoderRecommendation"> | null
  >(null);
  const [encoderDownloadingCp, setEncoderDownloadingCp] = useState<EncodingCp | null>(
    null,
  );
  const [enhancerDownloadingCp, setEnhancerDownloadingCp] = useState<EnhancerCp | null>(
    null,
  );
  const [encoderSessionId, setEncoderSessionId] = useState<string | null>(null);
  const [enhancerSessionId, setEnhancerSessionId] = useState<string | null>(null);
  const [encoderProgress, setEncoderProgress] =
    useState<ApiSuccessOf<"getModelDownloadProgress"> | null>(null);
  const [enhancerProgress, setEnhancerProgress] =
    useState<ApiSuccessOf<"getModelDownloadProgress"> | null>(null);
  const [encoderError, setEncoderError] = useState<string | null>(null);
  const [enhancerError, setEnhancerError] = useState<string | null>(null);
  const [encoderDeleteBusy, setEncoderDeleteBusy] = useState(false);
  const [enhancerDeleteBusy, setEnhancerDeleteBusy] = useState(false);

  const { hfAuthStatus, hfAuthPolling, startHuggingFaceLogin } = useHfAuth(active);

  const encoderCpToDownload = recommendation?.cp_to_download ?? null;
  const encoderInstalled = recommendation !== null && encoderCpToDownload === null;

  const enhancerInstalled =
    recommendation !== null &&
    recommendation.local_enhancer_cp !== null &&
    recommendation.active_local_enhancer_cp === recommendation.local_enhancer_cp;
  const enhancerCpToDownload =
    recommendation?.local_enhancer_cp && !enhancerInstalled
      ? recommendation.local_enhancer_cp
      : null;

  // Resolve which checkpoint each row refers to: prefer the installed ("active")
  // ID, fall back to the ID that still needs downloading. Covers both the
  // installed and not-installed states with one label source.
  const encoderDisplayCp =
    recommendation?.active_local_text_encoder_cp ?? encoderCpToDownload;
  const enhancerDisplayCp =
    recommendation?.active_local_enhancer_cp ??
    recommendation?.local_enhancer_cp ??
    null;
  const encoderDisplayName = encoderDisplayCp
    ? `${friendlyEncodingModelName(encoderDisplayCp)} (text encoder)`
    : "Local text encoder";
  const enhancerDisplayName = enhancerDisplayCp
    ? `${friendlyEncodingModelName(enhancerDisplayCp)} (prompt enhancer)`
    : "Local prompt enhancer";

  const encoderModelTypes = useMemo(
    () => (encoderCpToDownload ? [encoderCpToDownload] : []),
    [encoderCpToDownload],
  );
  const enhancerModelTypes = useMemo(
    () => (enhancerCpToDownload ? [enhancerCpToDownload] : []),
    [enhancerCpToDownload],
  );

  const encoderAccess = useHfModelAccess(encoderModelTypes, hfAuthStatus);
  const enhancerAccess = useHfModelAccess(enhancerModelTypes, hfAuthStatus);

  const refreshRecommendation = useCallback(async () => {
    const result = await ApiClient.getTextEncoderRecommendation();
    if (!result.ok) {
      logger.error(
        `Failed to fetch encoding models recommendation: ${result.error.message}`,
      );
      return;
    }
    setRecommendation(result.data);
  }, []);

  // Downloads live in the backend, which outlives both this component and a renderer reload.
  // Without this, remounting shows an idle Download button whose click gets
  // DOWNLOAD_ALREADY_RUNNING while the real job keeps running unseen.
  const reattachActiveDownload = useCallback(async () => {
    const [recResult, activeResult] = await Promise.all([
      ApiClient.getTextEncoderRecommendation(),
      ApiClient.getActiveDownload(),
    ]);
    if (!recResult.ok) {
      logger.error(
        `Failed to fetch encoding models recommendation: ${recResult.error.message}`,
      );
      return;
    }
    setRecommendation(recResult.data);
    if (!activeResult.ok || !activeResult.data.session_id) return;

    const sessionId = activeResult.data.session_id;
    const cpIds: readonly string[] = activeResult.data.cp_ids ?? [];
    const encoderCp = recResult.data.cp_to_download;
    const enhancerCp = recResult.data.local_enhancer_cp;

    if (encoderCp && cpIds.includes(encoderCp)) {
      setEncoderDownloadingCp(encoderCp);
      setEncoderSessionId(sessionId);
      return;
    }
    if (enhancerCp && cpIds.includes(enhancerCp)) {
      setEnhancerDownloadingCp(enhancerCp);
      setEnhancerSessionId(sessionId);
    }
  }, []);

  useEffect(() => {
    if (!active) return;
    void reattachActiveDownload();
  }, [active, reattachActiveDownload]);

  useEffect(() => {
    if (encoderDownloadingCp === null || !encoderSessionId) return;

    const poll = async () => {
      const result = await ApiClient.getModelDownloadProgress({
        sessionId: encoderSessionId,
      });
      if (!result.ok) return;
      setEncoderProgress(result.data);
      if (result.data.status === "complete") {
        setEncoderDownloadingCp(null);
        setEncoderSessionId(null);
        const rec = await ApiClient.getTextEncoderRecommendation();
        if (rec.ok) setRecommendation(rec.data);
        notifyModelsChanged();
      } else if (result.data.status === "error") {
        setEncoderError(result.data.error ?? "Download failed");
        setEncoderDownloadingCp(null);
        setEncoderSessionId(null);
      }
    };

    void poll();
    const interval = setInterval(() => void poll(), DOWNLOAD_POLL_MS);
    return () => clearInterval(interval);
  }, [encoderDownloadingCp, encoderSessionId, notifyModelsChanged]);

  useEffect(() => {
    if (enhancerDownloadingCp === null || !enhancerSessionId) return;

    const poll = async () => {
      const result = await ApiClient.getModelDownloadProgress({
        sessionId: enhancerSessionId,
      });
      if (!result.ok) return;
      setEnhancerProgress(result.data);
      if (result.data.status === "complete") {
        setEnhancerDownloadingCp(null);
        setEnhancerSessionId(null);
        const rec = await ApiClient.getTextEncoderRecommendation();
        if (rec.ok) setRecommendation(rec.data);
        notifyModelsChanged();
      } else if (result.data.status === "error") {
        setEnhancerError(result.data.error ?? "Download failed");
        setEnhancerDownloadingCp(null);
        setEnhancerSessionId(null);
      }
    };

    void poll();
    const interval = setInterval(() => void poll(), DOWNLOAD_POLL_MS);
    return () => clearInterval(interval);
  }, [enhancerDownloadingCp, enhancerSessionId, notifyModelsChanged]);

  const startDownload = async (cpId: EncodingCp | EnhancerCp, kind: "encoder" | "enhancer") => {
    const setCp = kind === "encoder" ? setEncoderDownloadingCp : setEnhancerDownloadingCp;
    const setError = kind === "encoder" ? setEncoderError : setEnhancerError;
    const setProgress = kind === "encoder" ? setEncoderProgress : setEnhancerProgress;
    const setSession = kind === "encoder" ? setEncoderSessionId : setEnhancerSessionId;

    setCp(cpId);
    setError(null);
    setProgress(null);
    const result = await ApiClient.startModelDownload({
      type: "download",
      cp_ids: [cpId],
    });
    if (!result.ok) {
      setError(result.error.message);
      setCp(null);
      return;
    }
    if (result.data.status === "started") {
      setSession(result.data.sessionId);
    }
  };

  const deleteCheckpoint = async (
    cpId: EncodingCp | EnhancerCp,
    kind: "encoder" | "enhancer",
  ) => {
    const setDeleteBusy =
      kind === "encoder" ? setEncoderDeleteBusy : setEnhancerDeleteBusy;
    const setError = kind === "encoder" ? setEncoderError : setEnhancerError;

    setError(null);
    if (
      !confirmCatalogModelDelete(
        kind === "encoder" ? encoderDisplayName : enhancerDisplayName,
      )
    ) {
      return;
    }
    setDeleteBusy(true);
    const result = await ApiClient.deleteModels({ cp_ids: [cpId] });
    if (!result.ok) {
      setError(result.error.message || "Failed to delete model.");
      setDeleteBusy(false);
      return;
    }
    notifyModelsChanged();
    await refreshRecommendation();
    setDeleteBusy(false);
  };

  useLayoutEffect(() => {
    onSearchHasMatches?.(
      matchesCatalogSearch(ENCODING_MODEL_SEARCH_HAYSTACK, searchQuery),
    );
  }, [onSearchHasMatches, searchQuery]);

  if (!active) {
    return null;
  }

  if (searchQuery && !matchesCatalogSearch(ENCODING_MODEL_SEARCH_HAYSTACK, searchQuery)) {
    return null;
  }

  if (!recommendation) {
    return null;
  }

  const encoderBusy = encoderDownloadingCp !== null || encoderDeleteBusy;
  const enhancerBusy = enhancerDownloadingCp !== null || enhancerDeleteBusy;
  const encoderDeleteCp = recommendation.active_local_text_encoder_cp;
  // The backend protects this checkpoint while it is the only way to encode prompts, so a
  // Delete request would be a guaranteed 409. Disable and explain instead of offering it.
  const encoderDeleteBlocked = !recommendation.local_text_encoder_removable;
  const enhancerDeleteCp =
    enhancerInstalled && recommendation.active_local_enhancer_cp
      ? recommendation.active_local_enhancer_cp
      : null;
  const encoderCanDownload =
    !encoderInstalled &&
    Boolean(encoderCpToDownload) &&
    encoderAccess.allAuthorized &&
    !encoderAccess.checking;
  const enhancerCanDownload =
    !enhancerInstalled &&
    Boolean(enhancerCpToDownload) &&
    enhancerAccess.allAuthorized &&
    !enhancerAccess.checking;

  const encoderShowHfGate =
    !encoderInstalled &&
    !encoderAccess.allAuthorized &&
    (Boolean(encoderAccess.checkError) ||
      Object.values(encoderAccess.accessMap).some((s) => s === "not_authorized"));

  const enhancerShowHfGate =
    !enhancerInstalled &&
    !enhancerAccess.allAuthorized &&
    (Boolean(enhancerAccess.checkError) ||
      Object.values(enhancerAccess.accessMap).some((s) => s === "not_authorized"));

  const showEnhancerRow = recommendation.local_enhancer_cp !== null;

  const enhancerDownloadLabel = recommendation.local_enhancement_supported
    ? "Upgrade"
    : "Download";

  const encoderDeleteButton = (
    <span className={modelCatalogTableStyles.actionButtonTooltipWrap}>
      <DsButton
        appearance="neutral"
        hierarchy="secondary"
        size="md"
        className={modelCatalogActionButtonWideClassName()}
        label="Delete"
        disabled={encoderBusy || !encoderDeleteCp || encoderDeleteBlocked}
        onClick={(event) => {
          event.stopPropagation();
          if (encoderDeleteBlocked) return;
          if (encoderDeleteCp) {
            void deleteCheckpoint(encoderDeleteCp, "encoder");
          }
        }}
      />
    </span>
  );

  const encoderDeleteControl = encoderDeleteBlocked ? (
    <Tooltip
      content={LOCAL_TEXT_ENCODER_REQUIRED_TOOLTIP}
      side="top"
      align="center"
      maxWidth={280}
      portalContainer={modelCatalogSettingsTooltipPortal}
    >
      {encoderDeleteButton}
    </Tooltip>
  ) : (
    encoderDeleteButton
  );

  return (
    <SettingsSubsection
      title="Encoding models"
      description="Local text encoding and optional on-device prompt enhancement. Choose API vs local encoding under General → Text encoding."
      descriptionClassName={contentStyles.apiKeysDescriptionLine}
      descriptionSize="sm"
    >
      <TooltipProvider delay={300}>
        <ModelCatalogTable>
          <Fragment>
            <tr className={modelCatalogRowClassNames()}>
              <ModelCatalogNameCell>
                <ModelCatalogCellText>{encoderDisplayName}</ModelCatalogCellText>
                {encoderInstalled ? <ModelCatalogInstalledCheck /> : null}
              </ModelCatalogNameCell>
              <ModelCatalogSizeCell>
                <ModelCatalogSizeText>
                  {formatBytes(gbToBytes(recommendation.expected_size_gb ?? 0))}
                </ModelCatalogSizeText>
              </ModelCatalogSizeCell>
              <ModelCatalogActionGroup
                action={
                  !encoderInstalled ? (
                    <DsButton
                      appearance="neutral"
                      hierarchy="secondary"
                      size="md"
                      className={modelCatalogActionButtonWideClassName()}
                      label={modelCatalogDownloadButtonLabel(
                        encoderDownloadingCp !== null,
                        downloadPercent(encoderProgress),
                      )}
                      isLoading={encoderDownloadingCp !== null}
                      disabled={
                        encoderDownloadingCp === null &&
                        (!encoderCanDownload || encoderDeleteBusy)
                      }
                      onClick={(event) => {
                        event.stopPropagation();
                        if (encoderCpToDownload) {
                          void startDownload(encoderCpToDownload, "encoder");
                        }
                      }}
                    />
                  ) : (
                    encoderDeleteControl
                  )
                }
              />
            </tr>
            {encoderShowHfGate || encoderError ? (
              <ModelCatalogDetailRow>
                <SettingsStack>
                  {encoderShowHfGate ? (
                    <HfModelAccessGate
                      accessMap={encoderAccess.accessMap}
                      allAuthorized={encoderAccess.allAuthorized}
                      hfAuthStatus={hfAuthStatus}
                      hfAuthPolling={hfAuthPolling}
                      startHuggingFaceLogin={startHuggingFaceLogin}
                      checkError={encoderAccess.checkError}
                      onRetryCheck={encoderAccess.recheckAccess}
                    />
                  ) : null}
                  {encoderError ? (
                    <Text
                      as="p"
                      variant="body"
                      size="xs"
                      className="inline-flex items-center gap-1.5 text-fg-danger"
                    >
                      <AlertCircle className="h-3 w-3 flex-shrink-0" aria-hidden />
                      {encoderError}
                    </Text>
                  ) : null}
                </SettingsStack>
              </ModelCatalogDetailRow>
            ) : null}
          </Fragment>

          {showEnhancerRow ? (
            <Fragment>
              <tr className={modelCatalogRowClassNames()}>
                <ModelCatalogNameCell>
                  <ModelCatalogCellText>{enhancerDisplayName}</ModelCatalogCellText>
                  {enhancerInstalled ? <ModelCatalogInstalledCheck /> : null}
                </ModelCatalogNameCell>
                <ModelCatalogSizeCell>
                  <ModelCatalogSizeText>
                    {formatBytes(
                      gbToBytes(recommendation.local_enhancer_expected_size_gb ?? 0),
                    )}
                  </ModelCatalogSizeText>
                </ModelCatalogSizeCell>
                <ModelCatalogActionGroup
                  action={
                    !enhancerInstalled ? (
                      <DsButton
                        appearance="neutral"
                        hierarchy="secondary"
                        size="md"
                        className={modelCatalogActionButtonWideClassName()}
                        label={
                          enhancerDownloadingCp !== null
                            ? modelCatalogDownloadButtonLabel(
                                true,
                                downloadPercent(enhancerProgress),
                              )
                            : enhancerDownloadLabel
                        }
                        isLoading={enhancerDownloadingCp !== null}
                        disabled={
                          enhancerDownloadingCp === null &&
                          (!enhancerCanDownload || enhancerDeleteBusy)
                        }
                        onClick={(event) => {
                          event.stopPropagation();
                          if (enhancerCpToDownload) {
                            void startDownload(enhancerCpToDownload, "enhancer");
                          }
                        }}
                      />
                    ) : (
                      <DsButton
                        appearance="neutral"
                        hierarchy="secondary"
                        size="md"
                        className={modelCatalogActionButtonWideClassName()}
                        label="Delete"
                        disabled={enhancerBusy || !enhancerDeleteCp}
                        onClick={(event) => {
                          event.stopPropagation();
                          if (enhancerDeleteCp) {
                            void deleteCheckpoint(enhancerDeleteCp, "enhancer");
                          }
                        }}
                      />
                    )
                  }
                />
              </tr>
              {enhancerShowHfGate || enhancerError ? (
                <ModelCatalogDetailRow>
                  <SettingsStack>
                    {enhancerShowHfGate ? (
                      <HfModelAccessGate
                        accessMap={enhancerAccess.accessMap}
                        allAuthorized={enhancerAccess.allAuthorized}
                        hfAuthStatus={hfAuthStatus}
                        hfAuthPolling={hfAuthPolling}
                        startHuggingFaceLogin={startHuggingFaceLogin}
                        checkError={enhancerAccess.checkError}
                        onRetryCheck={enhancerAccess.recheckAccess}
                      />
                    ) : null}
                    {enhancerError ? (
                      <Text
                        as="p"
                        variant="body"
                        size="xs"
                        className="inline-flex items-center gap-1.5 text-fg-danger"
                      >
                        <AlertCircle className="h-3 w-3 flex-shrink-0" aria-hidden />
                        {enhancerError}
                      </Text>
                    ) : null}
                  </SettingsStack>
                </ModelCatalogDetailRow>
              ) : null}
            </Fragment>
          ) : null}
        </ModelCatalogTable>
      </TooltipProvider>
    </SettingsSubsection>
  );
}
