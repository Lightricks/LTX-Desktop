import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { useCallback, useEffect, useRef, type ReactNode } from "react";

import { openExternalBrowserUrl } from "@/components/home/home-external-links";
import { formatBytes } from "@/lib/format";
import { catalogVariantKey } from "@/lib/lora-library";
import { useIcLoras, useLoraCatalog } from "@/hooks/use-catalog";
import { useHfAuth } from "@/hooks/use-hf-auth";

import { showSuccessToast } from "../../components/shared/Toast/toastService";
import type { CatalogVariantChoice } from "../../lib/catalogVariantChoice";
import { useExploreRuntime } from "../../runtime/ExploreRuntime";
import {
  LORA_DOWNLOADED_TOAST_COPY,
  LORA_DOWNLOADED_TOAST_MS,
  LORA_GATED_DOWNLOAD_COPY,
  isHuggingFaceGatedError,
  loraDownloadErrorCopy,
} from "./loraDownloadError";
import { listedIcLoraEntry, listedLoraEntry, type ListedEntry } from "./listedCatalogEntry";
import { nextGenerateAfterDismiss } from "./nextGenerateAfterDismiss";
import styles from "./CatalogInstallPreflight.module.scss";

export type CatalogGenerateControls = {
  generateLabel: string | undefined;
  generateHint: string | undefined;
  generateBusy: boolean;
  generateEnabled: boolean;
  actionError: ReactNode;
  interceptGenerate: (proceed: () => void) => void;
  onCancelGenerate: () => void;
  defaultLoraStrength: number | undefined;
  defaultAudioMode: "source" | "generated" | "off" | undefined;
  /** Read after a download, so the form sends the variant that is now on disk. */
  variants: CatalogVariantChoice[];
};

type CatalogKind = "lora" | "ic-lora";

type CatalogRows = { id: string; downloaded: boolean }[];

type CatalogSource = {
  catalogStatus: "loading" | "error" | "loaded";
  cancelDownload: () => void;
  downloadingKey: string | null;
  progress: number;
  downloadError: { key: string; message: string } | null;
  items: ListedEntry[];
  refresh: () => Promise<CatalogRows | null>;
  download: (catalogId: string) => void;
};

function useCatalogKind(
  kind: CatalogKind,
  api: Parameters<typeof useIcLoras>[1] & Parameters<typeof useLoraCatalog>[1],
  canInstallCatalog: boolean,
): CatalogSource {
  const ic = useIcLoras(kind === "ic-lora", api, {
    trackDownloads: canInstallCatalog,
  });
  const lora = useLoraCatalog(kind === "lora", api);
  switch (kind) {
    case "ic-lora":
      return {
        catalogStatus: ic.catalogStatus,
        cancelDownload: ic.cancelDownload,
        downloadingKey: ic.downloadingKey,
        progress: ic.progress,
        downloadError: ic.downloadError,
        items: ic.icLoras.map(listedIcLoraEntry),
        refresh: async () =>
          (await ic.refresh())?.map((entry) => ({
            id: entry.ic_lora.id,
            downloaded: entry.downloaded,
          })) ?? null,
        download: (catalogId) => {
          void ic.downloadIcLora(catalogId);
        },
      };
    case "lora":
      return {
        catalogStatus: lora.catalogStatus,
        cancelDownload: lora.cancelDownload,
        downloadingKey: lora.downloadingKey,
        progress: lora.progress,
        downloadError: lora.downloadError,
        items: lora.loras.map(listedLoraEntry),
        refresh: async () =>
          (await lora.refresh())?.map((entry) => ({
            id: entry.lora.id,
            downloaded: entry.downloaded,
          })) ?? null,
        download: (catalogId) => {
          void lora.downloadLora(catalogId);
        },
      };
    default: {
      const unreachable: never = kind;
      return unreachable;
    }
  }
}

export function PreflightCard({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <div
      className={styles.preflight}
      role="dialog"
      aria-modal="true"
      aria-labelledby="lora-preflight-title"
      aria-describedby="lora-preflight-description"
    >
      <div className={styles.card}>
        <Text
          as="h2"
          id="lora-preflight-title"
          variant="heading"
          size="lg"
          align="center"
          className={styles.title}
        >
          {title}
        </Text>
        <Text
          as="p"
          id="lora-preflight-description"
          variant="body"
          size="md"
          align="center"
          className={styles.description}
        >
          {description}
        </Text>
        {children}
      </div>
    </div>
  );
}

export function LockedWorkflow({
  children,
  overlay,
}: {
  children: ReactNode;
  overlay: ReactNode;
}) {
  return (
    <div className={styles.locked}>
      <div
        className={styles.workflow}
        aria-hidden="true"
        {...{ inert: "" }}
      >
        {children}
      </div>
      {overlay}
    </div>
  );
}

/**
 * Catalog download, Hugging Face sign-in, and the phone-cannot-install gate.
 * One second Generate click cannot start a second download.
 */
export function CatalogInstallPreflight({
  kind,
  catalogId,
  title,
  hold,
  renderForm,
}: {
  kind: CatalogKind;
  catalogId: string;
  title: string;
  hold: boolean;
  renderForm: (controls: CatalogGenerateControls) => ReactNode;
}) {
  const { api, catalogInstall } = useExploreRuntime();
  const canInstallCatalog = catalogInstall != null;
  const catalog = useCatalogKind(kind, api, canInstallCatalog);
  const {
    catalogStatus,
    cancelDownload,
    downloadingKey,
    progress,
    downloadError,
    items,
    refresh,
    download,
  } = catalog;
  const item = items.find((entry) => entry.id === catalogId);
  const { hfAuthStatus, startHuggingFaceLogin } = useHfAuth(
    item?.requiresHfLogin ?? false,
    api,
    catalogInstall?.openHuggingFaceAuth ?? null,
  );
  const downloadKey = catalogVariantKey(catalogId);
  const isDownloading = downloadingKey === downloadKey;
  const otherDownloading = downloadingKey !== null && downloadingKey !== downloadKey;
  const needsDownload = item != null && !item.downloaded && canInstallCatalog;
  // Home IC-LoRAs stay on the form, the same as a style LoRA. Sign-in happens
  // from Download and generate instead of a screen that hides the form.
  const signInFromForm =
    kind === "ic-lora" &&
    canInstallCatalog &&
    Boolean(item?.requiresHfLogin) &&
    hfAuthStatus !== "authenticated";
  const pendingGenerateRef = useRef<(() => void) | null>(null);
  const waitingForHfRef = useRef(false);
  const proceedAfterAuthRef = useRef<(() => void) | null>(null);
  const wasDownloadedRef = useRef<boolean | null>(null);
  const skipManageToastRef = useRef(false);
  const resolveGenerationRef = useRef(false);

  useEffect(() => {
    if (item == null && catalogStatus !== "loaded") return;
    const downloaded = Boolean(item?.downloaded);
    if (wasDownloadedRef.current === null) {
      wasDownloadedRef.current = downloaded;
      return;
    }
    const justDownloaded = downloaded && !wasDownloadedRef.current;
    wasDownloadedRef.current = downloaded;
    if (!justDownloaded || skipManageToastRef.current || !canInstallCatalog) return;
    showSuccessToast(LORA_DOWNLOADED_TOAST_COPY, {
      hideIcon: true,
      autoCloseAfter: LORA_DOWNLOADED_TOAST_MS,
      action: {
        label: "Settings",
        onClick: () => {
          window.dispatchEvent(
            new CustomEvent("open-settings", { detail: { tab: "models" } }),
          );
        },
      },
    });
  }, [canInstallCatalog, catalogStatus, item, item?.downloaded]);

  useEffect(() => {
    if (!item?.downloaded || pendingGenerateRef.current == null) return;
    const proceed = pendingGenerateRef.current;
    pendingGenerateRef.current = null;
    proceed();
  }, [item?.downloaded]);

  useEffect(() => {
    if (downloadError) pendingGenerateRef.current = null;
  }, [downloadError]);

  const interceptGenerate = useCallback(
    (proceed: () => void) => {
      if (pendingGenerateRef.current || resolveGenerationRef.current) return;
      if (!needsDownload && !signInFromForm) {
        proceed();
        return;
      }
      if (signInFromForm) {
        waitingForHfRef.current = true;
        proceedAfterAuthRef.current = proceed;
        void startHuggingFaceLogin();
        return;
      }
      resolveGenerationRef.current = true;
      skipManageToastRef.current = false;
      void (async () => {
        try {
          const rows = await refresh();
          if (skipManageToastRef.current || rows == null) return;
          const action = nextGenerateAfterDismiss(rows, catalogId);
          if (action === "proceed") {
            proceed();
            return;
          }
          pendingGenerateRef.current = proceed;
          download(catalogId);
        } finally {
          resolveGenerationRef.current = false;
        }
      })();
    },
    [catalogId, download, needsDownload, refresh, signInFromForm, startHuggingFaceLogin],
  );

  useEffect(() => {
    if (signInFromForm || !waitingForHfRef.current) return;
    waitingForHfRef.current = false;
    const proceed = proceedAfterAuthRef.current;
    proceedAfterAuthRef.current = null;
    if (proceed == null) return;
    if (!needsDownload) {
      proceed();
      return;
    }
    pendingGenerateRef.current = proceed;
    download(catalogId);
  }, [catalogId, download, needsDownload, signInFromForm]);

  const handleCancelDownload = useCallback(() => {
    pendingGenerateRef.current = null;
    skipManageToastRef.current = true;
    cancelDownload();
  }, [cancelDownload]);

  const generateLabel = needsDownload
    ? isDownloading
      ? `Downloading… ${Math.round(progress)}%`
      : "Download and generate"
    : undefined;
  const generateHint =
    needsDownload && !isDownloading && item?.sizeBytes
      ? `(${formatBytes(item.sizeBytes)})`
      : undefined;
  const downloadActionError =
    downloadError?.key === downloadKey ? (
      isHuggingFaceGatedError(downloadError.message) ? (
        <>
          {LORA_GATED_DOWNLOAD_COPY}
          {item?.repoId ? (
            <Button
              appearance="neutral"
              hierarchy="secondary"
              size="sm"
              label="Request access"
              onClick={() =>
                openExternalBrowserUrl(`https://huggingface.co/${item.repoId}`)
              }
            />
          ) : null}
        </>
      ) : (
        loraDownloadErrorCopy(downloadError.message)
      )
    ) : null;

  if (catalogStatus === "loading") {
    return (
      <LockedWorkflow
        overlay={
          <PreflightCard title={title} description="Checking model status…">
            <span />
          </PreflightCard>
        }
      >
        <span />
      </LockedWorkflow>
    );
  }

  const form = renderForm({
    generateLabel,
    generateHint,
    generateBusy: isDownloading,
    generateEnabled: !otherDownloading,
    actionError: downloadActionError,
    interceptGenerate,
    onCancelGenerate: handleCancelDownload,
    defaultLoraStrength: item?.defaultLoraStrength,
    defaultAudioMode: item?.defaultAudioMode,
    variants: item?.variants ?? [],
  });

  if (catalogStatus === "error") {
    return (
      <LockedWorkflow
        overlay={
          <PreflightCard
            title="Couldn't load this style"
            description="Check your connection and try again."
          >
            <Button
              appearance="brand"
              hierarchy="primary"
              size="lg"
              label="Retry"
              onClick={() => {
                void refresh();
              }}
            />
          </PreflightCard>
        }
      >
        {form}
      </LockedWorkflow>
    );
  }

  if (!item) {
    return (
      <LockedWorkflow
        overlay={
          <PreflightCard
            title={`${title} isn't available`}
            description="This style isn't available with the LTX models installed on this computer."
          >
            <span />
          </PreflightCard>
        }
      >
        {form}
      </LockedWorkflow>
    );
  }

  if (item.requiresHfLogin && hfAuthStatus !== "authenticated" && !signInFromForm) {
    return (
      <LockedWorkflow
        overlay={
          <PreflightCard
            title="Sign in to download this model"
            description={
              canInstallCatalog
                ? `A Hugging Face account is required before you can generate with ${title}.`
                : "This style needs a Hugging Face account. Sign in on the desktop app, then generate from here."
            }
          >
            {canInstallCatalog ? (
              <Button
                appearance="brand"
                hierarchy="primary"
                size="lg"
                label="Sign in to Hugging Face"
                onClick={() => void startHuggingFaceLogin()}
              />
            ) : (
              <span />
            )}
          </PreflightCard>
        }
      >
        {form}
      </LockedWorkflow>
    );
  }

  if (!item.downloaded && !canInstallCatalog) {
    return (
      <LockedWorkflow
        overlay={
          <PreflightCard
            title="Download this model to generate"
            description="Download this on the desktop app, then generate from here."
          >
            <span />
          </PreflightCard>
        }
      >
        {form}
      </LockedWorkflow>
    );
  }

  if (hold) return null;
  return form;
}
