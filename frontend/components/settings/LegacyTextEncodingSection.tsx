import { clsx } from "clsx";
import { AlertCircle } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { flushSync } from "react-dom";

import { Text } from "@/ds/Text/Text";

import type { AppSettings } from "../../contexts/AppSettingsContext";
import { useAppSettings } from "../../contexts/AppSettingsContext";
import { ApiClient, type ApiSuccessOf } from "../../lib/api-client";
import { logger } from "../../lib/logger";

import { SettingsHintWarning } from "./encodingModelDownloads";
import {
  SettingsNumberInput,
  SettingsOptionCard,
  SettingsStack,
  SettingsSubsection,
} from "./SettingsContent";
import contentStyles from "./SettingsContent.module.scss";

export function LegacyTextEncodingSection({
  active,
  settings,
  onOpenLtxApiKeys,
}: {
  active: boolean;
  settings: Pick<
    AppSettings,
    "useLocalTextEncoder" | "hasLtxApiKey" | "promptCacheSize"
  >;
  onOpenLtxApiKeys: () => void;
}) {
  // The recommendation endpoint is local-model-only: in force-API mode it answers 409
  // LOCAL_MODEL_RECOMMENDATIONS_DISABLED_IN_FORCE_API_MODE, and the local/API choice this
  // section offers has no meaning there, so the whole section stands down.
  const { updateSettings, forceApiGenerations, modelsVersion } = useAppSettings();
  const [textEncoderRecommendation, setTextEncoderRecommendation] =
    useState<ApiSuccessOf<"getTextEncoderRecommendation"> | null>(null);

  const apiEncodingSupported = textEncoderRecommendation?.api_encoding_supported ?? true;
  const [localEncoderUi, setLocalEncoderUi] = useState(settings.useLocalTextEncoder);

  useEffect(() => {
    setLocalEncoderUi(settings.useLocalTextEncoder);
  }, [settings.useLocalTextEncoder]);

  const localEncoderSelected = localEncoderUi || !apiEncodingSupported;
  const localEncoderNeedsDownload = Boolean(textEncoderRecommendation?.cp_to_download);

  const applyLocalEncoderUi = (useLocal: boolean) => {
    flushSync(() => setLocalEncoderUi(useLocal));
    updateSettings({ useLocalTextEncoder: useLocal });
  };

  useEffect(() => {
    if (!active || forceApiGenerations) return;

    const fetchRecommendation = async () => {
      const result = await ApiClient.getTextEncoderRecommendation();
      if (!result.ok) {
        logger.error(
          `Failed to fetch text encoder recommendation: ${result.error.message}`,
        );
        return;
      }
      setTextEncoderRecommendation(result.data);
    };

    void fetchRecommendation();
    // modelsVersion so deleting or downloading the encoder from the Models tab refreshes the
    // badges here instead of leaving them stale until Settings is reopened.
  }, [active, forceApiGenerations, modelsVersion]);

  const selectLtxApiEncoding = () => {
    if (!apiEncodingSupported || !localEncoderUi) return;
    if (!settings.hasLtxApiKey) {
      onOpenLtxApiKeys();
      return;
    }
    applyLocalEncoderUi(false);
  };

  const selectLocalEncoding = () => {
    if (localEncoderUi) return;
    applyLocalEncoderUi(true);
  };

  const promptCacheSize = settings.promptCacheSize ?? 100;

  const handlePromptCacheSizeChange = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    // `|| 100` would swallow a deliberate 0, which is how the cache is disabled.
    const parsed = Number.parseInt(event.target.value, 10);
    const size = Number.isNaN(parsed) ? 100 : Math.max(0, Math.min(1000, parsed));
    updateSettings({ promptCacheSize: size });
  };

  const unavailableBadge = (
    <span className={contentStyles.badgeInlineNeutral}>Unavailable</span>
  );
  const requiredBadge = <span className={contentStyles.badgeInline}>Required</span>;

  if (forceApiGenerations) return null;

  return (
    <SettingsSubsection
      showDivider
      title="Text encoding"
      description="How prompts are converted into data the model understands."
      descriptionClassName={contentStyles.apiKeysDescriptionLine}
      descriptionSize="sm"
    >
      <SettingsStack>
        <div className={contentStyles.textEncodingOptionRow}>
          <TextEncodingChoiceCard
            selected={!localEncoderSelected}
            disabled={!apiEncodingSupported}
            onClick={selectLtxApiEncoding}
            title="LTX API"
            badge={apiEncodingSupported ? undefined : unavailableBadge}
            description="Cloud encoding (~1 s). Requires an LTX API key on the API Keys tab."
            footer={
              <div className={contentStyles.textEncodingCacheBlock}>
                <span className={contentStyles.textEncodingCacheLabel}>
                  Prompt cache
                </span>
                <div
                  className={contentStyles.textEncodingCacheHintRow}
                  onPointerDown={(event) => event.stopPropagation()}
                  onClick={(event) => event.stopPropagation()}
                >
                  <p className={contentStyles.textEncodingCacheHint}>
                    Skip repeat API encoding for identical prompts.
                  </p>
                  <SettingsNumberInput
                    className={contentStyles.textEncodingCacheInput}
                    min={0}
                    max={1000}
                    value={promptCacheSize}
                    onChange={handlePromptCacheSizeChange}
                    // The enclosing card activates on Enter/Space; without this, editing the
                    // cache and pressing Enter would switch encoding from local to API.
                    onKeyDown={(event) => event.stopPropagation()}
                    disabled={!settings.hasLtxApiKey}
                    aria-label="Prompt cache size"
                  />
                </div>
              </div>
            }
          />

          <TextEncodingChoiceCard
            selected={localEncoderSelected}
            onClick={selectLocalEncoding}
            title="Local encoder"
            badge={!apiEncodingSupported ? requiredBadge : undefined}
            description={`Runs on this computer (slower than the API). Requires a ~${textEncoderRecommendation?.expected_size_gb ?? 25} GB download from the Models tab.`}
          />
        </div>

        {localEncoderSelected && localEncoderNeedsDownload ? (
          <SettingsHintWarning>
            Download the local text encoder in Settings → Models.
          </SettingsHintWarning>
        ) : null}

        {!apiEncodingSupported && (
          <LegacyHintWarning>
            Not available for{" "}
            {textEncoderRecommendation?.ltx_version_label ?? "this model"} — use
            the local encoder for this version.
          </LegacyHintWarning>
        )}

        {apiEncodingSupported &&
          !settings.useLocalTextEncoder &&
          !settings.hasLtxApiKey && (
            <LegacyHintWarning>
              Add an LTX API key on the API Keys tab to use cloud encoding.
            </LegacyHintWarning>
          )}
      </SettingsStack>
    </SettingsSubsection>
  );
}

function TextEncodingChoiceCard({
  selected,
  disabled = false,
  onClick,
  title,
  badge,
  description,
  footer,
}: {
  selected: boolean;
  disabled?: boolean;
  onClick: () => void;
  title: string;
  badge?: ReactNode;
  description: string;
  footer?: ReactNode;
}) {
  return (
    <div className={contentStyles.textEncodingOptionCell}>
      <SettingsOptionCard
        selected={selected}
        disabled={disabled}
        onClick={onClick}
        activateOnPointerDown
        applySelectedClass={false}
        className={clsx(
          contentStyles.textEncodingCard,
          selected && contentStyles.textEncodingCardSelected,
        )}
        innerClassName={contentStyles.textEncodingCardHead}
        footerClassName={contentStyles.textEncodingCardFooter}
        footer={footer}
      >
        <span
          className={clsx(
            contentStyles.textEncodingRadio,
            selected && contentStyles.textEncodingRadioSelected,
          )}
          aria-hidden
        />
        <div className={contentStyles.textEncodingCardCopy}>
          <div className={contentStyles.textEncodingCardTitleRow}>
            <span className={contentStyles.textEncodingCardTitle}>{title}</span>
            {badge}
          </div>
          <p className={contentStyles.textEncodingCardDescription}>{description}</p>
        </div>
      </SettingsOptionCard>
    </div>
  );
}

function LegacyHintWarning({ children }: { children: React.ReactNode }) {
  return (
    <Text
      as="p"
      variant="body"
      size="xs"
      className={contentStyles.hintWarning}
    >
      <span className="inline-flex items-start gap-1.5">
        <AlertCircle className="h-3 w-3 flex-shrink-0 mt-0.5" aria-hidden />
        <span>{children}</span>
      </span>
    </Text>
  );
}
