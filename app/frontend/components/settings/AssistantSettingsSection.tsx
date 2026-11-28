import { useState } from "react";

import { useAppSettings } from "@/contexts/AppSettingsContext";
import { useAssistantModels } from "@/hooks/use-assistant-models";
import { formatBytes } from "@/lib/format";

import {
  SettingsChoiceCard,
  SettingsHint,
  SettingsSelect,
  SettingsStack,
  SettingsSubsection,
} from "./SettingsContent";
import styles from "./SettingsContent.module.scss";

export function AssistantSettingsSection({
  onOpenGeminiApiKeys,
}: {
  onOpenGeminiApiKeys: () => void;
}) {
  const { settings, selectAssistant, assistantSelectionPending: saving } = useAppSettings();
  const [error, setError] = useState<string | null>(null);
  const select = (...args: Parameters<typeof selectAssistant>) => {
    setError(null);
    selectAssistant(...args).catch((err: Error) => setError(err.message));
  };
  const local = settings.assistantProvider === "local";
  const models = useAssistantModels();
  const installed =
    !models.isError && models.data?.available
      ? models.data.models.filter((model) => model.installed)
      : [];
  const selected =
    installed.find((model) => model.model === settings.assistantLocalModel) ??
    installed[0];
  const configuredModel = models.data?.models.find(
    (model) => model.model === settings.assistantLocalModel,
  );
  const downloadSize = configuredModel
    ? `~${formatBytes(configuredModel.size_bytes)} `
    : "";

  return (
    <SettingsSubsection
      showDivider
      title="Assistant"
      description="The model the assistant uses on Desktop and Remote. You can also change it from the assistant's composer."
    >
      <SettingsStack>
        <div className={styles.textEncodingOptionRow}>
          <SettingsChoiceCard
            selected={!local}
            disabled={saving}
            title="Gemini API"
            description="Cloud assistant. Requires a Gemini API key on the API Keys tab."
            onClick={() => {
              if (!settings.hasGeminiApiKey) {
                onOpenGeminiApiKeys();
                return;
              }
              select("gemini");
            }}
          />
          <SettingsChoiceCard
            selected={local}
            disabled={!selected || saving}
            title="Local model"
            description={`Runs on this computer. Requires a ${downloadSize}model download from the Models tab.`}
            onClick={() => {
              if (selected) select("local", selected.model);
            }}
            footer={
              <div
                className={styles.textEncodingCacheBlock}
                onPointerDown={(event) => event.stopPropagation()}
                onClick={(event) => event.stopPropagation()}
                onKeyDown={(event) => event.stopPropagation()}
              >
                <label
                  htmlFor="assistant-local-model"
                  className={styles.textEncodingCacheLabel}
                >
                  Model
                </label>
                <SettingsSelect
                  id="assistant-local-model"
                  value={settings.assistantLocalModel}
                  onChange={(event) => {
                    const model = installed.find(
                      (item) => item.model === event.target.value,
                    );
                    if (model) select(settings.assistantProvider, model.model);
                  }}
                  disabled={installed.length === 0 || saving}
                >
                  {models.data?.models.map((model) => (
                    <option
                      key={model.model}
                      value={model.model}
                      disabled={!model.installed}
                    >
                      {model.name}
                      {model.installed ? "" : " (not downloaded)"}
                    </option>
                  ))}
                </SettingsSelect>
              </div>
            }
          />
        </div>
        {error && <SettingsHint variant="warning">{error}</SettingsHint>}
        {(models.error || models.data?.error) && (
          <SettingsHint variant="warning">
            {models.error?.message ?? models.data?.error}
          </SettingsHint>
        )}
      </SettingsStack>
    </SettingsSubsection>
  );
}
