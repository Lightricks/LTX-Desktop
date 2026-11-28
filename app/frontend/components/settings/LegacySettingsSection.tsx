import { useAppSettings, type AppSettings } from "../../contexts/AppSettingsContext";
import { useGenerationSeed } from "../../hooks/use-generation-seed";
import { ApiClient } from "../../lib/api-client";
import { SeedField } from "../SeedField";
import { LegacyDefaultBaseModelSection } from "./LegacyDefaultBaseModelSection";
import { LegacyProjectAssetsSection } from "./LegacyProjectAssetsSection";
import { SettingToggle } from "./SettingToggle";
import {
  SettingsHint,
  SettingsStack,
  SettingsSubsection,
} from "./SettingsContent";
import contentStyles from "./SettingsContent.module.scss";
export function LegacySettingsSection({
  settings,
  onSettingsChange,
  onOpenLtxApiKeys,
  onOpenFalApiKeys,
}: {
  settings: AppSettings;
  onSettingsChange: (next: AppSettings) => void;
  onOpenLtxApiKeys: () => void;
  onOpenFalApiKeys: () => void;
}) {
  // In force-API mode local generation is unavailable, so the rows that only steer it are
  // noise. Project assets path and Seed still apply, which is why this tab stays reachable.
  const { forceApiGenerations } = useAppSettings();
  const generationSeed = useGenerationSeed(ApiClient);

  return (
    <SettingsStack loose>
      <LegacyProjectAssetsSection />
      {!forceApiGenerations ? <LegacyDefaultBaseModelSection /> : null}

      {!forceApiGenerations ? (
        <SettingsSubsection
          showDivider
          title="Video generation"
          description="When enabled, legacy video tools prefer the LTX API if a key is configured."
          descriptionClassName={contentStyles.apiKeysDescriptionLine}
          descriptionSize="sm"
        >
          <SettingToggle
            rowLayout="preference"
            title="Generate with LTX API"
            description="Cloud video generation instead of local GPU inference."
            enabled={settings.userPrefersLtxApiVideoGenerations}
            onToggle={() => {
              if (!settings.hasLtxApiKey) {
                onOpenLtxApiKeys();
                return;
              }
              onSettingsChange({
                ...settings,
                userPrefersLtxApiVideoGenerations:
                  !settings.userPrefersLtxApiVideoGenerations,
              });
            }}
          />
          {!settings.hasLtxApiKey && settings.userPrefersLtxApiVideoGenerations ? (
            <SettingsHint variant="warning">
              Add an LTX API key on the API Keys tab.
            </SettingsHint>
          ) : null}
        </SettingsSubsection>
      ) : null}

      {!forceApiGenerations ? (
        <SettingsSubsection
          showDivider
          title="Image generation"
          description="When enabled, legacy image tools prefer the FAL API if a key is configured."
          descriptionClassName={contentStyles.apiKeysDescriptionLine}
          descriptionSize="sm"
        >
          <SettingToggle
            rowLayout="preference"
            title="Generate with FAL API"
            description="Cloud image generation and editing via FAL."
            enabled={settings.userPrefersFalApiImageGenerations}
            onToggle={() => {
              if (!settings.hasFalApiKey) {
                onOpenFalApiKeys();
                return;
              }
              onSettingsChange({
                ...settings,
                userPrefersFalApiImageGenerations:
                  !settings.userPrefersFalApiImageGenerations,
              });
            }}
          />
          {!settings.hasFalApiKey && settings.userPrefersFalApiImageGenerations ? (
            <SettingsHint variant="warning">
              Add a FAL API key on the API Keys tab.
            </SettingsHint>
          ) : null}
        </SettingsSubsection>
      ) : null}

      <SettingsSubsection
        showDivider
        title="Seed"
        description="Control randomness for reproducible generations."
        descriptionClassName={contentStyles.apiKeysDescriptionLine}
        descriptionSize="sm"
      >
        <SettingsStack>
          <SettingToggle
            rowLayout="preference"
            title="Lock seed"
            description="Use the same seed each time instead of a random value. The Seed field on Generate forms and in the remote app is the same one."
            enabled={generationSeed.locked}
            onToggle={() => generationSeed.setLocked(!generationSeed.locked)}
            disabled={!generationSeed.ready}
          />
          <SeedField
            value={generationSeed.seed}
            onChange={generationSeed.setSeed}
            ariaLabel="Seed value"
            disabled={!generationSeed.ready}
          />
        </SettingsStack>
      </SettingsSubsection>
    </SettingsStack>
  );
}
