import type { AppSettings } from "../../contexts/AppSettingsContext";

import { SettingsStack } from "./SettingsContent";
import { SettingToggle } from "./SettingToggle";

export function EnhanceSettingsSection({
  settings,
  onToggleAutoEnhance,
  onToggleApiEnhance,
}: {
  settings: Pick<
    AppSettings,
    "exploreAutoEnhancePrompts" | "promptEnhancerEnabled" | "hasLtxApiKey"
  >;
  onToggleAutoEnhance: () => void;
  onToggleApiEnhance: () => void;
}) {
  const apiEnabled = settings.promptEnhancerEnabled;

  return (
    <SettingsStack loose>
      <SettingToggle
        rowLayout="preference"
        title="Auto-enhance on local generation"
        description={
          settings.exploreAutoEnhancePrompts
            ? "Rewrites prompts automatically before local Generate in Text to Video, Image to Video, Audio to Video, and style recipes. Gen Space is unchanged."
            : "Use prompts as typed, or tap Enhance next to the prompt. Applies to Text to Video, Image to Video, Audio to Video, and style recipes. Gen Space is unchanged."
        }
        enabled={settings.exploreAutoEnhancePrompts}
        onToggle={onToggleAutoEnhance}
      />

      <SettingToggle
        rowLayout="preference"
        title="Auto-enhance on API generation"
        description={
          settings.hasLtxApiKey
            ? apiEnabled
              ? "LTX API rewrites prompts before Text to Video, Image to Video, and Audio to Video generation."
              : "API prompts are sent as written for Text to Video, Image to Video, and Audio to Video."
            : "Requires an LTX API key on the API Keys tab."
        }
        enabled={apiEnabled}
        onToggle={onToggleApiEnhance}
      />
    </SettingsStack>
  );
}
