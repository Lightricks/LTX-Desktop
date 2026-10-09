import type { AppSettings } from "../../contexts/AppSettingsContext";

import { SettingToggle } from "./SettingToggle";
import { SettingsStack, SettingsSubsection } from "./SettingsContent";
import contentStyles from "./SettingsContent.module.scss";

export function CudaPerformanceSection({
  settings,
  onToggleTorchCompile,
  onToggleDiffusionStageCache,
}: {
  settings: Pick<AppSettings, "useTorchCompile" | "diffusionStageCacheEnabled">;
  onToggleTorchCompile: () => void;
  onToggleDiffusionStageCache: () => void;
}) {
  return (
    <SettingsSubsection
      showDivider
      title="Performance (CUDA)"
      description="Experimental options for NVIDIA GPUs. Restart may be required."
      descriptionClassName={contentStyles.apiKeysDescriptionLine}
      descriptionSize="sm"
    >
      <SettingsStack>
        <SettingToggle
          rowLayout="preference"
          title="Torch Compile"
          description={
            <>
              Compiles the model for optimized inference. First run can take several
              minutes; later runs may be 20–40% faster.
            </>
          }
          enabled={settings.useTorchCompile}
          onToggle={onToggleTorchCompile}
        />
        <SettingToggle
          rowLayout="preference"
          title="Diffusion stage cache"
          description="Reuses the transformer between stages in one generation on high-VRAM GPUs (32 GB+)."
          enabled={settings.diffusionStageCacheEnabled}
          onToggle={onToggleDiffusionStageCache}
        />
      </SettingsStack>
    </SettingsSubsection>
  );
}
