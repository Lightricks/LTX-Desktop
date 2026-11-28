import type { AppSettings } from "../../contexts/AppSettingsContext";

import { SettingToggle } from "./SettingToggle";
import { SettingsStack, SettingsSubsection } from "./SettingsContent";

export function HardwareBudgetSection({
  settings,
  onSetCautious,
  onSetAskBeforeRisky,
  onSetNeverSplit,
  saving = false,
  neverSplitSaving = false,
  syncError,
}: {
  settings: Pick<AppSettings, "budgetMode" | "neverSplitIntoParts" | "skipBudgetRiskConfirm">;
  onSetCautious: (cautious: boolean) => void;
  onSetAskBeforeRisky: (ask: boolean) => void;
  onSetNeverSplit: (neverSplit: boolean) => void;
  saving?: boolean;
  neverSplitSaving?: boolean;
  syncError?: string | null;
}) {
  const cautious = settings.budgetMode === "cautious";
  const askBeforeRisky = !settings.skipBudgetRiskConfirm;
  return (
    <SettingsSubsection
      title="Memory limit"
      description="The Cautious and Never split switches move when the save finishes."
      alert={syncError}
    >
      <SettingsStack>
        <SettingToggle
          rowLayout="preference"
          title="Cautious mode"
          description="Block jobs that may run out of memory or were never tested on this machine, instead of asking before you run them."
          enabled={cautious}
          disabled={saving}
          onToggle={() => onSetCautious(!cautious)}
        />
        <SettingToggle
          rowLayout="preference"
          title="Ask before risky generates"
          description="Confirm before running a job that may run out of memory or stall the app."
          enabled={askBeforeRisky}
          disabled={cautious}
          onToggle={() => onSetAskBeforeRisky(!askBeforeRisky)}
        />
        <SettingToggle
          rowLayout="preference"
          title="Never split into parts"
          description="Make every video all at once, so there are no joins between parts. Long or high-resolution videos then need much more memory: some lengths won't be offered, and in Brave mode some will need more than this computer has."
          enabled={settings.neverSplitIntoParts}
          disabled={neverSplitSaving}
          onToggle={() => onSetNeverSplit(!settings.neverSplitIntoParts)}
        />
      </SettingsStack>
    </SettingsSubsection>
  );
}
