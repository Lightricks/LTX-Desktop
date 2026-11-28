import type {
  SettingsScrollTarget,
} from "./settings-navigation";

export function settingsSectionElementId(section: SettingsScrollTarget): string {
  return `settings-section-${section}`;
}
