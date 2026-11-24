import { Monitor, Moon, Sun } from "lucide-react";

import {
  type ThemePreference,
  useTheme,
} from "@/ds/styles/themes/useTheme";

import { SlidingSegmentControl } from "../../ltx-io/components/SlidingSegmentControl/SlidingSegmentControl";

import styles from "./SettingsContent.module.scss";

const THEME_SEGMENTS: {
  value: ThemePreference;
  label: string;
  icon: typeof Monitor;
}[] = [
  { value: "system", label: "System", icon: Monitor },
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
];

export function ThemePreferenceSetting() {
  const { themePreference, setThemePreference } = useTheme();

  return (
    <div className={styles.themePreferenceRow}>
      <span className={styles.themePreferenceLabel}>Theme</span>
      <SlidingSegmentControl
        className={styles.themeSegmentControl}
        aria-label="Theme"
        size="sm"
        selectedSegment={themePreference}
        onSegmentChanged={setThemePreference}
        segments={THEME_SEGMENTS.map(({ value, label, icon: Icon }) => ({
          value,
          label: (
            <span className={styles.themeSegmentIcon} title={label}>
              <span className={styles.themeSegmentSrOnly}>{label}</span>
              <Icon className={styles.themeSegmentGlyph} aria-hidden />
            </span>
          ),
        }))}
      />
    </div>
  );
}
