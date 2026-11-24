import { useId, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";

import {
  NAG_KNOB_RANGES,
  NAG_PRESETS,
  nagStrengthSummary,
  type NagKnobs,
  type NagStrength,
} from "../../../lib/nagStrength";
import formStyles from "../FeatureFormView.module.scss";
import styles from "./NagStrengthControl.module.scss";

const KNOBS: { key: keyof NagKnobs; label: string; hint: string }[] = [
  {
    key: "alpha",
    label: "α (alpha)",
    hint: "How much of the guidance is kept. The most effective knob. Above 0.65 a scene can break.",
  },
  {
    key: "tau",
    label: "τ (tau)",
    hint: "Caps the guidance per token. Raising it adds strength, like a little more α.",
  },
  {
    key: "scale",
    label: "Scale",
    hint: "Pushes away from the negative. Above about 11 it changes little.",
  },
];

/**
 * "Negative prompt strength" under the negative prompt, behind the "Per-run strength controls" setting.
 * Collapsed by default: most runs keep the Settings strength.
 */
export function NagStrengthControl({
  value,
  onChange,
}: {
  value: NagStrength;
  onChange: (next: NagStrength) => void;
}) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const preset = NAG_PRESETS.find((entry) => entry.id === value.preset);

  return (
    <div className={styles.root}>
      <button
        type="button"
        className={styles.toggle}
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((current) => !current)}
      >
        {open ? <ChevronDown size={14} aria-hidden /> : <ChevronRight size={14} aria-hidden />}
        <Text as="span" variant="body" size="md" className={styles.toggleLabel}>
          Negative prompt strength
        </Text>
        <Text as="span" variant="body" size="md" className={styles.summary}>
          {nagStrengthSummary(value)}
        </Text>
      </button>
      {open ? (
        <div id={panelId} className={styles.panel}>
          <div className={styles.presets} role="group" aria-label="Strength preset">
            {NAG_PRESETS.map((entry) => (
              <Button
                key={entry.id}
                appearance="neutral"
                hierarchy={entry.id === value.preset ? "primary" : "secondary"}
                size="sm"
                label={entry.label}
                aria-pressed={entry.id === value.preset}
                onClick={() => onChange({ ...value, preset: entry.id })}
              />
            ))}
          </div>
          {preset ? (
            <Text as="span" variant="body" size="md" className={styles.description}>
              {preset.description}
            </Text>
          ) : null}
          {value.preset === "custom" ? (
            <div className={styles.knobs}>
              {KNOBS.map((knob) => {
                const range = NAG_KNOB_RANGES[knob.key];
                const current = value.custom[knob.key];
                return (
                  <div key={knob.key} className={styles.knob}>
                    <Text as="span" variant="body" size="md" className={formStyles.fieldLabel}>
                      {knob.label}
                    </Text>
                    <div className={formStyles.sliderControl}>
                      <input
                        type="range"
                        className={formStyles.slider}
                        min={range.min}
                        max={range.max}
                        step={range.step}
                        value={current}
                        aria-label={knob.label}
                        onChange={(event) =>
                          onChange({
                            ...value,
                            custom: { ...value.custom, [knob.key]: Number(event.currentTarget.value) },
                          })
                        }
                      />
                      <span className={formStyles.sliderValue}>
                        {current.toFixed(range.step < 0.1 ? 2 : 1)}
                      </span>
                    </div>
                    <Text as="span" variant="body" size="md" className={styles.hint}>
                      {knob.hint}
                    </Text>
                  </div>
                );
              })}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
