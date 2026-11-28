import { Text } from "@ds/Text/Text";

import { Info } from "../../../components/shared/Info/Info";
import formStyles from "../FeatureFormView.module.scss";

// Wider than the 200px default, and never wider than the screen minus a margin.
const FIELD_TOOLTIP_MAX_WIDTH = "min(20rem, calc(100vw - 2rem))";

function tooltipContent(tooltip: string) {
  const paragraphs = tooltip
    .split(/\n\s*\n/)
    .map((paragraph) => paragraph.trim())
    .filter((paragraph) => paragraph.length > 0);
  if (paragraphs.length < 2) return tooltip;
  return (
    <span className={formStyles.tooltipParagraphs}>
      {paragraphs.map((paragraph) => (
        <span key={paragraph} className={formStyles.tooltipParagraph}>
          {paragraph}
        </span>
      ))}
    </span>
  );
}

/** The label of a form field, with its help text on hover when the field has a tooltip. */
export function FieldLabel({ label, tooltip }: { label: string; tooltip?: string }) {
  return (
    <Text as="span" variant="body" size="md" className={formStyles.fieldLabel}>
      {label}
      {tooltip ? (
        <Info content={tooltipContent(tooltip)} maxWidth={FIELD_TOOLTIP_MAX_WIDTH} />
      ) : null}
    </Text>
  );
}
