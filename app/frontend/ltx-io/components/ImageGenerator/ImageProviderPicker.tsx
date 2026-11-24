import { Text } from "@ds/Text/Text";

import { SlidingSegmentControl } from "../SlidingSegmentControl/SlidingSegmentControl";
import type { ImageProvider, ImageProviderOption } from "./imageProviderAvailability";

export function ImageProviderPicker({
  options,
  provider,
  blockedReason,
  onPick,
  labelClassName,
  errorClassName,
  segmentClassName,
  fullWidth,
}: {
  options: readonly ImageProviderOption[];
  provider: ImageProvider;
  blockedReason: string | null;
  onPick: (provider: ImageProvider) => void;
  labelClassName?: string;
  errorClassName?: string;
  segmentClassName?: string;
  /** The segments share the full width of the column. */
  fullWidth?: boolean;
}) {
  return (
    <>
      <Text as="span" variant="body" size="md" className={labelClassName}>
        Run on
      </Text>
      <SlidingSegmentControl
        aria-label="Image provider"
        segments={options.map((option) => ({
          value: option.provider,
          label: option.label,
          disabled: option.blockedReason !== null,
          disabledTooltip: option.blockedReason,
        }))}
        selectedSegment={provider}
        onSegmentChanged={onPick}
        className={segmentClassName}
        fullWidth={fullWidth}
      />
      {blockedReason ? (
        <span className={errorClassName} role="alert">
          {blockedReason}
        </span>
      ) : null}
    </>
  );
}
