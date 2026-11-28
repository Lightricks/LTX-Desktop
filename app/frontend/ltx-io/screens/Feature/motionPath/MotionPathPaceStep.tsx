import {
  MOTION_PATH_EASING_OPTIONS,
  MOTION_PATH_HOLD_OPTIONS_SEC,
  type MotionPathEasing,
  type MotionPathPoint,
} from "../../../lib/motionPathSampler";
import { type ReactNode, useMemo, useState } from "react";

import { Button } from "@ds/Button/Button";
import { ItemAction } from "@ds/DropdownItems/ItemAction/ItemAction";
import { DropdownMenu } from "@ds/DropdownMenu/DropdownMenu";
import { PickerBox } from "@ds/PickerBox/PickerBox";
import { Text } from "@ds/Text/Text";
import { Flex } from "@ds/layout/Flex/Flex";
import { useThemeRootElement } from "@ds/styles/themes/useTheme";

import { Checkbox } from "@/ltx-io/components/Checkbox/Checkbox";

import { MotionPathEasingCurveIcon } from "./MotionPathEasingCurveIcon";
import { MotionPathPacePreview } from "./MotionPathPacePreview";
import styles from "./MotionPathPaceStep.module.scss";
import {
  MOTION_PATH_MODAL_PREVIEW_MAX_HEIGHT,
  getMotionPathStageMinStyle,
} from "./motionPathPreviewFrameStyle";
import type { TrackPaces } from "./useTrackPaces";

export type MotionPathPaceStepProps = {
  imageUrl?: string;
  /** The frame the pipeline renders for the image, or null until the image loads. */
  imageAspect: number | null;
  /** Every drawn track. The id keys its pace. */
  tracks: { id: number; points: MotionPathPoint[] }[];
  pacing: TrackPaces;
  /** The clip lengths the form offers. */
  durationOptions: readonly number[];
  onAccept: () => void;
  onCancel: () => void;
};

const EASING_LABELS: Record<MotionPathEasing, string> = {
  linear: "Linear",
  easeIn: "Ease in",
  easeOut: "Ease out",
  easeInOut: "Ease in and out",
  easeInBack: "Ease in back",
  easeOutBack: "Ease out back",
  easeInOutBack: "Ease in and out back",
  gentle: "Gentle",
  quick: "Quick",
  bouncy: "Bouncy",
  slow: "Slow",
};

function formatHoldSec(sec: number): string {
  return sec === 0 ? "None" : `${sec}s`;
}

function TimingPicker<T extends string | number>({
  label,
  value,
  options,
  formatLabel,
  onChange,
  renderLeftIcon,
}: {
  label: string;
  value: T;
  options: readonly T[];
  formatLabel: (option: T) => string;
  onChange: (value: T) => void;
  renderLeftIcon?: (option: T) => ReactNode;
}) {
  // The menu is portaled. Into the themed root, it keeps the theme tokens (background, text).
  const rootElement = useThemeRootElement();
  const items = useMemo(
    () =>
      options.map((option) => ({
        render: (
          <ItemAction
            text={formatLabel(option)}
            leftIcon={renderLeftIcon?.(option)}
            selectionPosition="right"
            isSelectable
            isSelected={option === value}
            onClick={() => onChange(option)}
          />
        ),
      })),
    [formatLabel, onChange, options, renderLeftIcon, value],
  );

  return (
    <div className={styles.field}>
      <Text as="span" variant="body" size="md" className={styles.fieldLabel}>
        {label}
      </Text>
      <DropdownMenu
        className={styles.menu}
        items={items}
        align="start"
        sideOffset={8}
        maxHeight="320px"
        minWidth="var(--radix-dropdown-menu-trigger-width)"
        portalContainer={rootElement}
      >
        <PickerBox
          className={styles.picker}
          value={formatLabel(value)}
          icon={renderLeftIcon?.(value)}
          shouldRotateArrowOnActive
        />
      </DropdownMenu>
    </div>
  );
}

/**
 * Step 2 — live paced preview + timing controls.
 * Footer: Cancel + Done (primary).
 */
export function MotionPathPaceStep({
  imageUrl,
  imageAspect,
  tracks,
  pacing,
  durationOptions,
  onAccept,
  onCancel,
}: MotionPathPaceStepProps) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const selected = tracks[Math.min(selectedIndex, tracks.length - 1)];
  const pace = pacing.paceOf(selected.id);
  const { paceOf } = pacing;
  const previewTracks = useMemo(
    () => tracks.map(({ id, points }) => ({ points, ...paceOf(id) })),
    [tracks, paceOf],
  );
  const hasManyTracks = tracks.length > 1;

  // The modal keeps the size of the draw step, so it does not jump between the steps.
  const layoutStyle =
    imageAspect === null
      ? undefined
      : getMotionPathStageMinStyle(imageAspect, MOTION_PATH_MODAL_PREVIEW_MAX_HEIGHT);

  return (
    <Flex direction="column" gap="lg" className={styles.root}>
      <div className={styles.layout} style={layoutStyle}>
        <MotionPathPacePreview
          imageUrl={imageUrl}
          imageAspect={imageAspect}
          tracks={previewTracks}
          durationSec={pacing.durationSec}
        />

        <div className={styles.controls}>
          <TimingPicker
            label="Duration"
            value={pacing.durationSec}
            options={durationOptions}
            formatLabel={(sec) => `${sec}s`}
            onChange={pacing.setDurationSec}
          />
          {hasManyTracks ? (
            <Checkbox
              checked={pacing.samePace}
              onChange={(event) =>
                pacing.setSamePace(
                  event.target.checked,
                  selected.id,
                  tracks.map((track) => track.id),
                )
              }
              data-testid="motion-path-same-pace"
            >
              <Text as="span" variant="body" size="md">
                Same pace for all tracks
              </Text>
            </Checkbox>
          ) : null}
          {hasManyTracks && !pacing.samePace ? (
            <Flex gap="sm" wrap="wrap" data-testid="motion-path-pace-tracks">
              {tracks.map((track, index) => (
                <Button
                  key={track.id}
                  appearance={track === selected ? "brand" : "neutral"}
                  hierarchy="secondary"
                  size="sm"
                  label={`Track ${index + 1}`}
                  onClick={() => setSelectedIndex(index)}
                />
              ))}
            </Flex>
          ) : null}
          <TimingPicker
            label="Easing"
            value={pace.easing}
            options={MOTION_PATH_EASING_OPTIONS}
            formatLabel={(option) => EASING_LABELS[option]}
            renderLeftIcon={(option) => <MotionPathEasingCurveIcon easing={option} />}
            onChange={(easing) => pacing.updatePace(selected.id, { easing })}
          />
          <TimingPicker
            label="Pause at the beginning"
            value={pace.startHoldSec}
            options={MOTION_PATH_HOLD_OPTIONS_SEC}
            formatLabel={formatHoldSec}
            onChange={(startHoldSec) => pacing.updatePace(selected.id, { startHoldSec })}
          />
          <TimingPicker
            label="Pause at the end"
            value={pace.endHoldSec}
            options={MOTION_PATH_HOLD_OPTIONS_SEC}
            formatLabel={formatHoldSec}
            onChange={(endHoldSec) => pacing.updatePace(selected.id, { endHoldSec })}
          />
        </div>
      </div>

      <Flex justify="end" gap="md">
        <Button
          appearance="neutral"
          hierarchy="plain"
          size="md"
          label="Cancel"
          onClick={onCancel}
        />
        <Button
          appearance="brand"
          hierarchy="primary"
          size="md"
          label="Done"
          onClick={onAccept}
        />
      </Flex>
    </Flex>
  );
}
