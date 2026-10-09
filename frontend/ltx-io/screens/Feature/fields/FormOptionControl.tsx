import { clsx } from "clsx";
import { type ReactNode, useState } from "react";

import { ItemAction } from "@ds/DropdownItems/ItemAction/ItemAction";
import { DropdownPicker } from "@ds/DropdownPicker/DropdownPicker";
import { PickerBox } from "@ds/PickerBox/PickerBox";
import { useTheme } from "@ds/styles/themes/useTheme";

import { SlidingSegmentControl } from "../../../components/SlidingSegmentControl/SlidingSegmentControl";
import type { FieldOption } from "../types";
import { AspectRatioWarningIcon } from "./AspectRatioWarningIcon";
import {
  getOptionControlPresentation,
  isOptionSelected,
  toOptionSegmentKey,
} from "./optionControlPresentation";
import styles from "./FormOptionControl.module.scss";

const PICKER_TEXT = {
  textVariant: "body" as const,
  textSize: "lg" as const,
  spacing: "compact" as const,
  textClassName: styles.pickerValue,
} as const;

const PICKER_CONTENT_PROPS = {
  side: "bottom" as const,
  align: "start" as const,
  sideOffset: 8,
  avoidCollisions: true,
  collisionPadding: 8,
  style: {
    width: "var(--radix-select-trigger-width)",
    maxHeight: "240px",
    minWidth: "var(--radix-select-trigger-width)",
    maxWidth: "var(--radix-select-trigger-width)",
    zIndex: "calc(var(--z-modal) + 1)",
  },
};

function buildOptionListItem<TValue extends string | number>(
  option: FieldOption<TValue>,
  value: TValue,
  onSelect: (option: FieldOption<TValue>) => void,
  getOptionIcon?: (option: FieldOption<TValue>) => ReactNode,
) {
  const icon = getOptionIcon?.(option);
  return (
    <ItemAction
      text={option.label}
      leftIcon={
        icon ? <span className={styles.optionListIcon}>{icon}</span> : undefined
      }
      rightCta={
        option.warning ? (
          <AspectRatioWarningIcon warning={option.warning} />
        ) : undefined
      }
      caption={option.disabledReason}
      isDisabled={option.disabledReason != null}
      textVariant="body"
      selectionPosition="right"
      isSelectable
      isSelected={isOptionSelected(value, option)}
      onClick={() => onSelect(option)}
    />
  );
}

export function FormOptionControl<TValue extends string | number>({
  options,
  value,
  onChange,
  ariaLabel,
  forcePicker,
  maxOptionCount,
  getOptionIcon,
  showTriggerIcon = true,
  showListIcons = false,
}: {
  options: readonly FieldOption<TValue>[];
  value: TValue;
  onChange: (value: TValue) => void;
  ariaLabel: string;
  forcePicker?: boolean;
  maxOptionCount?: number;
  getOptionIcon?: (option: FieldOption<TValue>) => ReactNode;
  showTriggerIcon?: boolean;
  showListIcons?: boolean;
}) {
  const { rootElement } = useTheme();
  const [isPickerOpen, setIsPickerOpen] = useState(false);
  const pickerClassName = clsx(styles.picker, styles.settingsPicker);

  const handleSelect = (option: FieldOption<TValue>) => {
    if (option.disabledReason != null) return;
    onChange(option.value);
  };

  const selectedOption =
    options.find((option) => isOptionSelected(value, option)) ?? options[0];
  const selectedLabel = selectedOption?.label ?? String(value ?? "");
  const selectedKey = selectedOption
    ? toOptionSegmentKey(selectedOption.value)
    : toOptionSegmentKey(value);
  const selectedIcon =
    showTriggerIcon && selectedOption
      ? getOptionIcon?.(selectedOption)
      : undefined;

  const presentation = getOptionControlPresentation(options, {
    forcePicker,
    maxOptionCount,
  });

  if (presentation === "display-only") {
    return (
      <PickerBox
        className={pickerClassName}
        value={selectedLabel}
        state="readOnly"
        hideArrow
        aria-label={ariaLabel}
        icon={selectedIcon}
        {...PICKER_TEXT}
      />
    );
  }

  if (presentation === "segment") {
    const segments = options.map((option) => {
      const icon = getOptionIcon?.(option);
      const isSelected = toOptionSegmentKey(option.value) === selectedKey;
      return {
        value: toOptionSegmentKey(option.value),
        disabled: option.disabledReason != null,
        disabledTooltip: option.disabledReason,
        label: (
          <span className={styles.formOptionSegmentLabel}>
            {icon}
            {option.label}
            {!isSelected && option.warning ? (
              <AspectRatioWarningIcon warning={option.warning} />
            ) : null}
          </span>
        ),
      };
    });

    return (
      <SlidingSegmentControl
        size="sm"
        fullWidth
        aria-label={ariaLabel}
        segments={segments}
        selectedSegment={selectedKey}
        onSegmentChanged={(nextKey) => {
          const option = options.find(
            (candidate) => toOptionSegmentKey(candidate.value) === nextKey,
          );
          if (option) handleSelect(option);
        }}
        className={styles.formOptionSegment}
      />
    );
  }

  const listIconGetter = showListIcons ? getOptionIcon : undefined;

  return (
    <div className={styles.pickerTrigger}>
      <DropdownPicker
        options={options.map((option) => ({
          value: toOptionSegmentKey(option.value),
          label: option.label,
          isDisabled: option.disabledReason != null,
          render: buildOptionListItem(
            option,
            value,
            handleSelect,
            listIconGetter,
          ),
        }))}
        value={selectedKey}
        label={ariaLabel}
        isOpen={isPickerOpen}
        onOpenChange={setIsPickerOpen}
        onValueChange={(nextKey) => {
          const option = options.find(
            (candidate) => toOptionSegmentKey(candidate.value) === nextKey,
          );
          if (option) handleSelect(option);
        }}
        portalContainer={rootElement}
        contentProps={PICKER_CONTENT_PROPS}
      >
        <PickerBox
          className={pickerClassName}
          value={selectedLabel}
          icon={selectedIcon}
          state={isPickerOpen ? "active" : "enabled"}
          shouldRotateArrowOnActive
          {...PICKER_TEXT}
        />
      </DropdownPicker>
    </div>
  );
}
