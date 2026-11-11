import { clsx } from "clsx";
import { type ReactNode, useState } from "react";

import { ItemAction } from "@ds/DropdownItems/ItemAction/ItemAction";
import { DropdownPicker } from "@ds/DropdownPicker/DropdownPicker";
import { PickerBox } from "@ds/PickerBox/PickerBox";
import { useTheme } from "@ds/styles/themes/useTheme";
import { Tooltip } from "@ds/Tooltip/Tooltip";

import { BudgetFraction } from "@/components/BudgetFraction";
import { budgetTooltip } from "@/lib/budget-figure";
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

const PARTS_TOOLTIP = "videos are generated in smaller overlapping pieces and stitched into one video";

function MemoryLabel({ option }: { option: FieldOption }) {
  if (option.memory == null) return null;
  const label = option.memory.label;
  const partsMatch = label.match(/^(.* · )(\d+ parts)$/);
  return (
    <span className={clsx(styles.memoryLabel, option.memory.over && styles.memoryLabelOver)}>
      {partsMatch ? (
        <>
          {partsMatch[1]}
          <Tooltip content={PARTS_TOOLTIP}>
            <span className={styles.partsTooltipTrigger}>{partsMatch[2]}</span>
          </Tooltip>
        </>
      ) : (
        label
      )}
    </span>
  );
}

function caveatIcon(option: FieldOption) {
  return option.warning ? (
    <AspectRatioWarningIcon warning={option.warning} tone={option.warningTone ?? "caveat"} />
  ) : null;
}

function buildOptionListItem<TValue extends string | number>(
  option: FieldOption<TValue>,
  value: TValue,
  onSelect: (option: FieldOption<TValue>) => void,
  getOptionIcon?: (option: FieldOption<TValue>) => ReactNode,
) {
  const icon = getOptionIcon?.(option);
  const budget = option.budget ?? null;
  const item = (
    <ItemAction
      text={option.label}
      leftIcon={
        icon ? <span className={styles.optionListIcon}>{icon}</span> : undefined
      }
      rightCta={
        budget != null ? (
          <BudgetFraction readout={budget} />
        ) : option.memory != null ? (
          <span className={styles.memoryCta}>
            <MemoryLabel option={option} />
            {caveatIcon(option)}
          </span>
        ) : (
          caveatIcon(option) ?? undefined
        )
      }
      thumbnail={option.thumbnail ? { src: option.thumbnail, alt: option.label } : undefined}
      caption={budget == null ? (option.disabledReason ?? option.caption) : undefined}
      isDisabled={option.disabledReason != null}
      textVariant="body"
      selectionPosition="right"
      isSelectable
      isSelected={isOptionSelected(value, option)}
      onClick={() => onSelect(option)}
    />
  );
  if (option.disabledReason === undefined) {
    if (budget == null) {
      return item;
    }
    return (
      <Tooltip content={budgetTooltip(budget, option.budgetNote)}>
        <span className={styles.disabledOptionTrigger}>
          {item}
        </span>
      </Tooltip>
    );
  }
  // Plain-span trigger, mirroring AspectRatioWarningIcon: the refusal must
  // fire even though the row itself is disabled.
  return (
    <Tooltip content={budget ? budgetTooltip(budget) : option.disabledReason}>
      <span className={styles.disabledOptionTrigger}>
        {item}
      </span>
    </Tooltip>
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
    // Backstop: the item/segment guards block pointer clicks, but keyboard
    // selection reaches here directly. Withheld options never commit.
    if (option.disabledReason != null) return;
    onChange(option.value);
  };

  const selectedOption =
    options.find((option) => isOptionSelected(value, option)) ?? options[0];
  const selectedLabel = selectedOption?.label ?? String(value ?? "");
  const selectedKey = selectedOption
    ? toOptionSegmentKey(selectedOption.value)
    : toOptionSegmentKey(value);
  const riskMarker =
    selectedOption?.warningTone === "risk" && selectedOption.warning ? (
      <AspectRatioWarningIcon
        warning={
          selectedOption.budget
            ? budgetTooltip(selectedOption.budget, selectedOption.budgetNote)
            : selectedOption.warning
        }
        tone="risk"
      />
    ) : null;
  const selectedIcon =
    showTriggerIcon && selectedOption
      ? getOptionIcon?.(selectedOption)
      : undefined;

  const pickerLeading = selectedOption?.thumbnail
    ? { thumbnail: { src: selectedOption.thumbnail, alt: selectedOption.label } }
    : { icon: selectedIcon };

  const presentation = getOptionControlPresentation(options, {
    forcePicker,
    maxOptionCount,
  });

  if (presentation === "display-only") {
    const box = (
      <PickerBox
        className={pickerClassName}
        value={selectedLabel}
        state="readOnly"
        hideArrow
        aria-label={ariaLabel}
        icon={selectedIcon}
        trailing={riskMarker}
        {...PICKER_TEXT}
      />
    );
    if (selectedOption?.disabledReason === undefined) return box;
    return (
      <Tooltip
        content={
          selectedOption.budget
            ? budgetTooltip(selectedOption.budget)
            : selectedOption.disabledReason
        }
      >
        {box}
      </Tooltip>
    );
  }

  if (presentation === "segment") {
    const segments = options.map((option) => {
      const icon = getOptionIcon?.(option);
      const isSelected = toOptionSegmentKey(option.value) === selectedKey;
      return {
        value: toOptionSegmentKey(option.value),
        disabled: option.disabledReason != null,
        disabledTooltip: option.budget
          ? budgetTooltip(option.budget)
          : option.disabledReason,
        label: (
          <span className={styles.formOptionSegmentLabel}>
            {icon}
            {option.label}
            <MemoryLabel option={option} />
            {option.budget ? (
              <>
                <BudgetFraction readout={option.budget} />
                {option.budgetNote && option.disabledReason == null ? (
                  <AspectRatioWarningIcon warning={option.budgetNote} />
                ) : null}
              </>
            ) : option.warning &&
              (!isSelected || option.warningTone === "risk") ? (
              <AspectRatioWarningIcon
                warning={option.warning}
                tone={option.warningTone ?? "caveat"}
              />
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
          {...pickerLeading}
          trailing={riskMarker}
          state={isPickerOpen ? "active" : "enabled"}
          shouldRotateArrowOnActive
          {...PICKER_TEXT}
        />
      </DropdownPicker>
    </div>
  );
}
