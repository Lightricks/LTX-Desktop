import { ActivityCircular } from "@ds/ActivityCircular/ActivityCircular";
import { ItemAction } from "@ds/DropdownItems/ItemAction/ItemAction";
import { DropdownPicker } from "@ds/DropdownPicker/DropdownPicker";
import { PickerBox } from "@ds/PickerBox/PickerBox";
import { useThemeRootElement } from "@ds/styles/themes/useTheme";
import { clsx } from "clsx";
import { ChevronUp, Loader2 } from "lucide-react";
import { type ReactNode, useState } from "react";

import type { EnhanceProvider } from "@/hooks/use-prompt-enhancer-provider";
import {
  ENHANCE_PROVIDER_MENU_OPTIONS,
  ENHANCE_VIA_MENU_TITLE,
} from "@/lib/enhance-ui-copy";

import styles from "./EnhanceSplitButton.module.scss";
import { SettingsDropdown } from "./SettingsDropdown";

const FORM_MENU_MIN_WIDTH = "168px";

const FORM_PICKER_CONTENT_PROPS = {
  side: "bottom" as const,
  align: "end" as const,
  sideOffset: 6,
  avoidCollisions: true,
  collisionPadding: 8,
  style: {
    minWidth: FORM_MENU_MIN_WIDTH,
    width: FORM_MENU_MIN_WIDTH,
    maxHeight: "240px",
    zIndex: "calc(var(--z-modal) + 1)",
  },
};

type EnhanceSplitButtonProps = {
  label: string;
  onEnhance: () => void;
  disabled?: boolean;
  isLoading?: boolean;
  hasError?: boolean;
  title?: string;
  leftIcon?: ReactNode;
  /** Feature forms use design-system picker; Gen Space uses prompt-bar SettingsDropdown. */
  surface?: "form" | "promptBar";
  provider?: EnhanceProvider;
  onProviderChange?: (provider: EnhanceProvider) => void;
  showProviderMenu?: boolean;
};

export function EnhanceSplitButton({
  label,
  onEnhance,
  disabled = false,
  isLoading = false,
  hasError = false,
  title,
  leftIcon,
  surface = "form",
  provider = "api",
  onProviderChange,
  showProviderMenu = false,
}: EnhanceSplitButtonProps) {
  const menuEnabled = showProviderMenu && onProviderChange != null;
  const rootElement = useThemeRootElement();
  const [isPickerOpen, setIsPickerOpen] = useState(false);
  const [isPromptMenuOpen, setIsPromptMenuOpen] = useState(false);

  if (surface === "promptBar") {
    return (
      <div className={clsx(styles.promptBarGroup, isPromptMenuOpen && styles.menuOpen)}>
        <button
          type="button"
          className={clsx(styles.promptBarMain, hasError && styles.error)}
          onClick={onEnhance}
          disabled={disabled || isLoading}
          title={title ?? label}
        >
          {isLoading ? (
            <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" aria-hidden />
          ) : (
            leftIcon
          )}
          <span>{label}</span>
        </button>
        {menuEnabled ? (
          <>
            <span className={styles.promptBarDivider} aria-hidden />
            <div className={styles.promptBarMenu}>
              <SettingsDropdown
                title={ENHANCE_VIA_MENU_TITLE}
                value={provider}
                onChange={(value) => onProviderChange(value === "api" ? "api" : "local")}
                options={[...ENHANCE_PROVIDER_MENU_OPTIONS]}
                showSelectedIndicator={false}
                trigger={
                  <ChevronUp
                    className={clsx(
                      "h-3 w-3 text-zinc-500",
                      styles.promptBarChevron,
                      isPromptMenuOpen && styles.open,
                    )}
                    aria-hidden
                  />
                }
                triggerClassName="rounded-none !px-0 !py-0 h-7 w-7 flex items-center justify-center"
                onOpenChange={setIsPromptMenuOpen}
              />
            </div>
          </>
        ) : null}
      </div>
    );
  }

  return (
    <div
      className={clsx(styles.formGroup, (disabled || isLoading) && styles.disabled)}
    >
      <button
        type="button"
        className={clsx(styles.formMain, hasError && styles.error)}
        onClick={onEnhance}
        disabled={disabled || isLoading}
        title={title ?? label}
      >
        {isLoading ? <ActivityCircular appearance="default" size={10} /> : leftIcon}
        <span>{label}</span>
      </button>
      {menuEnabled ? (
        <div className={styles.formMenuWrap}>
          <DropdownPicker
            label={ENHANCE_VIA_MENU_TITLE}
            value={provider}
            minWidth={FORM_MENU_MIN_WIDTH}
            isOpen={isPickerOpen}
            onOpenChange={setIsPickerOpen}
            onValueChange={(value) => onProviderChange(value === "api" ? "api" : "local")}
            portalContainer={rootElement}
            contentProps={FORM_PICKER_CONTENT_PROPS}
            options={ENHANCE_PROVIDER_MENU_OPTIONS.map((option) => ({
              value: option.value,
              label: option.label,
              render: (
                <ItemAction
                  text={option.label}
                  textVariant="body"
                  isSelectable
                  isActive={provider === option.value}
                />
              ),
            }))}
          >
            <PickerBox
              className={styles.formMenuTrigger}
              hidePlaceholder
              value=""
              state={isPickerOpen ? "active" : "enabled"}
              shouldRotateArrowOnActive
              spacing="tight"
              textVariant="body"
              textSize="md"
            />
          </DropdownPicker>
        </div>
      ) : null}
    </div>
  );
}
