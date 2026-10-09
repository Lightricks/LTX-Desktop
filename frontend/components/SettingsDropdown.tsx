import React, { useEffect } from "react";
import { createPortal } from "react-dom";
import { AlertTriangle } from "lucide-react";

import { useThemedPortalContainer } from "@/ds/styles/themes/useTheme";

import { useFixedMenu } from "../hooks/use-fixed-menu";

import { Tooltip } from "./ui/tooltip";

// Generic settings dropdown used across the prompt bar's control rows.
// Portaled into the themed `.ltx-io` root so overflow-hidden ancestors
// cannot clip it and semantic tokens still apply in light and dark.
export function SettingsDropdown({
  trigger,
  options,
  value,
  onChange,
  title,
  tooltip,
  triggerClassName,
  placement = "above",
  variant = "default",
  onOpenChange,
  showSelectedIndicator = true,
}: {
  trigger: React.ReactNode;
  options: {
    value: string;
    label: string;
    disabled?: boolean;
    tooltip?: string;
    warning?: string;
    icon?: React.ReactNode;
  }[];
  value: string;
  onChange: (value: string) => void;
  title: string;
  tooltip?: string;
  // Extra classes on the trigger button — e.g. to visually attach it to an adjacent button
  // as a split-button (rounded-l-none, no left padding, etc).
  triggerClassName?: string;
  // Prompt-bar menus open upward; gallery toolbar menus open downward.
  placement?: "above" | "below";
  /** Attached segment of EnhanceSplitButton — no outer chrome on the trigger. */
  variant?: "default" | "split";
  onOpenChange?: (open: boolean) => void;
  showSelectedIndicator?: boolean;
}) {
  const { isOpen, setIsOpen, triggerRef, menuRef, style } = useFixedMenu(placement);
  const portalRoot = useThemedPortalContainer();

  useEffect(() => {
    onOpenChange?.(isOpen);
  }, [isOpen, onOpenChange]);

  const setOpen = (open: boolean) => {
    setIsOpen(open);
  };

  const triggerButton = (
    <button
      type="button"
      aria-haspopup="listbox"
      aria-expanded={isOpen}
      onClick={() => setOpen(!isOpen)}
      className={
        variant === "split"
          ? `flex h-full w-[30px] shrink-0 items-center justify-center border-0 bg-transparent transition-colors duration-150 ${
              isOpen
                ? "bg-fg-primary/10"
                : "hover:bg-fg-primary/6"
            } ${triggerClassName ?? ""}`
          : `flex shrink-0 items-center gap-1 whitespace-nowrap px-2 py-1.5 rounded-md transition-colors ${isOpen ? "bg-action-hover hover:bg-action-hover" : "hover:bg-action"} ${triggerClassName ?? ""}`
      }
    >
      {trigger}
    </button>
  );

  return (
    <div ref={triggerRef} className="relative">
      {tooltip && !isOpen ? (
        <Tooltip content={tooltip}>{triggerButton}</Tooltip>
      ) : (
        triggerButton
      )}

      {isOpen &&
        createPortal(
          <div
            ref={menuRef}
            style={style}
            className="fixed w-max bg-action border border-separator rounded-md p-2 min-w-[160px] shadow-xl"
          >
            <div className="text-[10px] text-fg-tertiary uppercase tracking-wider mb-2">
              {title}
            </div>
            {/* Cap height + scroll so a long option list (e.g. many catalog / custom IC-LoRAs)
              doesn't clip off-screen — matches the LoRA picker's max-h-80. */}
            <div className="space-y-1 max-h-80 overflow-y-auto">
              {options.map((option) => (
                <div key={option.value} className="relative group/option">
                  <button
                    onClick={() => {
                      if (!option.disabled) {
                        onChange(option.value);
                        setOpen(false);
                      }
                    }}
                    className={`w-full flex items-center justify-between px-2 py-2 rounded-md transition-colors text-left ${
                      option.disabled
                        ? "cursor-not-allowed"
                        : value === option.value
                          ? "bg-surface-select hover:bg-surface-select"
                          : "hover:bg-action-hover"
                    }`}
                  >
                    <span
                      className={`flex items-center gap-2.5 text-sm ${
                        option.disabled
                          ? "text-fg-tertiary"
                          : value === option.value
                            ? "text-fg-primary"
                            : "text-fg-secondary"
                      }`}
                    >
                      {option.icon && (
                        <span className="flex-shrink-0">{option.icon}</span>
                      )}
                      {option.label}
                    </span>
                    <span className="flex items-center gap-2 flex-shrink-0">
                      {option.warning && (
                        <Tooltip content={option.warning}>
                          <span
                            className="inline-flex text-fg-tertiary/80"
                            onClick={(event) => event.stopPropagation()}
                            aria-label={option.warning}
                          >
                            <AlertTriangle className="h-3 w-3" />
                          </span>
                        </Tooltip>
                      )}
                      {showSelectedIndicator &&
                        value === option.value &&
                        !option.disabled && (
                          <svg
                            className="w-5 h-5 text-fg-primary"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M5 13l4 4L19 7"
                            />
                          </svg>
                        )}
                    </span>
                  </button>
                  {option.disabled && option.tooltip && (
                    <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 px-2 py-1 bg-action-hover rounded text-xs text-fg-secondary whitespace-nowrap opacity-0 group-hover/option:opacity-100 pointer-events-none z-[10000] transition-opacity">
                      {option.tooltip}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>,
          portalRoot ?? document.body,
        )}
    </div>
  );
}
