import { useCallback, useEffect, useState } from "react";

import { useTheme } from "@ds/styles/themes/useTheme";

export type ThemeControlClass =
  | "dark-only-theme-style"
  | "light-only-theme-style"
  | "inverted-theme-style";

interface UseThemeVariablesOptions {
  contextNode?: HTMLElement | null;
}

export const useThemeVariables = (options?: UseThemeVariablesOptions) => {
  const { colorScheme } = useTheme();

  // This is used to force a re-render when the theme changes
  const [themeVersion, setThemeVersion] = useState(0);

  useEffect(() => {
    setThemeVersion((v) => v + 1);
  }, [colorScheme]);

  /**
   * Retrieves the value of a CSS variable.
   *
   * @param name - The name of the CSS variable, with or without the '--' prefix.
   * @param overrideNode - Optional. Overrides the context node specified during hook initialization.
   * @returns The value of the CSS variable.
   */
  const getCSSVariableValue = useCallback(
    (name: string, overrideNode?: HTMLElement) => {
      const normalizedName = name.startsWith("--") ? name : `--${name}`;

      const targetElement = overrideNode || options?.contextNode || document.body;
      return getComputedStyle(targetElement).getPropertyValue(normalizedName).trim();
    },
    [themeVersion, options?.contextNode],
  );

  /**
   * Retrieves the value of a CSS variable for a specific theme control class.
   * This creates a temporary element with the appropriate theme class to get the themed value.
   *
   * @param name - The name of the CSS variable, with or without the '--' prefix.
   * @param themeClass - The theme control class to apply
   * @returns The value of the CSS variable in the specified theme.
   */
  const getThemedCSSVariableValue = useCallback(
    (name: string, themeClass: ThemeControlClass) => {
      const normalizedName = name.startsWith("--") ? name : `--${name}`;

      // Create a temporary element with the specified theme class
      const tempElement = document.createElement("div");
      tempElement.className = themeClass;
      tempElement.style.position = "absolute";
      tempElement.style.visibility = "hidden";
      tempElement.style.pointerEvents = "none";

      // Add to the same context as the target element
      const targetElement = options?.contextNode || document.body;
      targetElement.appendChild(tempElement);

      // Get the computed style value
      const value = getComputedStyle(tempElement).getPropertyValue(normalizedName).trim();

      // Clean up the temporary element
      targetElement.removeChild(tempElement);

      return value;
    },
    [themeVersion, options?.contextNode],
  );

  return {
    getCSSVariableValue,
    getThemedCSSVariableValue,
  };
};
