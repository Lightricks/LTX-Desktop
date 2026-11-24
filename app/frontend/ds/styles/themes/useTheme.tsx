import { clsx } from "clsx";
import {
  type MutableRefObject,
  type PropsWithChildren,
  type Ref,
  createContext,
  forwardRef,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useState,
} from "react";
import { flushSync } from "react-dom";

import { storage } from "@ds/lib/storage";

export type ColorScheme = "light" | "dark";
export const COLOR_SCHEMES: readonly ColorScheme[] = ["light", "dark"] as const;
export type ThemePreference = "system" | "light" | "dark";
export const THEME_PREFERENCES: readonly ThemePreference[] = [
  "system",
  "light",
  "dark",
] as const;

interface ThemeContextProps {
  colorScheme: ColorScheme;
  themePreference: ThemePreference;
  setThemePreference: (preference: ThemePreference) => void;
  /** The `.ltx-io` root node — use as a portal container so menus keep theme tokens. */
  rootElement: HTMLDivElement | null;
}

function assignRef<T>(ref: Ref<T> | undefined, value: T | null) {
  if (!ref) return;
  if (typeof ref === "function") {
    ref(value);
    return;
  }
  (ref as MutableRefObject<T | null>).current = value;
}

const ThemeContext = createContext<ThemeContextProps | undefined>(undefined);

const THEME_STORAGE_KEY = "ltxio.theme";

function notifyWindowAppearance(preference: ThemePreference, persist: boolean) {
  // Desktop only — remote/phone webviews have no electronAPI.
  window.electronAPI?.setWindowAppearance?.({ theme: preference, persist })?.catch(() => {});
}

function readStoredPreference(): ThemePreference | null {
  try {
    const stored = storage.getItem(THEME_STORAGE_KEY);
    return stored !== null &&
      (THEME_PREFERENCES as readonly string[]).includes(stored)
      ? (stored as ThemePreference)
      : null;
  } catch {
    // localStorage can throw in private mode; stay on the fallback.
    return null;
  }
}

const ThemeProvider = forwardRef<
  HTMLDivElement,
  PropsWithChildren<{
    className?: string;
    /** Preference used when the user hasn't stored one. */
    defaultPreference?: ThemePreference;
    /** Live override. When set, wins over stored preference and defaultPreference. */
    scheme?: ColorScheme;
  }>
>(function ThemeProvider(
  { children, className, scheme, defaultPreference = "dark" },
  forwardedRef,
) {

  const getSystemColorScheme = (): ColorScheme =>
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";

  const getColorSchemeFromPreference = (
    preference: ThemePreference,
  ): ColorScheme => {
    switch (preference) {
      case "system":
        return getSystemColorScheme();
      case "light":
        return "light";
      case "dark":
        return "dark";
    }
  };

  const initialPreference: ThemePreference =
    scheme ?? readStoredPreference() ?? defaultPreference;

  const [themePreference, setThemePreferenceState] =
    useState<ThemePreference>(initialPreference);
  const [colorScheme, setColorScheme] = useState<ColorScheme>(
    getColorSchemeFromPreference(initialPreference),
  );
  const [rootElement, setRootElement] = useState<HTMLDivElement | null>(null);

  const setRootRef = useCallback(
    (node: HTMLDivElement | null) => {
      setRootElement(node);
      assignRef(forwardedRef, node);
    },
    [forwardedRef],
  );

  function setThemePreference(preference: ThemePreference) {
    const newScheme = getColorSchemeFromPreference(preference);

    const update = () => {
      flushSync(() => {
        setThemePreferenceState(preference);
        setColorScheme(newScheme);
      });
      storage.setItem(THEME_STORAGE_KEY, preference);
    };

    if (newScheme !== colorScheme && document.startViewTransition) {
      document.startViewTransition(update);
    } else {
      update();
    }
    // Only the preference-owning provider (no scheme override) drives main.
    if (scheme == null) {
      notifyWindowAppearance(preference, true);
    }
  }

  // Mount-only sync: only the preference-owning provider (no scheme override)
  // notifies main, so scheme-override providers (e.g. light setup screens)
  // can't overwrite the user's glass. Persisting heals upgrades that stored
  // ltxio.theme before main learned to persist windowTheme — one write/launch.
  useEffect(() => {
    if (scheme == null) {
      notifyWindowAppearance(initialPreference, true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!scheme) return;
    setThemePreferenceState(scheme);
    setColorScheme(scheme);
  }, [scheme]);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

    const updateColorScheme = (e: MediaQueryListEvent | MediaQueryList) => {
      if (themePreference === "system") {
        setColorScheme(e.matches ? "dark" : "light");
      }
    };

    updateColorScheme(mediaQuery);
    mediaQuery.addEventListener("change", updateColorScheme);

    return () => mediaQuery.removeEventListener("change", updateColorScheme);
  }, [themePreference]);

  // The provider owns the `.ltx-io` root: `data-theme` is applied here (not on
  // `document.body`) so the semantic light/dark variables stay contained to the
  // `.ltx-io` subtree in Desktop's single-document app. Rendering `data-theme` in
  // JSX applies it before first paint, avoiding a flash of the wrong theme.
  return (
    <ThemeContext.Provider
      value={{ colorScheme, themePreference, setThemePreference, rootElement }}
    >
      <div
        ref={setRootRef}
        className={clsx("ltx-io", className)}
        data-theme={colorScheme}
      >
        {children}
      </div>
    </ThemeContext.Provider>
  );
});

ThemeProvider.displayName = "ThemeProvider";

const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
};

/**
 * Non-throwing access to the `.ltx-io` root, for use as a portal container so
 * portaled content (tooltips, menus) stays inside the themed subtree and keeps
 * its semantic tokens. Returns null outside a ThemeProvider.
 */
const useThemeRootElement = (): HTMLDivElement | null =>
  useContext(ThemeContext)?.rootElement ?? null;

/** Non-throwing color scheme for portaled UI (e.g. tooltips on `document.body`). */
const useThemeColorScheme = (): ColorScheme =>
  useContext(ThemeContext)?.colorScheme ?? "dark";

const LTX_IO_ROOT_SELECTOR = ".ltx-io";

/** Nearest `.ltx-io` root so portaled UI keeps `data-theme` tokens. */
export function closestLtxIoRoot(
  from?: ParentNode | EventTarget | null,
): HTMLElement | null {
  if (from instanceof Element) {
    const closest = from.closest(LTX_IO_ROOT_SELECTOR);
    if (closest instanceof HTMLElement) return closest;
  }
  if (typeof document === "undefined") return null;
  const fallback = document.querySelector(LTX_IO_ROOT_SELECTOR);
  return fallback instanceof HTMLElement ? fallback : null;
}

/**
 * Portal container that keeps inverted tooltip/menu tokens. Prefers ThemeProvider,
 * then the nearest `.ltx-io` (first-run setup and other screens that theme that node themselves).
 */
export function useThemedPortalContainer(): HTMLElement | null {
  const contextRoot = useThemeRootElement();
  const [fallback, setFallback] = useState<HTMLElement | null>(null);

  useLayoutEffect(() => {
    if (contextRoot) {
      setFallback(null);
      return;
    }
    setFallback(closestLtxIoRoot(document.documentElement));
  }, [contextRoot]);

  return contextRoot ?? fallback;
}

export { ThemeProvider, useTheme, useThemeRootElement, useThemeColorScheme };
