/**
 * Window chrome contract.
 *
 * Mac embeds chrome in the page (hidden title bar, in-page traffic lights, glass).
 * Windows and Linux keep a native caption above the webview — the page starts at y=0.
 * Install hides the Windows/Linux File menu; the caption stays. The main app shows it again.
 * CSS follows `html[data-platform="darwin"]`; do not add win32/linux platform selectors.
 * The one exception is Mac with an RTL system language: macOS mirrors the traffic lights to the
 * right, and `macTrafficLightsOnEnd()` sets `html[data-traffic-lights="end"]` (never on win32/linux).
 */

/** Matches `--color-gray-0` / `--color-gray-900` — Electron has no CSS variables. */
export const WINDOW_BACKGROUND = {
  light: '#ffffff',
  dark: '#232424',
} as const

export const WINDOW_BACKGROUND_CLEAR = '#00000000'

/**
 * macOS traffic lights. x=16 is `--spacing-lg`; they sit in a `--spacing-4xl`
 * (`--window-chrome-top`) strip. Keep this aligned with the CSS token — do not invent a new px.
 */
export const MACOS_TRAFFIC_LIGHT_POSITION = { x: 16, y: 18 } as const

const RTL_SCRIPTS = new Set(['Arab', 'Hebr', 'Syrc', 'Thaa', 'Nkoo', 'Adlm', 'Rohg', 'Mand', 'Samr', 'Mend'])

function isRtlLocale(locale: string): boolean {
  try {
    return RTL_SCRIPTS.has(new Intl.Locale(locale).maximize().script ?? '')
  } catch {
    // A malformed tag must not blank the app; fall back to the LTR layout.
    return false
  }
}

/**
 * macOS mirrors the traffic lights to the top-right when the system language is RTL
 * (the `x: 16` position mirrors with them). The page itself stays LTR, so the renderer
 * uses this to make room on the right. Windows and Linux keep the native caption.
 */
export function macTrafficLightsOnEnd(platform: string | undefined, locale: string): boolean {
  return platform === 'darwin' && isRtlLocale(locale)
}

/**
 * Mac vibrancy for install and the main app. `sidebar` (like `under-window`) tints
 * toward neutral gray instead of transmitting the wallpaper; `popover` reads as
 * glass over pale desktops.
 */
export const GLASS_MATERIAL = 'sidebar' as const

/**
 * Fallback when the user hasn't chosen a window appearance yet.
 * After that, main persists the in-app theme preference (`light` / `dark` /
 * `system`) so Mac glass matches the UI instead of only the OS.
 */
export const APP_THEME_PREFERENCE = 'system' as const
