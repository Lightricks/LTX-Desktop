/** Copy for Explore when local generation is not viable. Must not mention the API. */

export const LOCAL_GENERATION_UNSUPPORTED_TITLE = 'Coming soon to your device'
export const LOCAL_GENERATION_UNSUPPORTED_BODY =
  "Local generation isn't available here yet.\nIn the meantime, keep creating with LTX Explore on the web."
export const LOCAL_GENERATION_UNSUPPORTED_WEB_APP_LABEL = 'Go to LTX Explore'
export const LOCAL_GENERATION_UNSUPPORTED_REQUIREMENTS_LABEL = 'View system requirements'

export const LOCAL_GENERATION_UNSUPPORTED_URLS = {
  webApp: 'https://app.ltx.io/',
  systemRequirements: 'https://ltx.io/ltx-desktop-nh#installation-guide',
} as const

// Underwater banner clip from ltx.io. videos.ltx.io is allowed by media-src in electron/csp.ts.
export const LOCAL_GENERATION_UNSUPPORTED_VIDEO_URL =
  'https://videos.ltx.io/LTX-2/Banners/water-original-opt.mp4'
