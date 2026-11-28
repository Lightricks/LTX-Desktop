import type { SettingsOpenDetail } from './settings-navigation'

/** Opens Settings on API Keys and focuses the LTX input (see useApiKeyFocus). */
export const LTX_KEY_REQUIRED_SETTINGS_DETAIL = {
  tab: 'apiKeys',
  reason: 'ltxKeyRequired',
} as const satisfies SettingsOpenDetail

/** Opens Settings on General, scrolled to Text encoding. */
export const TEXT_ENCODING_SETTINGS_DETAIL = {
  tab: 'general',
  scrollAnchor: 'textEncoding',
} as const satisfies SettingsOpenDetail
