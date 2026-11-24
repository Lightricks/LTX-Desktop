import { logger } from '../../lib/logger.ts'

export const HOME_EXTERNAL_URLS = {
  contactSales: 'https://ltx.io/forms/ltx-contact-sales?placement=desktop_home',
  support: 'https://support.ltx.studio/',
  reportIssue: 'https://github.com/Lightricks/LTX-Desktop/issues/new',
} as const

export type HomeExternalUrl =
  (typeof HOME_EXTERNAL_URLS)[keyof typeof HOME_EXTERNAL_URLS]

function getElectronOpenExternalUrl():
  | ((input: { url: string }) => Promise<boolean>)
  | undefined {
  return globalThis.window?.electronAPI?.openExternalUrl
}

export function shouldInterceptHomeExternalUrl(): boolean {
  return getElectronOpenExternalUrl() != null
}

/** Opens http(s) in the OS browser on Desktop; Remote uses `window.open`. */
export function openExternalBrowserUrl(url: string): void {
  const openExternalUrl = getElectronOpenExternalUrl()
  if (openExternalUrl) {
    void openExternalUrl({ url }).catch((error: unknown) => {
      logger.warn(`Failed to open external Home link: ${String(error)}`)
    })
    return
  }
  globalThis.window?.open?.(url, '_blank', 'noopener,noreferrer')
}

export function openHomeExternalUrl(url: HomeExternalUrl): void {
  openExternalBrowserUrl(url)
}
