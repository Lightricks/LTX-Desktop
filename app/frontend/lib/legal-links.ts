import { openExternalBrowserUrl } from '@/components/home/home-external-links'

/** LTX-2 Community License — same URL the catalog and Home already use. */
export const LTX_COMMUNITY_LICENSE_URL =
  'https://github.com/Lightricks/LTX-2/blob/main/LICENSE'

/** LTX Platform Privacy Policy (Lightricks legal). */
export const LTX_PRIVACY_POLICY_URL =
  'https://static.lightricks.com/legal/Privacy%20Policy%20-%20LTX%20Platform.pdf'

export function openLegalUrl(url: string): void {
  openExternalBrowserUrl(url)
}
