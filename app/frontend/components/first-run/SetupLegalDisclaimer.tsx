import { Text } from '@ds/Text/Text'

import {
  LTX_COMMUNITY_LICENSE_URL,
  LTX_PRIVACY_POLICY_URL,
  openLegalUrl,
} from '@/lib/legal-links'
import styles from '../FirstRunSetup.module.scss'

function LegalLink({ href, children }: { href: string; children: string }) {
  return (
    <a
      className={styles.legalLink}
      href={href}
      onClick={(event) => {
        event.preventDefault()
        openLegalUrl(href)
      }}
    >
      {children}
    </a>
  )
}

export function SetupLegalDisclaimer({ className }: { className?: string }) {
  return (
    <Text as="p" variant="body" size="md" className={className}>
      By using this software, you agree to the{' '}
      <LegalLink href={LTX_COMMUNITY_LICENSE_URL}>LTX Community License</LegalLink>
      {' '}and{' '}
      <LegalLink href={LTX_PRIVACY_POLICY_URL}>LTX privacy policy</LegalLink>
      .
    </Text>
  )
}
