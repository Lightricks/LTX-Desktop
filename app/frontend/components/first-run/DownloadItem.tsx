import { Text } from '@ds/Text/Text'

import type { ApiSuccessOf } from '@/lib/api-client'
import { formatBytes } from '@/lib/format'

import { Checkbox } from '../../ltx-io/components/Checkbox/Checkbox'
import { Info } from '../../ltx-io/components/shared/Info/Info'
import styles from '../FirstRunSetup.module.scss'

type CheckpointDescriptor = ApiSuccessOf<'describeCheckpoints'>['checkpoints'][number]

interface DownloadItemProps {
  item: CheckpointDescriptor
  optIn?: { checked: boolean; onToggle: () => void; skippedLabel?: string }
  showCheckbox?: boolean
}

// User-facing explanation per checkpoint role (moved out of the backend so wording/i18n
// iterates without a backend deploy). Keyed by the stable `role` the backend ships.
const CP_INFO_BY_ROLE: Record<CheckpointDescriptor['role'], string> = {
  base: 'Core LTX video model. Turns your prompt into video frames.',
  upscaler: 'Doubles video resolution. Also reduces glitches and stray artifacts near the end of longer clips.',
  text_encoder: 'Reads your prompt so the model can understand it. Recommended — skip only if you will use an LTX API key.',
  vae: 'Turns the model’s internal frames into playable video (and audio). Required for this install.',
  image: 'Generates still images from text. Used for image-to-video and image tools.',
  support: 'Supporting model for guided generation (depth, edges, or pose).',
  prompt_enhancer: 'Rewrites your prompt into a richer caption before generation. Optional — add later from Settings → Models.',
}

// One line in the first-run "What will be downloaded" list: checkpoint name, an
// info-tooltip icon, and the size. Items an API key covers get a checkbox instead of being
// silently dropped, so the download stays available to anyone who wants to run offline.
export function DownloadItem({ item, optIn, showCheckbox = true }: DownloadItemProps) {
  const onDisk = item.downloaded
  const canToggle = showCheckbox && optIn !== undefined && !onDisk
  const skipped = canToggle && !optIn.checked
  return (
    <div className={styles.downloadRow}>
      <span className={styles.downloadName}>
        {showCheckbox ? (
          <Checkbox
            checked={onDisk || (canToggle ? optIn.checked : true)}
            disabled={!canToggle}
            appearance="neutral"
            onChange={() => optIn?.onToggle()}
          >
            <Text as="span" variant="body" size="md">
              {item.name}
            </Text>
          </Checkbox>
        ) : (
          <Text as="span" variant="body" size="md">
            {item.name}
          </Text>
        )}
        <Info content={CP_INFO_BY_ROLE[item.role]} side="top" maxWidth={280} />
      </span>
      <Text
        as="span"
        variant="body"
        size="md"
        className={`${styles.downloadMeta} ${onDisk ? styles.downloadMetaInstalled : ''}`}
      >
        {onDisk ? 'Installed' : skipped ? (optIn?.skippedLabel ?? 'Skipped (API key)') : formatBytes(item.size_bytes)}
      </Text>
    </div>
  )
}
