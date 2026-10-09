import { Button } from '@ds/Button/Button'
import { Text } from '@ds/Text/Text'

import type { ApiSuccessOf } from '@/lib/api-client'
import { formatBytes } from '@/lib/format'
import { firstRunLoraKey, type FirstRunLoraItem } from '@/lib/first-run-downloads'

import styles from '../FirstRunSetup.module.scss'
import { DownloadGroup } from './DownloadGroup'
import { DownloadItem } from './DownloadItem'
import { LoraDownloadItem } from './LoraDownloadItem'

type CheckpointDescriptor = ApiSuccessOf<'describeCheckpoints'>['checkpoints'][number]

interface ModelDownloadsCardProps {
  modelPackItems: CheckpointDescriptor[]
  encoderItem: CheckpointDescriptor | null
  encoderSelected: boolean
  onToggleEncoder: () => void
  recommendedItems: CheckpointDescriptor[]
  isRecommendedSelected: (item: CheckpointDescriptor) => boolean
  onToggleRecommended: (item: CheckpointDescriptor) => void
  imageItem: CheckpointDescriptor | null
  imageSelected: boolean
  onToggleImage: () => void
  loraItems: FirstRunLoraItem[]
  selectedLoraKeys: string[]
  onToggleLora: (key: string) => void
  allLorasSelected: boolean
  lorasIndeterminate: boolean
  onToggleAllLoras: () => void
  pendingLoraItems: FirstRunLoraItem[]
  totalDownloadBytes: number
  recommendationsError: string | null
  onRetryRecommendations: () => void
  installPath: string
  availableSpace: string
  onChangeModelsDir: () => void
}

const LORAS_GROUP_INFO =
  'Used by many tools. You can select the LoRAs you want — others can be added later.'

export function ModelDownloadsCard({
  modelPackItems,
  encoderItem,
  encoderSelected,
  onToggleEncoder,
  recommendedItems,
  isRecommendedSelected,
  onToggleRecommended,
  imageItem,
  imageSelected,
  onToggleImage,
  loraItems,
  selectedLoraKeys,
  onToggleLora,
  allLorasSelected,
  lorasIndeterminate,
  onToggleAllLoras,
  pendingLoraItems,
  totalDownloadBytes,
  recommendationsError,
  onRetryRecommendations,
  installPath,
  availableSpace,
  onChangeModelsDir,
}: ModelDownloadsCardProps) {
  return (
    <div className={styles.card}>
      {recommendationsError && (
        <div className={styles.recommendationsError}>
          <Text as="p" variant="body" size="md" className={styles.recommendationsErrorMessage}>
            {recommendationsError}
          </Text>
          <Button
            appearance="neutral"
            hierarchy="secondary"
            size="md"
            label="Retry"
            onClick={onRetryRecommendations}
          />
        </div>
      )}
      {(modelPackItems.length > 0 ||
        encoderItem ||
        imageItem ||
        recommendedItems.length > 0 ||
        loraItems.length > 0) && (
        <>
          <div className={styles.cardHeader}>
            <Text as="h3" variant="heading" size="md">
              Model downloads
            </Text>
          </div>
          {modelPackItems.length > 0 && (
            <DownloadGroup
              title="LTX 2.5"
              checked
              checkboxDisabled
              meta={
                modelPackItems.every((item) => item.downloaded)
                  ? 'Installed'
                  : formatBytes(
                      modelPackItems
                        .filter((item) => !item.downloaded)
                        .reduce((sum, item) => sum + item.size_bytes, 0),
                    )
              }
            >
              {modelPackItems.map((item) => (
                <DownloadItem key={item.cp_id} item={item} />
              ))}
            </DownloadGroup>
          )}
          {encoderItem && (
            <DownloadItem
              item={encoderItem}
              optIn={{
                checked: encoderSelected,
                onToggle: onToggleEncoder,
                skippedLabel: 'Use API key',
              }}
            />
          )}
          {recommendedItems.map((item) => (
            <DownloadItem
              key={item.cp_id}
              item={item}
              optIn={{
                checked: isRecommendedSelected(item),
                onToggle: () => onToggleRecommended(item),
                skippedLabel: 'Skipped',
              }}
            />
          ))}
          {imageItem && (
            <DownloadItem
              item={imageItem}
              optIn={{
                checked: imageSelected,
                onToggle: onToggleImage,
                skippedLabel: 'Skipped',
              }}
            />
          )}
          {loraItems.length > 0 && (
            <DownloadGroup
              title="LoRAs"
              infoContent={LORAS_GROUP_INFO}
              checked={allLorasSelected || loraItems.every((item) => item.downloaded)}
              indeterminate={lorasIndeterminate}
              checkboxDisabled={loraItems.every((item) => item.downloaded)}
              onToggle={onToggleAllLoras}
              meta={
                pendingLoraItems.length > 0
                  ? formatBytes(pendingLoraItems.reduce((sum, item) => sum + item.sizeBytes, 0))
                  : selectedLoraKeys.length === 0
                    ? 'None selected'
                    : 'Installed'
              }
            >
              {loraItems.map((item) => (
                <LoraDownloadItem
                  key={firstRunLoraKey(item)}
                  item={item}
                  checked={item.downloaded || selectedLoraKeys.includes(firstRunLoraKey(item))}
                  onToggle={() => onToggleLora(firstRunLoraKey(item))}
                />
              ))}
            </DownloadGroup>
          )}
          <div className={`${styles.metaRow} ${styles.totalRow}`}>
            <Text as="span" variant="heading" size="xxs">
              Total: {formatBytes(totalDownloadBytes)}
            </Text>
          </div>
        </>
      )}

      <div className={styles.locationBlock}>
        <div className={styles.locationHeader}>
          <Text as="span" variant="label" size="md" className={styles.locationLabel}>
            Models folder
          </Text>
          <Text as="span" variant="body" size="md" className={styles.locationFree}>
            {availableSpace} available
          </Text>
        </div>
        <div className={styles.pathRow} onClick={onChangeModelsDir}>
          <input
            type="text"
            value={installPath}
            readOnly
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                onChangeModelsDir()
              }
            }}
            aria-label="Models folder"
            className={styles.pathField}
          />
          <Button
            appearance="neutral"
            hierarchy="secondary"
            size="md"
            label="Change"
            onClick={(event) => {
              event.stopPropagation()
              onChangeModelsDir()
            }}
          />
        </div>
      </div>
    </div>
  )
}
