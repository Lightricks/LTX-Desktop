import { clsx } from 'clsx'
import { Check } from 'lucide-react'
import type { ReactNode } from 'react'

import styles from './ModelCatalogTable.module.scss'

export {
  confirmCatalogModelDelete,
  modelCatalogDownloadButtonLabel,
} from '../../lib/model-catalog'

export const LTX_CANNOT_DELETE_DEFAULT_MODEL_TOOLTIP =
  'This model is the default for local generation. Choose another installed version as default before deleting it.'

export const LOCAL_TEXT_ENCODER_REQUIRED_TOOLTIP =
  'This encoder is the only way to encode prompts right now. Switch to LTX API encoding under General → Text encoding (an API key is required) before deleting it.'

export const modelCatalogSettingsTooltipPortal =
  typeof document !== 'undefined' ? document.body : null

export function ModelCatalogTable({ children }: { children: ReactNode }) {
  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <colgroup>
          <col className={styles.colName} />
          <col className={styles.colSize} />
          <col className={styles.colActions} />
        </colgroup>
        <tbody>{children}</tbody>
      </table>
    </div>
  )
}

export function ModelCatalogEmpty({ message }: { message: string }) {
  return <p className={styles.empty}>{message}</p>
}

export function modelCatalogRowClassNames(opts: { clickable?: boolean } = {}): string {
  return [styles.dataRow, opts.clickable ? styles.rowClickable : '']
    .filter(Boolean)
    .join(' ')
}

export function ModelCatalogNameCell({ children }: { children: ReactNode }) {
  return (
    <td className={styles.nameCell}>
      <div className={styles.cellInner}>
        <div className={styles.nameLine}>{children}</div>
      </div>
    </td>
  )
}

export function ModelCatalogSizeCell({ children }: { children: ReactNode }) {
  return (
    <td className={styles.sizeCell}>
      <div className={styles.cellInner}>{children}</div>
    </td>
  )
}

export function ModelCatalogCellText({ children }: { children: ReactNode }) {
  return <span className={styles.cellText}>{children}</span>
}

/** Shown after the model name when the checkpoint is on disk. */
export function ModelCatalogInstalledCheck() {
  return (
    <span className={styles.installedCheckWrap} aria-label="Installed">
      <Check className={styles.installedCheck} strokeWidth={2.5} aria-hidden />
    </span>
  )
}

export function ModelCatalogSizeText({ children }: { children: ReactNode }) {
  return <span className={styles.cellTextSize}>{children}</span>
}

export function ModelCatalogActionsCell({ children }: { children: ReactNode }) {
  return <td className={styles.actionsCell}>{children}</td>
}

/** Right-aligned action slot: optional status pill + primary control share one grid. */
export function ModelCatalogActionGroup({
  leading,
  status,
  action,
}: {
  leading?: ReactNode
  status?: ReactNode
  action?: ReactNode
}) {
  const splitStatusAndAction = Boolean(status && action && !leading)

  return (
    <ModelCatalogActionsCell>
      <div className={styles.cellInnerEnd}>
        <div
          className={clsx(styles.actionRow, splitStatusAndAction && styles.actionRowSplit)}
        >
          {leading}
          {status}
          {action}
        </div>
      </div>
    </ModelCatalogActionsCell>
  )
}

/** Gray secondary actions (Download, Delete). */
export function modelCatalogActionButtonClassName(): string {
  return styles.actionButton
}

/** Single primary action in the row (Download or Active-only slot). */
export function modelCatalogActionButtonWideClassName(): string {
  return styles.actionButtonWide
}

export function ModelCatalogDetailRow({ children }: { children: ReactNode }) {
  return (
    <tr className={styles.detailRow}>
      <td colSpan={3}>{children}</td>
    </tr>
  )
}

export function ModelCatalogStatusBadge({
  label,
  icon,
  solo,
}: {
  label: string
  icon?: ReactNode
  /** Match width of Download/Delete when this badge is the only control. */
  solo?: boolean
}) {
  return (
    <span className={[styles.statusBadge, solo ? styles.statusBadgeSolo : ''].filter(Boolean).join(' ')}>
      {icon}
      {label}
    </span>
  )
}

export { styles as modelCatalogTableStyles }
