import { useState } from 'react'
import { Download, FolderOpen, MoreHorizontal, RefreshCw, WrapText } from 'lucide-react'
import { Button } from '@ds/Button/Button'
import { ItemAction } from '@ds/DropdownItems/ItemAction/ItemAction'
import { DropdownMenu } from '@ds/DropdownMenu/DropdownMenu'
import { useTheme } from '@ds/styles/themes/useTheme'
import { revealInFolderLabel } from '../../lib/revealInFolderLabel'
import { LogTooltip } from './LogTooltip'
import styles from './LogViewer.module.scss'

// The log dialog sits above --z-modal, so the menu must sit above the dialog.
// `style` replaces the menu's own style, so the width is set here too.
const MENU_STYLE = { zIndex: 'var(--z-max)', minWidth: '180px' }

interface LogActionsMenuProps {
  wrap: boolean
  onWrapChange: (wrap: boolean) => void
  isLoading: boolean
  hasLines: boolean
  onRefresh: () => void
  onDownload: () => void
  onRevealLog: () => void
}

/** The less common actions, so the toolbar keeps only the frequent ones in view. */
export function LogActionsMenu({
  wrap,
  onWrapChange,
  isLoading,
  hasLines,
  onRefresh,
  onDownload,
  onRevealLog,
}: LogActionsMenuProps) {
  const { rootElement } = useTheme()
  const [isOpen, setIsOpen] = useState(false)

  return (
    // The span lets the tooltip and the menu trigger share one button.
    <LogTooltip content="More actions" disabled={isOpen}>
      <span className={styles.tooltipAnchor}>
        <DropdownMenu
          side="bottom"
          align="end"
          sideOffset={8}
          collisionPadding={8}
          style={MENU_STYLE}
          portalContainer={rootElement}
          isOpen={isOpen}
          onMenuStateChange={setIsOpen}
          items={[
            {
              // Stays open, so the user sees the wrap state change.
              closeOnSelect: false,
              render: (
                <ItemAction
                  text="Word wrap"
                  leftIcon={<WrapText aria-hidden />}
                  isSelectable
                  isSelected={wrap}
                  selectionPosition="right"
                  onClick={() => onWrapChange(!wrap)}
                />
              ),
            },
            {
              hasUpperSeparator: true,
              isDisabled: isLoading,
              render: (
                <ItemAction text="Refresh" leftIcon={<RefreshCw aria-hidden />} onClick={onRefresh} />
              ),
            },
            {
              isDisabled: !hasLines,
              render: (
                <ItemAction text="Download" leftIcon={<Download aria-hidden />} onClick={onDownload} />
              ),
            },
            {
              render: (
                <ItemAction
                  text={revealInFolderLabel()}
                  leftIcon={<FolderOpen aria-hidden />}
                  onClick={onRevealLog}
                />
              ),
            },
          ]}
        >
          <Button
            hierarchy="plain"
            size="md"
            isIconOnly
            aria-label="More actions"
            aria-expanded={isOpen}
            leftIcon={<MoreHorizontal aria-hidden />}
          />
        </DropdownMenu>
      </span>
    </LogTooltip>
  )
}
