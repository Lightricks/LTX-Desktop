import { useState } from 'react'
import { Type } from 'lucide-react'
import { Button } from '@ds/Button/Button'
import { ItemAction } from '@ds/DropdownItems/ItemAction/ItemAction'
import { DropdownPicker } from '@ds/DropdownPicker/DropdownPicker'
import { useTheme } from '@ds/styles/themes/useTheme'
import { LogTooltip } from './LogTooltip'
import { LOG_FONT_SIZES, type LogFontSizeId } from './useLogViewPrefs'
import styles from './LogViewer.module.scss'

// The log dialog sits above --z-modal, so the menu must sit above the dialog.
const PICKER_CONTENT_PROPS = {
  side: 'bottom' as const,
  align: 'end' as const,
  sideOffset: 8,
  avoidCollisions: true,
  collisionPadding: 8,
  style: { zIndex: 'var(--z-max)' },
}

export function LogFontSizePicker({
  value,
  onValueChange,
}: {
  value: LogFontSizeId
  onValueChange: (value: string) => void
}) {
  const { rootElement } = useTheme()
  const [isOpen, setIsOpen] = useState(false)

  return (
    // The span lets the tooltip and the menu trigger share one button.
    <LogTooltip content="Font size" disabled={isOpen}>
      <span className={styles.tooltipAnchor}>
        <DropdownPicker
          options={LOG_FONT_SIZES.map((size) => ({
            value: size.value,
            label: size.label,
            render: (
              <ItemAction
                text={size.label}
                textVariant="body"
                selectionPosition="right"
                isSelectable
                isSelected={value === size.value}
              />
            ),
          }))}
          value={value}
          label="Log font size"
          isOpen={isOpen}
          onOpenChange={setIsOpen}
          onValueChange={onValueChange}
          portalContainer={rootElement}
          contentProps={PICKER_CONTENT_PROPS}
          minWidth="160px"
        >
          <Button
            appearance="neutral"
            hierarchy="plain"
            size="md"
            isIconOnly
            aria-expanded={isOpen}
            aria-label="Log font size"
            leftIcon={<Type aria-hidden />}
          />
        </DropdownPicker>
      </span>
    </LogTooltip>
  )
}
