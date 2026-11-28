import { ChevronDown, ChevronUp } from 'lucide-react'
import { SettingsDropdown } from '../../components/SettingsDropdown'
import {
  GENSPACE_SORT_OPTIONS,
  type GenSpaceSortDir,
  type GenSpaceSortKey,
} from '../../lib/genspace-gallery'

const toolbarTriggerClass = 'px-3 py-1.5 text-sm font-medium text-fg-secondary hover:text-fg-primary rounded-r-none'

export function GenSpaceSortMenu({
  sortKey,
  sortDir,
  onSortKeyChange,
  onToggleSortDir,
}: {
  sortKey: GenSpaceSortKey
  sortDir: GenSpaceSortDir
  onSortKeyChange: (key: GenSpaceSortKey) => void
  onToggleSortDir: () => void
}) {
  const currentLabel = GENSPACE_SORT_OPTIONS.find(option => option.value === sortKey)?.label ?? 'Date'
  const DirectionIcon = sortDir === 'desc' ? ChevronDown : ChevronUp
  const directionLabel = sortDir === 'desc' ? 'Descending' : 'Ascending'

  return (
    <div className="flex items-center">
      <SettingsDropdown
        placement="below"
        title="Sort"
        value={sortKey}
        onChange={next => onSortKeyChange(next as GenSpaceSortKey)}
        triggerClassName={toolbarTriggerClass}
        trigger={
          <>
            <span>{currentLabel}</span>
            <ChevronDown className="h-3.5 w-3.5 text-fg-tertiary" />
          </>
        }
        options={GENSPACE_SORT_OPTIONS}
      />
      <button
        type="button"
        onClick={onToggleSortDir}
        title={directionLabel}
        aria-label={`Sort ${directionLabel.toLowerCase()}`}
        className="p-1.5 rounded-md rounded-l-none border-l border-separator text-fg-secondary hover:text-fg-primary hover:bg-action transition-colors"
      >
        <DirectionIcon className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}
