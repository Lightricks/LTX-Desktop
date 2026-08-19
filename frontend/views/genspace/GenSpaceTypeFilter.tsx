import { ChevronDown } from 'lucide-react'
import { SettingsDropdown } from '../../components/SettingsDropdown'
import {
  GENSPACE_TYPE_FILTER_OPTIONS,
  type GenSpaceTypeFilter,
} from '../../lib/genspace-gallery'

const toolbarTriggerClass = 'px-3 py-1.5 text-sm font-medium text-zinc-400 hover:text-white'

export function GenSpaceTypeFilter({
  value,
  onChange,
}: {
  value: GenSpaceTypeFilter
  onChange: (value: GenSpaceTypeFilter) => void
}) {
  const currentLabel = GENSPACE_TYPE_FILTER_OPTIONS.find(option => option.value === value)?.label ?? 'All'

  return (
    <SettingsDropdown
      placement="below"
      title="Type"
      value={value}
      onChange={next => onChange(next as GenSpaceTypeFilter)}
      triggerClassName={toolbarTriggerClass}
      trigger={
        <>
          <span>{currentLabel}</span>
          <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />
        </>
      }
      options={GENSPACE_TYPE_FILTER_OPTIONS}
    />
  )
}
