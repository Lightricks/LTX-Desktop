import React, { useState, useRef, useEffect, useMemo } from 'react'
import { Search } from 'lucide-react'
import { Text } from '@ds/Text/Text'

export interface MenuItem {
  id: string
  label: string
  shortcut?: string
  action?: () => void
  disabled?: boolean
  separator?: boolean
  submenu?: MenuItem[]
}

export interface MenuDefinition {
  id: string
  label: string
  items: MenuItem[]
}

interface MenuBarProps {
  menus: MenuDefinition[]
  rightContent?: React.ReactNode
}

export function MenuBar({ menus, rightContent }: MenuBarProps) {
  const [openMenuId, setOpenMenuId] = useState<string | null>(null)
  const [hoverMenuId, setHoverMenuId] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [highlightedResult, setHighlightedResult] = useState(0)
  const menuBarRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)

  const activeMenuId = openMenuId ? (hoverMenuId || openMenuId) : null

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuBarRef.current && !menuBarRef.current.contains(e.target as Node)) {
        setOpenMenuId(null)
        setHoverMenuId(null)
      }
    }
    if (openMenuId) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [openMenuId])

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpenMenuId(null)
        setHoverMenuId(null)
        setSearchQuery('')
      }
    }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [])

  const searchResults = useMemo(() => {
    const query = searchQuery.trim().toLowerCase()
    if (!query) return []
    const results: { menuLabel: string; item: MenuItem }[] = []
    for (const menu of menus) {
      for (const item of menu.items) {
        if (item.separator) continue
        if (item.label.toLowerCase().includes(query)) {
          results.push({ menuLabel: menu.label, item })
        }
        if (item.submenu) {
          for (const sub of item.submenu) {
            if (sub.separator) continue
            if (sub.label.toLowerCase().includes(query)) {
              results.push({ menuLabel: `${menu.label} > ${item.label}`, item: sub })
            }
          }
        }
      }
    }
    return results
  }, [menus, searchQuery])

  useEffect(() => {
    setHighlightedResult(prev => {
      if (searchResults.length === 0) return 0
      return Math.min(prev, searchResults.length - 1)
    })
  }, [searchResults.length])

  const handleItemClick = (item: MenuItem) => {
    if (item.disabled || !item.action) return
    item.action()
    setOpenMenuId(null)
    setHoverMenuId(null)
  }

  const handleSearchResultClick = (item: MenuItem) => {
    if (item.disabled || !item.action) return
    item.action()
    setSearchQuery('')
    setOpenMenuId(null)
    setHoverMenuId(null)
  }

  const handleSearchKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlightedResult(prev => Math.min(prev + 1, searchResults.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlightedResult(prev => Math.max(prev - 1, 0))
    } else if (e.key === 'Enter' && searchResults[highlightedResult]) {
      e.preventDefault()
      handleSearchResultClick(searchResults[highlightedResult].item)
    }
  }

  const renderMenuItem = (item: MenuItem, index: number) => {
    if (item.separator) {
      return <div key={`sep-${index}`} className="h-px bg-separator-secondary my-1 mx-2" />
    }

    return (
      <button
        key={item.id}
        onClick={() => handleItemClick(item)}
        disabled={item.disabled}
        className={`w-full flex items-center justify-between px-3 py-1.5 text-left transition-colors ${
          item.disabled
            ? 'text-fg-tertiary cursor-not-allowed'
            : 'text-fg-primary hover:bg-action-hover'
        }`}
      >
        <Text as="span" variant="body" size="md">{item.label}</Text>
        {item.shortcut && (
          <Text
            as="span"
            variant="body"
            size="sm"
            className={`ml-8 ${item.disabled ? 'text-fg-tertiary' : 'text-fg-secondary'}`}
          >
            {item.shortcut}
          </Text>
        )}
      </button>
    )
  }

  return (
    <div ref={menuBarRef} className="flex items-center bg-surface-primary border-b border-separator-secondary select-none relative z-[60]">
      <div className="flex items-center flex-1">
      {menus.map(menu => {
        const isActive = activeMenuId === menu.id
        const isHelpMenu = menu.id === 'help'

        return (
          <div key={menu.id} className="relative">
            <button
              onMouseDown={() => {
                if (openMenuId === menu.id) {
                  setOpenMenuId(null)
                  setHoverMenuId(null)
                } else {
                  setOpenMenuId(menu.id)
                  setHoverMenuId(null)
                  if (isHelpMenu) {
                    setTimeout(() => searchInputRef.current?.focus(), 50)
                  }
                }
              }}
              onMouseEnter={() => {
                if (openMenuId) setHoverMenuId(menu.id)
              }}
              className={`px-3 py-1.5 transition-colors ${
                isActive
                  ? 'bg-action text-fg-primary'
                  : 'text-fg-secondary hover:text-fg-primary'
              }`}
            >
              <Text as="span" variant="label" size="md">{menu.label}</Text>
            </button>

            {isActive && (
              <div className="absolute top-full left-0 min-w-[240px] bg-surface-primary border border-separator-secondary rounded-b-lg shadow-xl py-1 z-[60]">
                {isHelpMenu && (
                  <div className="px-2 py-1.5 border-b border-separator-secondary">
                    <div className="flex items-center gap-2 bg-surface-secondary rounded px-2 py-1">
                      <Search className="h-3.5 w-3.5 text-fg-tertiary flex-shrink-0" />
                      <input
                        ref={searchInputRef}
                        type="text"
                        placeholder="Search menus..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyDown={handleSearchKeyDown}
                        className="flex-1 bg-transparent text-sm text-fg-primary placeholder-fg-tertiary outline-none"
                        autoFocus
                      />
                    </div>
                    {searchQuery && (
                      <div className="mt-1 max-h-48 overflow-y-auto">
                        {searchResults.length === 0 ? (
                          <Text as="div" variant="body" size="sm" align="center" className="px-2 py-2 text-fg-tertiary">
                            No results
                          </Text>
                        ) : (
                          searchResults.map((result, i) => (
                            <button
                              key={`${result.item.id}-${i}`}
                              onClick={() => handleSearchResultClick(result.item)}
                              className={`w-full flex items-center justify-between px-2 py-1.5 text-left rounded transition-colors ${
                                i === highlightedResult
                                  ? 'bg-brand text-fg-white'
                                  : 'text-fg-secondary hover:bg-action'
                              }`}
                            >
                              <div>
                                <Text as="span" variant="body" size="sm">{result.item.label}</Text>
                                <Text as="span" variant="body" size="xs" className="ml-2 text-fg-tertiary">{result.menuLabel}</Text>
                              </div>
                              {result.item.shortcut && (
                                <Text as="span" variant="body" size="xs" className="text-fg-tertiary">{result.item.shortcut}</Text>
                              )}
                            </button>
                          ))
                        )}
                      </div>
                    )}
                  </div>
                )}

                {menu.items.map((item, i) => renderMenuItem(item, i))}
              </div>
            )}
          </div>
        )
      })}
      </div>
      {rightContent && (
        <div className="flex items-center mr-2">
          {rightContent}
        </div>
      )}
    </div>
  )
}
