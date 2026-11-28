import { useCallback, useMemo, useState } from 'react'
import type { Asset } from '../../types/project-model'
import {
  defaultSortDir,
  filterGenSpaceAssets,
  sortGenSpaceAssets,
  type GenSpaceSortDir,
  type GenSpaceSortKey,
  type GenSpaceTypeFilter,
} from '../../lib/genspace-gallery'

export function useGenSpaceGallery(assets: Asset[]) {
  const [typeFilter, setTypeFilter] = useState<GenSpaceTypeFilter>('all')
  const [sortKey, setSortKey] = useState<GenSpaceSortKey>('createdAt')
  const [sortDir, setSortDir] = useState<GenSpaceSortDir>(() => defaultSortDir('createdAt'))
  const [showFavorites, setShowFavorites] = useState(false)

  const filteredAssets = useMemo(
    () => sortGenSpaceAssets(
      filterGenSpaceAssets(assets, typeFilter, showFavorites),
      sortKey,
      sortDir,
    ),
    [assets, typeFilter, showFavorites, sortKey, sortDir],
  )

  const favoriteCount = useMemo(
    () => filterGenSpaceAssets(assets, 'all', true).length,
    [assets],
  )

  const hasTypeMatches = useMemo(
    () => filterGenSpaceAssets(assets, typeFilter, false).length > 0,
    [assets, typeFilter],
  )

  const onSortKeyChange = useCallback((key: GenSpaceSortKey) => {
    if (key === sortKey) {
      setSortDir(direction => (direction === 'asc' ? 'desc' : 'asc'))
      return
    }
    setSortKey(key)
    setSortDir(defaultSortDir(key))
  }, [sortKey])

  const onToggleSortDir = useCallback(() => {
    setSortDir(direction => (direction === 'asc' ? 'desc' : 'asc'))
  }, [])

  const onToggleFavorites = useCallback(() => {
    setShowFavorites(value => !value)
  }, [])

  return {
    typeFilter,
    setTypeFilter,
    sortKey,
    sortDir,
    onSortKeyChange,
    onToggleSortDir,
    showFavorites,
    onToggleFavorites,
    filteredAssets,
    favoriteCount,
    hasTypeMatches,
  }
}
