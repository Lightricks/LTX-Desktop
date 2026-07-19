import type { components } from '../generated/backend-openapi'
import type { ApiSuccessOf } from './api-client'
import type { LoraCatalogListItem } from '../hooks/use-catalog'

type InstructionSection = NonNullable<components['schemas']['LoraCatalogItem']['instructions']>[number]
type LoraCatalogItem = components['schemas']['LoraCatalogItem']

// One row the library renders, normalized across both kinds and across catalog vs.
// on-disk-only (manually-placed) items. Catalog-only fields (instructions/repoId/
// requiresHfLogin/sizeBytes) are absent for on-disk-only files → card degrades to
// filename + "Use". installedPath is set once an item is on disk (enables "Use").
export interface LibraryEntry {
  id: string
  name: string
  description?: string
  sizeBytes?: number
  downloaded: boolean
  installedPath?: string
  // variant id → absolute/installed path for each on-disk checkpoint.
  variantInstalledPaths?: Record<string, string>
  recommendedStrength?: number
  instructions?: InstructionSection[]
  repoId?: string
  requiresHfLogin?: boolean
  thumbnailUrl?: string
  demoVideoUrl?: string
  author?: { name: string; url?: string | null; affiliation?: 'ltx' | 'community' }
  license?: { name: string; url?: string | null }
  // When length > 1, the library card shows a variant picker (download per checkpoint).
  variants?: { id: string; label: string; filename: string; sizeBytes: number }[]
  defaultVariantId?: string
  // Subset of variants[].id present on disk (API ∪ on-disk filename match).
  downloadedVariantIds?: string[]
}

// Map a catalog item (plain or IC — IcLoraCatalogItem extends the base) to the shared
// LibraryEntry fields. Callers add the per-kind bits (downloaded, installedPath).
export function catalogItemToEntry(item: LoraCatalogItem): Omit<LibraryEntry, 'downloaded'> {
  // First variant is the catalog default (SSOT with backend DownloadSpec.default_variant()).
  const variants = item.download.variants.map(v => ({
    id: v.id,
    label: v.label,
    filename: v.filename,
    sizeBytes: v.size_bytes,
  }))
  const defaultVariant = variants[0]
  return {
    id: item.id,
    name: item.name,
    description: item.description || undefined,
    sizeBytes: defaultVariant.sizeBytes,
    recommendedStrength: item.recommended_strength ?? undefined,
    instructions: item.instructions ?? undefined,
    repoId: item.download.repo_id,
    requiresHfLogin: item.requires_hf_login,
    thumbnailUrl: item.media?.thumbnail ?? undefined,
    demoVideoUrl: item.media?.demo_video ?? undefined,
    author: item.author ?? undefined,
    license: item.license ?? undefined,
    variants,
    defaultVariantId: defaultVariant.id,
  }
}

// Shared catalog id ↔ variant encoding for selector values and download session keys.
export const CATALOG_VARIANT_SEP = '::'

export function catalogVariantKey(catalogId: string, variantId?: string | null): string {
  return variantId ? `${catalogId}${CATALOG_VARIANT_SEP}${variantId}` : catalogId
}

export function parseCatalogVariantKey(value: string): { catalogId: string; variantId?: string } {
  const sep = value.indexOf(CATALOG_VARIANT_SEP)
  if (sep === -1) return { catalogId: value }
  return { catalogId: value.slice(0, sep), variantId: value.slice(sep + CATALOG_VARIANT_SEP.length) }
}

/** True when the selector should emit one option per installed variant. */
export function hasMultiVariantSelector(
  variants: { id: string }[] | undefined,
  downloadedVariantIds: string[] | undefined,
): boolean {
  return (variants?.length ?? 0) > 1 && (downloadedVariantIds?.length ?? 0) > 0
}

/** Encode selector value using the same rule as buildIcLoraSelectorOptions. */
export function encodeIcLoraSelectorValue(
  catalogId: string,
  variantId: string | null | undefined,
  opts?: { variants?: { id: string }[]; downloadedVariantIds?: string[] },
): string {
  if (opts && hasMultiVariantSelector(opts.variants, opts.downloadedVariantIds) && variantId) {
    return catalogVariantKey(catalogId, variantId)
  }
  return catalogId
}

export function parseIcLoraSelectorValue(value: string): { catalogId: string; variantId?: string } {
  return parseCatalogVariantKey(value)
}

export function buildIcLoraSelectorOptions(
  entries: {
    id: string
    name: string
    downloaded: boolean
    downloadedVariantIds?: string[]
    variants?: { id: string; label: string }[]
  }[],
  baseTypes: readonly { value: string; label: string }[],
  opts: { includeCatalog: boolean; includeCustom: boolean },
): { value: string; label: string }[] {
  const custom = baseTypes.find(t => t.value === 'custom')
  const catalogOpts: { value: string; label: string }[] = []
  if (opts.includeCatalog) {
    for (const e of entries) {
      if (!e.downloaded) continue
      if (hasMultiVariantSelector(e.variants, e.downloadedVariantIds)) {
        const ready = new Set(e.downloadedVariantIds)
        for (const v of e.variants!) {
          if (!ready.has(v.id)) continue
          catalogOpts.push({
            value: catalogVariantKey(e.id, v.id),
            label: `${e.name} — ${v.label}`,
          })
        }
      } else {
        catalogOpts.push({ value: e.id, label: e.name })
      }
    }
  }
  return [
    ...baseTypes.filter(t => t.value !== 'custom').map(t => ({ value: t.value, label: t.label })),
    ...catalogOpts,
    ...(opts.includeCustom && custom ? [{ value: 'custom', label: custom.label }] : []),
  ]
}

const baseName = (p: string): string => p.split(/[\\/]/).pop() ?? p
const normalizeSlashes = (p: string): string => p.replace(/\\/g, '/')

/** Prefer API-reported variant ids; also mark any variant whose filename is among installed basenames. */
export function resolveDownloadedVariantIds(
  variants: { id: string; filename: string }[] | undefined,
  apiIds: string[] | undefined,
  installedPaths: string[],
): string[] {
  const fromApi = apiIds ?? []
  if (!variants?.length) return fromApi
  const installedNames = new Set(installedPaths.map(baseName))
  const fromDisk = variants.filter(v => installedNames.has(v.filename)).map(v => v.id)
  return [...new Set([...fromApi, ...fromDisk])]
}

/** Prefer the catalog default when installed; otherwise the first installed variant. Never an undownloaded id. */
export function preferredVariantId(
  variants: { id: string }[] | undefined,
  defaultVariantId: string | undefined,
  downloadedVariantIds: string[] | undefined,
): string | undefined {
  if (!variants?.length) return undefined
  const ready = new Set(downloadedVariantIds ?? [])
  if (ready.size === 0) return undefined
  if (defaultVariantId && ready.has(defaultVariantId)) return defaultVariantId
  return variants.find(v => ready.has(v.id))?.id
}

/** Name for UI chips/pickers. Append ` — {label}` only when the catalog item has multiple variants. */
export function variantDisplayName(
  name: string,
  variantLabel?: string | null,
  variantCount?: number,
): string {
  if (!variantLabel || (variantCount !== undefined && variantCount <= 1)) return name
  return `${name} — ${variantLabel}`
}

// Never substitutes a different variant's path: a caller asking for a specific variant
// must get exactly that file or nothing (see use-lora-library's useEntry for the retry path).
export function resolveInstalledPath(
  entry: LibraryEntry,
  variantId?: string | null,
): string | undefined {
  if (!variantId) return entry.installedPath
  return entry.variantInstalledPaths?.[variantId]
}

// Persisted LoRA refs (generationParams) are stored relative to modelsDir instead of
// absolute so a project survives a reinstall or a move to another machine — modelsDir
// varies per install. The backend resolves relative refs against modelsDir directly
// (resolve_lora_ref), so no conversion back to absolute is needed to use one.
export function toModelsDirRelativeRef(absPath: string, modelsDir: string): string {
  if (!modelsDir) return absPath
  const base = normalizeSlashes(modelsDir).replace(/\/+$/, '')
  const norm = normalizeSlashes(absPath)
  if (!norm.toLowerCase().startsWith(`${base.toLowerCase()}/`)) return absPath
  return norm.slice(base.length + 1)
}

// Merge the plain-LoRA catalog with on-disk files into the library gallery list.
// Cross-reference by download filename: a catalog entry carries its info + installed
// path (if present). Installed catalog LoRAs appear once (enriched), never duplicated —
// all variant files are marked matched. On-disk files with no catalog match are omitted
// from the gallery; they remain available in the generation selector via listModels.
export function mergeLoraLibrary(
  catalog: LoraCatalogListItem[],
  installed: ApiSuccessOf<'listModels'>['models'],
): LibraryEntry[] {
  const byFilename = new Map(installed.map(m => [baseName(m.path), m]))
  const installedPaths = installed.map(m => m.path)
  return catalog.map(c => {
    const base = catalogItemToEntry(c.lora)
    const downloadedVariantIds = resolveDownloadedVariantIds(
      base.variants,
      c.downloaded_variant_ids,
      installedPaths,
    )
    const variantInstalledPaths: Record<string, string> = {}
    for (const v of base.variants ?? []) {
      const m = byFilename.get(v.filename)
      if (!m) continue
      variantInstalledPaths[v.id] = m.path
    }
    const preferredId = preferredVariantId(base.variants, base.defaultVariantId, downloadedVariantIds)
    const installedPath =
      (preferredId ? variantInstalledPaths[preferredId] : undefined)
      ?? Object.values(variantInstalledPaths)[0]
    return {
      ...base,
      downloaded: c.downloaded || downloadedVariantIds.length > 0 || Boolean(installedPath),
      downloadedVariantIds,
      variantInstalledPaths: Object.keys(variantInstalledPaths).length > 0 ? variantInstalledPaths : undefined,
      installedPath,
      sizeBytes: (installedPath ? byFilename.get(baseName(installedPath))?.size_bytes : undefined)
        ?? base.sizeBytes,
    }
  })
}
