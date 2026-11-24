/** Copy for catalog Download while a job is in flight. Percent only: the action slot is 8.5rem with a spinner. */
export function modelCatalogDownloadButtonLabel(
  downloading: boolean,
  percent: number,
): string {
  if (!downloading) return 'Download'
  return `${Math.round(percent)}%`
}

export function catalogModelDeleteConfirmMessage(modelName: string): string {
  return `Delete "${modelName}"? This removes the downloaded weights from this computer.`
}

export function confirmCatalogModelDelete(modelName: string): boolean {
  return window.confirm(catalogModelDeleteConfirmMessage(modelName))
}

export const ENCODING_MODEL_SEARCH_HAYSTACK =
  'encoding models local text encoder gemma local prompt enhancer'

export const IMAGE_MODEL_SEARCH_HAYSTACK = 'image models z image turbo text-to-image'

export function matchesCatalogSearch(haystack: string, query: string): boolean {
  if (!query) return true
  return haystack.toLowerCase().includes(query)
}
