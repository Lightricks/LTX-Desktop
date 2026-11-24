import type { EnhanceProvider } from '../hooks/use-prompt-enhancer-provider'

/** Menu above the Gen Space / Home Enhance split button. */
export const ENHANCE_VIA_MENU_TITLE = 'Enhance via'

/** API first — matches the common default when local Gemma is unavailable. */
export const ENHANCE_PROVIDER_MENU_OPTIONS: ReadonlyArray<{
  value: EnhanceProvider
  label: string
}> = [
  { value: 'api', label: 'Gemini API' },
  { value: 'local', label: 'Local Gemma' },
]

export function enhanceActionLabel(
  provider: EnhanceProvider,
  isEnhancing: boolean,
): string {
  if (isEnhancing) return 'Enhancing...'
  return provider === 'api' ? 'Enhance (Gemini)' : 'Enhance (Local)'
}
