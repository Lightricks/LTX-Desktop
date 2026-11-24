export const H3_VARIANTS = ['h3-fl2va', 'h3-fl2va-distill', 'h3-ref2va'] as const
export type H3Variant = typeof H3_VARIANTS[number]

export const H3_NAMES: Record<H3Variant, string> = {
  'h3-fl2va': 'MiniMax H3 · FL2VA Base',
  'h3-fl2va-distill': 'MiniMax H3 · FL2VA 8-step Turbo',
  'h3-ref2va': 'MiniMax H3 · Ref2VA',
}

export function isH3Variant(value: unknown): value is H3Variant {
  return H3_VARIANTS.some(kind => kind === value)
}
