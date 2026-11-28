export function shouldShowLastFrameChip(params: {
  mode: string
  hasFirstFrame: boolean
  duration: number | null
}): boolean {
  return params.mode === 'video' && params.hasFirstFrame && params.duration != null
}
