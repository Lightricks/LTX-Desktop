export type GenSpaceMode =
  | 'image'
  | 'video'
  | 'multi-keyframe'
  | 'retake'
  | 'extend'
  | 'ic-lora'

export function autoDurationOptionVisible(
  mode: GenSpaceMode,
  autoDurationAvailable: boolean,
): boolean {
  return mode !== 'multi-keyframe' && autoDurationAvailable
}

type MultiKeyframeCapabilities = {
  multi_keyframe?: boolean
}

type ModeAvailability = {
  canUseMultiKeyframe: boolean
  canUseRetake: boolean
  canUseExtend: boolean
  canUseIcLora: boolean
}

export function canUseMultiKeyframeMode({
  isLocalMode,
  localCaps,
}: {
  isLocalMode: boolean
  localCaps?: MultiKeyframeCapabilities | null
}): boolean {
  return isLocalMode && Boolean(localCaps?.multi_keyframe)
}

export function fallbackGenSpaceMode(
  mode: GenSpaceMode,
  availability: ModeAvailability,
): GenSpaceMode {
  if (mode === 'multi-keyframe' && !availability.canUseMultiKeyframe) return 'video'
  if (mode === 'retake' && !availability.canUseRetake) return 'video'
  if (mode === 'extend' && !availability.canUseExtend) return 'video'
  if (mode === 'ic-lora' && !availability.canUseIcLora) return 'video'
  return mode
}

export function modeOptionValues({
  canUseMultiKeyframe,
  canUseRetake,
  canUseExtend,
  canUseIcLora,
}: ModeAvailability): GenSpaceMode[] {
  return [
    'image',
    'video',
    ...(canUseMultiKeyframe ? (['multi-keyframe'] as const) : []),
    ...(canUseRetake ? (['retake'] as const) : []),
    ...(canUseExtend ? (['extend'] as const) : []),
    ...(canUseIcLora ? (['ic-lora'] as const) : []),
  ]
}

export function isGenSpaceLibraryMode(mode: GenSpaceMode): boolean {
  return mode === 'video' || mode === 'image'
}

export function modeAfterCompletedGeneration(generationMode: string): GenSpaceMode | null {
  return generationMode === 'multi-keyframe' ? 'video' : null
}

export function isEnhanceAvailableForMode(mode: GenSpaceMode): boolean {
  return mode === 'video' || mode === 'ic-lora' || mode === 'image' || mode === 'multi-keyframe'
}

/** Multi-keyframe cannot combine with A2V — leftover audio must not pick A2V envelopes or go on the wire. */
export function genSpaceUsesAudioInput(mode: GenSpaceMode): boolean {
  return mode === 'video'
}
