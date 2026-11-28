import type { EffectType, TextOverlayStyle } from './project-model'

export interface EffectParamDef {
  min: number
  max: number
  step: number
  label: string
}

export interface EffectDefinition {
  name: string
  category: 'filter' | 'stylize' | 'color-preset'
  icon: string
  defaultParams: Record<string, number>
  paramRanges: Record<string, EffectParamDef>
}

export const EFFECT_DEFINITIONS: Record<EffectType, EffectDefinition> = {
  'blur': {
    name: 'Gaussian Blur',
    category: 'filter',
    icon: 'Droplets',
    defaultParams: { amount: 5 },
    paramRanges: { amount: { min: 0, max: 50, step: 0.5, label: 'Radius' } },
  },
  'sharpen': {
    name: 'Sharpen',
    category: 'filter',
    icon: 'Diamond',
    defaultParams: { amount: 50 },
    paramRanges: { amount: { min: 0, max: 100, step: 1, label: 'Amount' } },
  },
  'glow': {
    name: 'Glow',
    category: 'stylize',
    icon: 'Sun',
    defaultParams: { amount: 30, radius: 10 },
    paramRanges: {
      amount: { min: 0, max: 100, step: 1, label: 'Intensity' },
      radius: { min: 0, max: 50, step: 1, label: 'Radius' },
    },
  },
  'vignette': {
    name: 'Vignette',
    category: 'stylize',
    icon: 'Circle',
    defaultParams: { amount: 50 },
    paramRanges: { amount: { min: 0, max: 100, step: 1, label: 'Amount' } },
  },
  'grain': {
    name: 'Film Grain',
    category: 'stylize',
    icon: 'Scan',
    defaultParams: { amount: 30 },
    paramRanges: { amount: { min: 0, max: 100, step: 1, label: 'Amount' } },
  },
  'lut-cinematic': {
    name: 'Cinematic',
    category: 'color-preset',
    icon: 'Film',
    defaultParams: { intensity: 100 },
    paramRanges: { intensity: { min: 0, max: 100, step: 1, label: 'Intensity' } },
  },
  'lut-vintage': {
    name: 'Vintage',
    category: 'color-preset',
    icon: 'Clock',
    defaultParams: { intensity: 100 },
    paramRanges: { intensity: { min: 0, max: 100, step: 1, label: 'Intensity' } },
  },
  'lut-bw': {
    name: 'Black & White',
    category: 'color-preset',
    icon: 'Contrast',
    defaultParams: { intensity: 100 },
    paramRanges: { intensity: { min: 0, max: 100, step: 1, label: 'Intensity' } },
  },
  'lut-cool': {
    name: 'Cool Tone',
    category: 'color-preset',
    icon: 'Snowflake',
    defaultParams: { intensity: 100 },
    paramRanges: { intensity: { min: 0, max: 100, step: 1, label: 'Intensity' } },
  },
  'lut-warm': {
    name: 'Warm Tone',
    category: 'color-preset',
    icon: 'Flame',
    defaultParams: { intensity: 100 },
    paramRanges: { intensity: { min: 0, max: 100, step: 1, label: 'Intensity' } },
  },
  'lut-muted': {
    name: 'Muted',
    category: 'color-preset',
    icon: 'CloudFog',
    defaultParams: { intensity: 100 },
    paramRanges: { intensity: { min: 0, max: 100, step: 1, label: 'Intensity' } },
  },
  'lut-vivid': {
    name: 'Vivid',
    category: 'color-preset',
    icon: 'Palette',
    defaultParams: { intensity: 100 },
    paramRanges: { intensity: { min: 0, max: 100, step: 1, label: 'Intensity' } },
  },
}

export interface TextPreset {
  id: string
  name: string
  category: 'titles' | 'lower-thirds' | 'captions' | 'end-cards'
  style: Partial<TextOverlayStyle>
}

export const TEXT_PRESETS: TextPreset[] = [
  { id: 'centered-title', name: 'Centered Title', category: 'titles', style: { text: 'Title', fontSize: 72, fontWeight: 'bold', positionX: 50, positionY: 50, textAlign: 'center' } },
  { id: 'big-bold', name: 'Big & Bold', category: 'titles', style: { text: 'HEADLINE', fontSize: 96, fontWeight: '900', positionX: 50, positionY: 45, textAlign: 'center', letterSpacing: 4 } },
  { id: 'subtitle-style', name: 'Subtitle', category: 'captions', style: { text: 'Subtitle text', fontSize: 36, fontWeight: 'normal', positionX: 50, positionY: 88, textAlign: 'center', backgroundColor: 'rgba(0,0,0,0.6)', padding: 8, borderRadius: 4 } },
  { id: 'lower-third-basic', name: 'Lower Third', category: 'lower-thirds', style: { text: 'Name Here', fontSize: 32, fontWeight: '600', positionX: 10, positionY: 82, textAlign: 'left', backgroundColor: 'rgba(0,0,0,0.7)', padding: 12, borderRadius: 6, maxWidth: 40 } },
  { id: 'lower-third-accent', name: 'Accent Lower Third', category: 'lower-thirds', style: { text: 'Speaker Name', fontSize: 28, fontWeight: '500', positionX: 8, positionY: 85, textAlign: 'left', color: '#FFFFFF', backgroundColor: 'rgba(124,58,237,0.85)', padding: 10, borderRadius: 4, maxWidth: 35 } },
  { id: 'end-card', name: 'End Card', category: 'end-cards', style: { text: 'Thank You', fontSize: 80, fontWeight: '300', positionX: 50, positionY: 45, textAlign: 'center', letterSpacing: 8, color: '#E4E4E7' } },
  { id: 'corner-tag', name: 'Corner Tag', category: 'captions', style: { text: 'LIVE', fontSize: 20, fontWeight: '700', positionX: 92, positionY: 8, textAlign: 'right', color: '#FFFFFF', backgroundColor: 'rgba(239,68,68,0.9)', padding: 6, borderRadius: 4 } },
]
