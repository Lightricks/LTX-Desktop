// Project and Asset types for LTX Desktop

// Parameters needed to regenerate a shot
export interface GenerationParams {
  mode: 'text-to-video' | 'image-to-video' | 'audio-to-video' | 'text-to-image' | 'retake' | 'ic-lora'
  prompt: string
  model: string
  duration: number
  resolution: string
  fps: number
  audio: boolean
  cameraMotion: string
  imageAspectRatio?: string
  imageSteps?: number
  inputImageUrl?: string // For I2V: the input image used
  inputAudioUrl?: string // For A2V: the input audio used
  retakeVideoPath?: string
  retakeStartTime?: number
  retakeDuration?: number
  retakeMode?: string
  icLoraVideoPath?: string
  icLoraConditioningType?: string
  icLoraConditioningStrength?: number
}

// A single "take" (version) of a generated asset
export interface AssetTake {
  path: string
  bigThumbnailPath?: string
  smallThumbnailPath?: string
  width?: number
  height?: number
  createdAt: number
}

export interface Asset {
  id: string
  type: 'image' | 'video' | 'audio' | 'adjustment'
  path: string
  bigThumbnailPath?: string
  smallThumbnailPath?: string
  width?: number
  height?: number
  prompt: string
  resolution: string
  duration?: number // For videos
  createdAt: number
  favorite?: boolean
  bin?: string // Bin/folder name for organization (undefined = no bin)
  // Regeneration support
  generationParams?: GenerationParams
  takes?: AssetTake[] // All takes (index 0 = original). If undefined, the asset itself is the only take.
  activeTakeIndex?: number // Which take is currently active (default = 0 / latest)
  colorLabel?: string // Color label for organization (e.g. 'violet', 'blue', 'green', 'yellow', 'red', 'rose', 'orange', 'mango')
}

export interface Track {
  id: string
  name: string
  muted: boolean
  locked: boolean
  solo?: boolean                 // Audio solo: when any track is soloed, only soloed tracks produce audio
  enabled?: boolean              // Track output toggle: false = clips on this track hidden in preview (default true)
  sourcePatched?: boolean        // Source/record patch: false = insert/overwrite edits skip this track (default true)
  type?: 'default' | 'subtitle'  // default = media track, subtitle = subtitle track
  kind?: 'video' | 'audio'      // NLE track kind for display ordering (video tracks stack up, audio down)
  subtitleStyle?: Partial<SubtitleStyle>  // Global style for all subtitles on this track (overrides DEFAULT, overridden by per-sub style)
}

// Subtitle styling options
export interface SubtitleStyle {
  fontSize: number         // in px, default 32
  fontFamily: string       // default 'sans-serif'
  fontWeight: 'normal' | 'bold'
  color: string            // hex color, default '#FFFFFF'
  backgroundColor: string  // hex color with alpha, default '#00000099'
  position: 'bottom' | 'top' | 'center'  // vertical position, default 'bottom'
  italic: boolean
}

export const DEFAULT_SUBTITLE_STYLE: SubtitleStyle = {
  fontSize: 32,
  fontFamily: 'sans-serif',
  fontWeight: 'normal',
  color: '#FFFFFF',
  backgroundColor: 'transparent',
  position: 'bottom',
  italic: false,
}

// A single subtitle cue on the timeline
export interface SubtitleClip {
  id: string
  text: string
  startTime: number
  endTime: number
  trackIndex: number
  style?: Partial<SubtitleStyle>
}

export type TransitionType = 'none' | 'dissolve' | 'fade-to-black' | 'fade-to-white' | 'wipe-left' | 'wipe-right' | 'wipe-up' | 'wipe-down'

export interface ClipTransition {
  type: TransitionType
  duration: number // in seconds
}

export interface ColorCorrection {
  brightness: number   // -100 to 100, default 0
  contrast: number     // -100 to 100, default 0
  saturation: number   // -100 to 100, default 0
  temperature: number  // -100 to 100, default 0 (negative = cooler/blue, positive = warmer/orange)
  tint: number         // -100 to 100, default 0 (negative = green, positive = magenta)
  exposure: number     // -100 to 100, default 0
  highlights: number   // -100 to 100, default 0
  shadows: number      // -100 to 100, default 0
}

export const DEFAULT_COLOR_CORRECTION: ColorCorrection = {
  brightness: 0,
  contrast: 0,
  saturation: 0,
  temperature: 0,
  tint: 0,
  exposure: 0,
  highlights: 0,
  shadows: 0,
}

// Letterbox overlay settings for adjustment layers
export interface LetterboxSettings {
  enabled: boolean
  aspectRatio: '2.35:1' | '2.39:1' | '2.76:1' | '1.85:1' | '4:3' | 'custom'
  customRatio?: number  // width / height, used when aspectRatio === 'custom'
  color: string          // default '#000000'
  opacity: number        // 0-100, default 100
}

export const DEFAULT_LETTERBOX: LetterboxSettings = {
  enabled: false,
  aspectRatio: '2.35:1',
  color: '#000000',
  opacity: 100,
}

// --- Effects System ---

export type EffectType =
  | 'blur' | 'sharpen' | 'glow' | 'vignette' | 'grain'
  | 'lut-cinematic' | 'lut-vintage' | 'lut-bw' | 'lut-cool' | 'lut-warm' | 'lut-muted' | 'lut-vivid'

export type EffectMaskShape = 'rectangle' | 'ellipse'

export interface EffectMask {
  enabled: boolean
  shape: EffectMaskShape
  x: number      // center X as % of frame (0-100)
  y: number      // center Y as % of frame (0-100)
  width: number   // width as % of frame (0-100)
  height: number  // height as % of frame (0-100)
  feather: number // edge softness in px (0-100)
  invert: boolean // if true, effect applies OUTSIDE the mask
  rotation: number // rotation in degrees
}

export const DEFAULT_EFFECT_MASK: EffectMask = {
  enabled: false,
  shape: 'ellipse',
  x: 50,
  y: 50,
  width: 40,
  height: 40,
  feather: 20,
  invert: false,
  rotation: 0,
}

export interface ClipEffect {
  id: string
  type: EffectType
  enabled: boolean
  params: Record<string, number>
  mask?: EffectMask
}

export interface EffectParamDef {
  min: number
  max: number
  step: number
  label: string
}

export interface EffectDefinition {
  name: string
  category: 'filter' | 'stylize' | 'color-preset'
  icon: string // Lucide icon name
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

// Text overlay styling
export interface TextOverlayStyle {
  text: string
  fontFamily: string       // e.g. 'Inter', 'Arial', 'Georgia'
  fontSize: number         // in px, relative to 1080p canvas
  fontWeight: 'normal' | 'bold' | '100' | '200' | '300' | '400' | '500' | '600' | '700' | '800' | '900'
  fontStyle: 'normal' | 'italic'
  color: string            // hex color
  backgroundColor: string  // hex + alpha, 'transparent' for none
  textAlign: 'left' | 'center' | 'right'
  // Position as percentage of frame (0-100)
  positionX: number        // 50 = centered horizontally
  positionY: number        // 50 = centered vertically
  // Optional styling
  strokeColor: string      // outline color, 'transparent' for none
  strokeWidth: number      // outline width in px
  shadowColor: string      // text shadow
  shadowBlur: number
  shadowOffsetX: number
  shadowOffsetY: number
  letterSpacing: number    // in px
  lineHeight: number       // multiplier, e.g. 1.2
  maxWidth: number         // max width as percentage of frame (0-100), 0 = no limit
  padding: number          // padding inside background in px
  borderRadius: number     // for background box
  opacity: number          // 0-100
}

export const DEFAULT_TEXT_STYLE: TextOverlayStyle = {
  text: 'Title Text',
  fontFamily: 'Inter, Arial, sans-serif',
  fontSize: 64,
  fontWeight: 'bold',
  fontStyle: 'normal',
  color: '#FFFFFF',
  backgroundColor: 'transparent',
  textAlign: 'center',
  positionX: 50,
  positionY: 50,
  strokeColor: 'transparent',
  strokeWidth: 0,
  shadowColor: 'rgba(0,0,0,0.5)',
  shadowBlur: 4,
  shadowOffsetX: 2,
  shadowOffsetY: 2,
  letterSpacing: 0,
  lineHeight: 1.2,
  maxWidth: 80,
  padding: 0,
  borderRadius: 0,
  opacity: 100,
}

// Text overlay preset templates
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

export interface TimelineClip {
  id: string
  assetId: string | null
  type: 'video' | 'image' | 'audio' | 'adjustment' | 'text'
  startTime: number
  duration: number
  trimStart: number
  trimEnd: number
  speed: number
  reversed: boolean
  muted: boolean
  volume: number
  trackIndex: number
  asset: Asset | null
  importedName?: string
  // Effects
  flipH: boolean
  flipV: boolean
  transitionIn: ClipTransition
  transitionOut: ClipTransition
  colorCorrection: ColorCorrection
  opacity: number // 0 to 100, default 100
  // Take management
  takeIndex?: number // Which take to show (overrides asset.activeTakeIndex). undefined = use latest.
  isRegenerating?: boolean // Visual flag: true while a regeneration is in progress for this clip
  // Linked audio/video
  linkedClipIds?: string[] // If set, this clip is linked to other clips (e.g. video ↔ audio pairs). Moving/deleting one affects all linked clips.
  colorLabel?: string // Color label override (if set, uses this; otherwise inherits from asset)
  // Applied effects (blur, sharpen, LUTs, etc.)
  effects?: ClipEffect[]
  // Adjustment layer effects
  letterbox?: LetterboxSettings
  // Text overlay
  textStyle?: TextOverlayStyle
}

export interface Timeline {
  id: string
  name: string
  createdAt: number
  tracks: Track[]
  clips: TimelineClip[]
  subtitles?: SubtitleClip[]  // Subtitle cues on subtitle tracks
}

export interface Project {
  id: string
  name: string
  createdAt: number
  updatedAt: number
  assets: Asset[]
  timelines: Timeline[]
  activeTimelineId?: string
}

export type ViewType = 'home' | 'project'
export type ProjectTab = 'gen-space' | 'video-editor'

// Default tracks for new timelines
export const DEFAULT_TRACKS: Track[] = [
  { id: 'track-v1', name: 'V1', muted: false, locked: false, sourcePatched: true,  kind: 'video' },
  { id: 'track-v2', name: 'V2', muted: false, locked: false, sourcePatched: false, kind: 'video' },
  { id: 'track-v3', name: 'V3', muted: false, locked: false, sourcePatched: false, kind: 'video' },
  { id: 'track-a1', name: 'A1', muted: false, locked: false, sourcePatched: true,  kind: 'audio' },
  { id: 'track-a2', name: 'A2', muted: false, locked: false, sourcePatched: false, kind: 'audio' },
]

// Helper to create a new timeline with default tracks
export function createDefaultTimeline(name: string = 'Timeline 1'): Timeline {
  return {
    id: `timeline-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    name,
    createdAt: Date.now(),
    tracks: DEFAULT_TRACKS.map(t => ({ ...t })),
    clips: [],
  }
}
