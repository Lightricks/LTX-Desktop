import { formatEstimatedDuration } from './duration-format.ts'
import {
  passShapeChoice,
  selectedFastRisk,
  selectedShapeAsk,
  type PassShapeSelection,
} from './pass-shape.ts'
import {
  isFastCellOffered,
  offeredFastDurations,
  VIDEO_GENERATION_RESOLUTION_ORDER,
  type JobMemoryAsk,
  type MemoryCapacity,
  type SelectionRisk,
  type VideoGenerationModelSpecsResponse,
} from './video-generation-model-specs.ts'
import { videoGenerationResolutionLabel } from './video-resolution.ts'

/** Home memory meter: one bar under the settings, drawn from the cell the backend priced. */

export type MemoryMeterKind = 'crisp' | 'soft' | 'rough'

export type MemoryMeterLine = { text: string; tone: 'neutral' | 'over' }

/** A one-click change that makes an over job fit. */
export type MemoryMeterFix = {
  label: string
  dataKey: 'passShape' | 'duration' | 'resolution'
  value: number | string
}

export type MemoryMeterModel = {
  title: string
  kind: MemoryMeterKind
  askGib: number
  capacityGib: number
  /** Where the soft edge fades out, on estimates only. */
  band: [number, number] | null
  /** Light, Medium or Heavy on the rough scale. */
  roughLevel: 0 | 1 | 2 | null
  valueText: string
  over: boolean
  /** 1 for a single pass. */
  parts: number
  line: MemoryMeterLine | null
  tooltip: string | null
  fixes: MemoryMeterFix[]
  estimatedTimeText?: string | null
}

export type MemoryMeterFixFields = readonly MemoryMeterFix['dataKey'][]

export type MemoryMeterInput = {
  fixFields?: MemoryMeterFixFields
  /**
   * The cell length a resolution would run at, when the length is not a field: audio to video
   * runs the shortest cell that covers the audio, and that cell differs per resolution.
   */
  durationAt?: (resolution: string) => number | null
}

const ROUGH_LEVELS = ['Light', 'Medium', 'Heavy'] as const
const ROUGH_MEDIUM_FROM = 0.5
const ROUGH_HEAVY_FROM = 0.85

/** One decimal, without a trailing ".0": "16 GB", "15.9 GB". */
function formatGib(value: number): string {
  return String(Number(value.toFixed(1)))
}

function roughLevel(askGib: number, capacityGib: number): 0 | 1 | 2 {
  const share = askGib / capacityGib
  if (share >= ROUGH_HEAVY_FROM) return 2
  return share >= ROUGH_MEDIUM_FROM ? 1 : 0
}

function overLine(mac: boolean, rough: boolean): string {
  if (rough) return 'Probably more than this GPU has. It may run out of memory or freeze the app.'
  return mac
    ? 'More than your Mac lets video use. It will probably run out of memory or freeze the app.'
    : 'More than this GPU has. It will probably run out of memory or freeze the app.'
}

function partsLine(parts: number): string {
  return `Made in ${parts} parts, so it takes longer. Lighting or subjects can drift a little between parts.`
}

function tightLine(mac: boolean): string {
  // On a Mac the limit is a fixed allowance, so closing apps does not help.
  return mac
    ? 'Close to the limit. It should still run.'
    : 'Close to the limit. Close other heavy apps (like games, browsers, or video editors) first.'
}

function tooltipFor(kind: MemoryMeterKind, mac: boolean, capacityGib: number): string | null {
  if (mac) {
    return `macOS lets video use up to ${formatGib(capacityGib)} GB on this Mac. Activity Monitor can show more, because macOS keeps extra cache when it has room.`
  }
  if (kind === 'rough') return 'A rough guide. This GPU has no detailed memory profile yet.'
  if (kind === 'soft') return 'Estimated from similar jobs.'
  return null
}

function fits(ask: JobMemoryAsk | null, capacityGib: number): ask is JobMemoryAsk {
  return ask != null && ask.ask_gib <= capacityGib
}

/**
 * Parts for a forced one pass that parts would fit, then the longest shorter length and the
 * sharpest lower resolution that fit, in that order.
 */
function fixesFor(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  selection: PassShapeSelection,
  capacityGib: number,
  input: MemoryMeterInput,
): MemoryMeterFix[] {
  const fields = input.fixFields ?? []
  const fixes: MemoryMeterFix[] = []
  if (fields.includes('passShape') && selection.passShape === 'one_pass') {
    const parts = passShapeChoice(specs, selection)?.parts ?? null
    if (fits(parts, capacityGib)) {
      fixes.push({ label: 'Make it in parts', dataKey: 'passShape', value: 'parts' })
    }
  }
  const offeredAndFits = (candidate: PassShapeSelection) =>
    isFastCellOffered(specs, candidate) && fits(selectedShapeAsk(specs, candidate), capacityGib)
  if (fields.includes('duration') && selection.duration != null) {
    const current = selection.duration
    const shorter = [...offeredFastDurations(specs, selection)]
      .filter((duration) => duration < current)
      .sort((left, right) => right - left)
      .find((duration) => offeredAndFits({ ...selection, duration }))
    if (shorter != null) {
      fixes.push({ label: `Shorten to ${shorter} s`, dataKey: 'duration', value: shorter })
    }
  }
  if (fields.includes('resolution')) {
    const order = VIDEO_GENERATION_RESOLUTION_ORDER as readonly string[]
    const index = order.indexOf(selection.resolution)
    const lower = order
      .slice(0, Math.max(0, index))
      .reverse()
      .find((resolution) => {
        const duration = input.durationAt ? input.durationAt(resolution) : selection.duration
        return duration != null && offeredAndFits({ ...selection, resolution, duration })
      })
    if (lower != null) {
      fixes.push({
        label: `Use ${videoGenerationResolutionLabel(lower)}`,
        dataKey: 'resolution',
        value: lower,
      })
    }
  }
  return fixes
}

/**
 * The meter for one priced job. Lines in order of what the user must know first: more than the
 * machine has, then a short RAM line, then the run in parts, then a single pass close to the limit.
 */
export function buildAskMeterModel(
  ask: JobMemoryAsk,
  memory: MemoryCapacity | null | undefined,
  input: { ramRisk?: SelectionRisk | null; fixes?: (capacityGib: number) => MemoryMeterFix[] } = {},
): MemoryMeterModel | null {
  const capacityGib = memory?.capacity_gib
  if (capacityGib == null || capacityGib <= 0) return null
  const mac = memory?.device === 'mac'
  const kind: MemoryMeterKind =
    ask.confidence === 'rough' ? 'rough' : ask.confidence === 'estimate' ? 'soft' : 'crisp'
  const over = ask.ask_gib > capacityGib
  const ramRisk = input.ramRisk ?? null
  const ramShort = ramRisk?.readout?.kind === 'ram'
  const parts = Math.max(1, ask.windows)
  const level = kind === 'rough' ? roughLevel(ask.ask_gib, capacityGib) : null

  let valueText: string
  switch (kind) {
    case 'crisp':
      valueText = `About ${formatGib(ask.ask_gib)} GB of ${formatGib(capacityGib)} GB`
      break
    case 'soft':
      valueText = `Around ${Math.round(ask.ask_gib)} GB of ${Math.round(capacityGib)} GB`
      break
    case 'rough':
      valueText = `${ROUGH_LEVELS[level ?? 0]} for this ${mac ? 'Mac' : 'GPU'}`
      break
    default: {
      const unhandled: never = kind
      return unhandled
    }
  }

  let line: MemoryMeterLine | null = null
  if (over) {
    line = { text: overLine(mac, kind === 'rough'), tone: 'over' }
  } else if (ramShort && ramRisk != null) {
    line = { text: ramRisk.message, tone: 'over' }
  } else if (parts > 1) {
    line = { text: partsLine(parts), tone: 'neutral' }
  } else if (ask.tight) {
    line = { text: tightLine(mac), tone: 'neutral' }
  }

  const estimatedTimeText = formatEstimatedDuration(ask.estimated_seconds)

  return {
    title: meterTitle(mac),
    kind,
    askGib: ask.ask_gib,
    capacityGib,
    band:
      kind === 'soft' && ask.band_low_gib != null && ask.band_high_gib != null
        ? [ask.band_low_gib, ask.band_high_gib]
        : null,
    roughLevel: level,
    valueText,
    over: over || ramShort,
    parts,
    line,
    tooltip: tooltipFor(kind, mac, capacityGib),
    fixes: over && input.fixes != null ? input.fixes(capacityGib) : [],
    estimatedTimeText,
  }
}

/** The meter for a Fast selection, or null when the backend sent no ask for it (API models, an
 * unknown cell, specs still loading). */
export function buildMemoryMeterModel(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  selection: PassShapeSelection,
  input: MemoryMeterInput = {},
): MemoryMeterModel | null {
  const ask = selectedShapeAsk(specs, selection)
  if (ask == null) return null
  return buildAskMeterModel(ask, specs?.memory, {
    ramRisk: selectedFastRisk(specs, selection),
    fixes:
      input.fixFields != null
        ? (capacityGib) => fixesFor(specs, selection, capacityGib, input)
        : undefined,
  })
}

export function meterTitle(mac: boolean): string {
  return mac ? 'Memory for video' : 'GPU memory'
}

/** What one option row shows: its ask with the other choices kept, and where the bar would land. */
export type OptionMemory = {
  askGib: number
  label: string
  over: boolean
}

/**
 * The ask an option would land on. Undefined on the legacy route, which shows no numbers, and
 * where the backend sent no ask.
 */
export function optionMemory(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  selection: PassShapeSelection,
): OptionMemory | undefined {
  return askOptionMemory(selectedShapeAsk(specs, selection), specs?.memory)
}

/** An option row's memory for one priced ask. */
export function askOptionMemory(
  ask: JobMemoryAsk | null,
  memory: MemoryCapacity | null | undefined,
): OptionMemory | undefined {
  const capacityGib = memory?.capacity_gib
  if (capacityGib == null || ask == null || ask.confidence === 'rough') return undefined
  const amount =
    ask.confidence === 'estimate' ? `~${Math.round(ask.ask_gib)} GB` : `${formatGib(ask.ask_gib)} GB`
  return {
    askGib: ask.ask_gib,
    label: ask.windows > 1 ? `${amount} · ${ask.windows} parts` : amount,
    over: ask.ask_gib > capacityGib,
  }
}

/** "No number yet": which sizes fit before the input that sets the number is picked. */
export type MemoryFitsCell = {
  value: string
  label: string
  state: 'full' | 'partial' | 'none'
  /** "up to 6 s" on a partial cell. */
  note: string | null
  selected: boolean
}

export type MemoryFitsModel = {
  title: string
  cells: MemoryFitsCell[]
  line: MemoryMeterLine
}
