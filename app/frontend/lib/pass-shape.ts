import { closestVideoAspectRatio } from './closest-video-aspect-ratio.ts'
import {
  fastCellAsk,
  fastSelectionRisk,
  getOfferingVideoGenerationModelSpecs,
  isVideoGenerationAspectRatio,
  type FastCellSelection,
  type JobMemoryAsk,
  type SelectionRisk,
  type VideoGenerationModelSpecsResponse,
} from './video-generation-model-specs.ts'

/** How a Fast clip is cut. `auto` lets the backend pick: one pass when it fits, otherwise parts. */
export type PassShape = 'auto' | 'one_pass' | 'parts'

/** The form's "Make it" value. Null is the automatic pick. */
export type ForcedPassShape = Exclude<PassShape, 'auto'>

export type PassShapeSelection = FastCellSelection & { passShape?: ForcedPassShape | null }

/** Both shapes a Brave user can choose between for one clip. */
export type PassShapeChoice = {
  automatic: ForcedPassShape
  onePass: JobMemoryAsk
  parts: JobMemoryAsk
}

export function isForcedPassShape(value: unknown): value is ForcedPassShape {
  return value === 'one_pass' || value === 'parts'
}

/**
 * Both shapes and their asks, or null wherever the "Make it" toggle stays hidden. The backend
 * sends the other shape only where the toggle shows (Brave, below 1440p, past one part, not
 * behind a hard wall, and not a one pass inside the measured size); the mode and the switch are
 * checked here too, so specs from before a settings change never offer a choice.
 */
export function passShapeChoice(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  selection: FastCellSelection,
): PassShapeChoice | null {
  if (specs?.budgetMode !== 'brave' || specs.neverSplitIntoParts) return null
  const cell = fastCellAsk(specs, selection)
  const alternative = cell?.alternative
  if (cell == null || alternative == null) return null
  return cell.windows > 1
    ? { automatic: 'parts', onePass: alternative, parts: cell }
    : { automatic: 'one_pass', onePass: cell, parts: alternative }
}

/** The shape create sends: the forced one only while the toggle shows it. */
export function effectivePassShape(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  selection: PassShapeSelection,
): PassShape {
  const forced = selection.passShape
  if (forced == null || passShapeChoice(specs, selection) == null) return 'auto'
  return forced
}

/** What the job asks in the shape it will run. Null while the cell is unknown. */
export function selectedShapeAsk(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  selection: PassShapeSelection,
): JobMemoryAsk | null {
  const cell = fastCellAsk(specs, selection)
  const forced = selection.passShape
  if (cell == null || forced == null) return cell
  const choice = passShapeChoice(specs, selection)
  if (choice == null) return cell
  return forced === 'one_pass' ? choice.onePass : choice.parts
}

/** The budget warning for the shape the job will run, or null when it fits cleanly. */
export function selectedFastRisk(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  selection: PassShapeSelection,
): SelectionRisk | null {
  const choice = passShapeChoice(specs, selection)
  if (choice != null && selection.passShape != null && selection.passShape !== choice.automatic) {
    return (selection.passShape === 'one_pass' ? choice.onePass.risk : choice.parts.risk) ?? null
  }
  return fastSelectionRisk(specs, selection)
}

/**
 * Drops a forced shape the toggle no longer shows for the priced cell. While the cell is unknown
 * (null: the audio's length or the start image's size is still loading) the choice is kept, so a
 * restored shape survives; create sends `auto` until the cell is known.
 */
export function withVisiblePassShape<TValues extends { passShape?: ForcedPassShape | null }>(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  values: TValues,
  cell: PassShapeSelection | null,
): TValues {
  if (values.passShape == null || cell == null || effectivePassShape(specs, cell) !== 'auto') {
    return values
  }
  return { ...values, passShape: null }
}

export type ImageSize = { width: number; height: number }

/**
 * The cell a job is priced at. Auto runs at the start image's closest advertised ratio for the
 * resolution, the same pick the backend makes, so it is null until the image's size is known.
 */
export function pricedCell<TSelection extends PassShapeSelection>(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  selection: TSelection,
  startFrameSize: ImageSize | null | undefined,
): TSelection | null {
  if (selection.aspectRatio !== 'auto') return selection
  if (startFrameSize == null) return null
  const row = getOfferingVideoGenerationModelSpecs(specs).find((item) => item.model === selection.model)
  const ratios = (row?.spec.supported_resolutions_durations[selection.resolution]?.aspect_ratios ?? [])
    .filter(isVideoGenerationAspectRatio)
  const aspectRatio = closestVideoAspectRatio(startFrameSize.width, startFrameSize.height, ratios)
  return aspectRatio == null ? null : { ...selection, aspectRatio }
}

/** The shape create sends: `auto` while the priced cell is unknown or its toggle is hidden. */
export function passShapeToSend(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  selection: PassShapeSelection,
  startFrameSize: ImageSize | null | undefined,
): PassShape {
  const cell = pricedCell(specs, selection, startFrameSize)
  return cell == null ? 'auto' : effectivePassShape(specs, cell)
}

/** What the meter and option rows price: the cell, or the automatic pick while it is unknown. */
export function meterCell<TSelection extends PassShapeSelection>(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  selection: TSelection,
  startFrameSize: ImageSize | null | undefined,
): TSelection {
  return pricedCell(specs, selection, startFrameSize) ?? { ...selection, passShape: null }
}
