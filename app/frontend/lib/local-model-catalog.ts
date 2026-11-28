import { H3_NAMES, H3_VARIANTS, isH3Variant } from './h3-models.ts'
import { modelGrid } from '../ltx-io/lib/amdVideoPicker.ts'
import type { VideoGenerationModelSpecsResponse } from './video-generation-model-specs.ts'

export const modelSetupLabels = {
  'ltx-2.3': 'LTX 2.3 Fast · Native',
  'ltx-2.5': 'LTX 2.5 Fast · Native',
  'comfy-int8': 'LTX 2.5 Fast · ComfyUI INT8',
  kandinsky: 'Kandinsky 6 Video Lite Distill · ComfyUI',
  ...H3_NAMES,
}
export type ModelSetupKind = keyof typeof modelSetupLabels

/** Only offer setup for the renderers advertised by the installed backend. */
export function localModelChoices(specs: VideoGenerationModelSpecsResponse): ModelSetupKind[] {
  const kinds: ModelSetupKind[] = ['ltx-2.5', 'ltx-2.3']
  if (specs.comfy_local_model) kinds.push('comfy-int8')
  if (specs.kandinsky_local_model) kinds.push('kandinsky')
  return [...kinds, ...H3_VARIANTS.filter(id => modelGrid(specs, id))]
}

export function matchesModelSearch(kind: ModelSetupKind, query: string): boolean {
  const keywords = isH3Variant(kind)
    ? kind === 'h3-ref2va' ? 'reference images' : 'keyframe'
    : kind.startsWith('ltx-') ? 'bf16' : ''
  const text = `local video renderer amd radeon rocm nvidia cuda apple metal mps ${modelSetupLabels[kind]} ${keywords}`.toLowerCase()
  return query.trim().toLowerCase().split(/\s+/).every(word => text.includes(word))
}
