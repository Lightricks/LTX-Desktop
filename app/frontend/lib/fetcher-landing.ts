import { isFetcherToolId, type FetcherToolId } from '../../shared/deep-link.ts'
import { isIcLoraRecipeId } from './ic-lora-recipes.ts'
import { getLoraRecipe, isLoraRecipeId } from './lora-recipes.ts'
import type { HomeFeatureDefinition } from './home-features.ts'
import type { VideoGenerationOfferingCapabilities } from './video-generation-model-specs.ts'

function offeringSupportsTool(
  slug: FetcherToolId,
  capabilities: VideoGenerationOfferingCapabilities,
): boolean {
  if (isLoraRecipeId(slug)) {
    const recipe = getLoraRecipe(slug)
    const videoCap = recipe.mode === 'i2v' ? capabilities.i2v : capabilities.t2v
    return videoCap && capabilities.user_loras
  }
  if (isIcLoraRecipeId(slug)) {
    return capabilities.ic_lora
  }
  switch (slug) {
    case 't2v':
    case 'i2v':
    case 'a2v':
    case 'retake':
    case 'extend':
      return capabilities[slug]
  }
}

export type FetcherLandingDecision =
  | { status: 'resolving' }
  | { status: 'redirect'; path: string }
  | { status: 'blocked' }

export function resolveFetcherLanding(args: {
  slug: string
  isLoading: boolean
  errorMessage: string | null
  capabilities: VideoGenerationOfferingCapabilities | null
  features: readonly HomeFeatureDefinition[]
}): FetcherLandingDecision {
  if (!isFetcherToolId(args.slug)) {
    return { status: 'blocked' }
  }

  if (args.isLoading) {
    return { status: 'resolving' }
  }

  if (args.errorMessage || args.capabilities === null) {
    return { status: 'blocked' }
  }

  if (!offeringSupportsTool(args.slug, args.capabilities)) {
    return { status: 'blocked' }
  }

  const feature = args.features.find(
    (candidate) => 'fetcherTool' in candidate && candidate.fetcherTool === args.slug,
  )
  if (!feature) {
    return { status: 'blocked' }
  }

  return { status: 'redirect', path: feature.path }
}
