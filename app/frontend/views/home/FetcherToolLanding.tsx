import { Loader2 } from 'lucide-react'
import { Navigate, useParams } from 'react-router'
import { LocalGenerationUnsupportedNotice } from '../../components/LocalGenerationUnsupportedNotice'
import { useVideoGenerationModelSpecs } from '../../hooks/use-video-generation-model-specs'
import { resolveFetcherLanding } from '../../lib/fetcher-landing'
import { HOME_FEATURES } from '../../lib/home-features'
import { getLocalOfferingCapabilities } from '../../lib/video-generation-model-specs'

export function FetcherToolLanding() {
  const { tool } = useParams()
  const { modelSpecs, isLoading, errorMessage } = useVideoGenerationModelSpecs()
  const decision = resolveFetcherLanding({
    slug: tool ?? '',
    isLoading,
    errorMessage,
    capabilities: getLocalOfferingCapabilities(modelSpecs),
    features: HOME_FEATURES,
  })

  if (decision.status === 'resolving') {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  if (decision.status === 'redirect') {
    return <Navigate to={decision.path} replace />
  }

  return <LocalGenerationUnsupportedNotice />
}
