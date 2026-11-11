import { useEffect, useState } from 'react'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { ApiClient } from '../lib/api-client'
import {
  withoutBetCells,
  type VideoGenerationModelSpecsResponse,
} from '../lib/video-generation-model-specs'

interface VideoGenerationModelSpecsState {
  modelSpecs: VideoGenerationModelSpecsResponse | null
  isLoading: boolean
  errorMessage: string | null
}

/** GenSpace and the editor have no risk confirm, so their grid never offers a bet. */
export function useVideoGenerationModelSpecs(): VideoGenerationModelSpecsState {
  // Local specs describe the *active* LTX version, so they go stale as soon as the user
  // downloads, deletes, or activates a version — refetch on the models-changed signal.
  const { modelsVersion, settings } = useAppSettings()
  const [state, setState] = useState<VideoGenerationModelSpecsState>({
    modelSpecs: null,
    isLoading: true,
    errorMessage: null,
  })

  useEffect(() => {
    const abortController = new AbortController()
    let isActive = true

    void (async () => {
      const result = await ApiClient.getGenerateVideoModelSpecs(undefined, {
        signal: abortController.signal,
      })
      if (!isActive) return

      if (result.ok) {
        setState({
          modelSpecs: withoutBetCells(result.data),
          isLoading: false,
          errorMessage: null,
        })
        return
      }

      setState({
        modelSpecs: null,
        isLoading: false,
        errorMessage: result.error.message,
      })
    })()

    return () => {
      isActive = false
      abortController.abort()
    }
  }, [settings.budgetMode, settings.neverSplitIntoParts, modelsVersion])

  return state
}
