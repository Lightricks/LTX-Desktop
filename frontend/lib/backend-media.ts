import { getBackendCredentials } from './backend'

export type BackendMediaType = 'image' | 'audio' | 'video'

export interface PreparedBackendMedia {
  path?: string
  mediaId?: string
}

export interface BackendArtifact {
  artifact_id: string
  media_type: BackendMediaType
  filename: string
  content_type: string
  size_bytes: number
  sha256: string
  expires_at: string
}

export async function prepareBackendMedia(
  filePath: string | null | undefined,
  mediaType: BackendMediaType,
): Promise<PreparedBackendMedia | null> {
  if (!filePath) return null
  const backend = await getBackendCredentials()
  if (backend.mode === 'managed-local') return { path: filePath }

  const result = await window.electronAPI.uploadBackendMedia({ filePath, mediaType })
  if (!result.success) throw new Error(result.error)
  return { mediaId: result.media.media_id }
}

export function toBackendArtifact(value: unknown): BackendArtifact | null {
  if (!value || typeof value !== 'object') return null
  const record = value as Partial<BackendArtifact>
  if (
    typeof record.artifact_id !== 'string'
    || (record.media_type !== 'image' && record.media_type !== 'audio' && record.media_type !== 'video')
    || typeof record.filename !== 'string'
    || typeof record.content_type !== 'string'
    || typeof record.size_bytes !== 'number'
    || typeof record.sha256 !== 'string'
    || typeof record.expires_at !== 'string'
  ) {
    return null
  }
  return record as BackendArtifact
}

export async function materializeBackendOutput(
  legacyPath: string | null | undefined,
  artifactValue: unknown,
): Promise<string> {
  const backend = await getBackendCredentials()
  if (backend.mode === 'managed-local' && legacyPath) return legacyPath

  const artifact = toBackendArtifact(artifactValue)
  if (!artifact) {
    throw new Error(backend.mode === 'external'
      ? 'The remote backend completed without a downloadable artifact.'
      : 'Generation completed without an output path or artifact.')
  }
  const result = await window.electronAPI.materializeBackendArtifact({ artifact })
  if (!result.success) throw new Error(result.error)
  return result.path
}

export async function materializeBackendOutputs(
  legacyPaths: readonly string[] | null | undefined,
  artifactValues: unknown,
): Promise<string[]> {
  const backend = await getBackendCredentials()
  if (backend.mode === 'managed-local' && legacyPaths && legacyPaths.length > 0) {
    return [...legacyPaths]
  }
  if (!Array.isArray(artifactValues) || artifactValues.length === 0) {
    throw new Error('The remote backend completed without downloadable artifacts.')
  }
  return Promise.all(artifactValues.map((artifact) => materializeBackendOutput(null, artifact)))
}
