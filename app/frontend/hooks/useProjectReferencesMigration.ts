import { useCallback, useRef, useState } from 'react'
import { useProjects } from '../contexts/ProjectContext'
import { projectReferenceSchema, projectSchema, type Project } from '../types/project-model'
import {
  PROJECT_IDS_STORAGE_KEY,
  PROJECT_STORAGE_KEY_PREFIX,
  getProjectStorageKey,
  readProject,
  readProjectIds,
  writeProjectIds,
} from '../lib/project-storage'
import { logger } from '../lib/logger'

export type ProjectReferencesMigrationStatus =
  | { status: 'needed' }
  | { status: 'inProgress'; ratio: number }
  | { status: 'completed' }

const LEGACY_PROJECTS_STORAGE_KEY = 'ltx-projects'

interface LegacyProjectRecord {
  projectData: unknown
  projectId: string
}

function yieldToUi(): Promise<void> {
  return new Promise(resolve => {
    window.setTimeout(resolve, 0)
  })
}

export function hasLegacyProjectsEntry(): boolean {
  return localStorage.getItem(LEGACY_PROJECTS_STORAGE_KEY) !== null
}

export function deleteLegacyProjectsEntry(): void {
  localStorage.removeItem(LEGACY_PROJECTS_STORAGE_KEY)
}

export function writeLegacyProjects(projects: readonly Project[]): Project[] {
  const normalizedProjects = projects.map(project => projectSchema.parse(project))
  localStorage.setItem(
    LEGACY_PROJECTS_STORAGE_KEY,
    JSON.stringify(normalizedProjects),
  )
  return normalizedProjects
}

export function readLegacyProjects(): LegacyProjectRecord[] {
  const stored = localStorage.getItem(LEGACY_PROJECTS_STORAGE_KEY)
  if (!stored) return []

  const parsed = JSON.parse(stored)
  if (!Array.isArray(parsed)) {
    throw new Error('Legacy projects payload is not an array')
  }

  return parsed.map(projectData => ({
    projectData,
    projectId: projectReferenceSchema.parse(projectData).id,
  }))
}

export function writeRawProject(projectId: string, projectData: unknown): void {
  localStorage.setItem(
    getProjectStorageKey(projectId),
    JSON.stringify(projectData),
  )
}

export function deleteProjectIdsEntry(): void {
  localStorage.removeItem(PROJECT_IDS_STORAGE_KEY)
}

export function readProjectsFromReferences(): Project[] {
  return readProjectIds().map(projectId => {
    const project = readProject(projectId)
    if (!project) {
      throw new Error(`Missing project entry for id ${projectId}`)
    }
    return project
  })
}

export function deleteAllProjectEntries(): void {
  const keysToDelete: string[] = []
  for (let i = 0; i < localStorage.length; i += 1) {
    const storageKey = localStorage.key(i)
    if (storageKey?.startsWith(PROJECT_STORAGE_KEY_PREFIX)) {
      keysToDelete.push(storageKey)
    }
  }

  for (const storageKey of keysToDelete) {
    localStorage.removeItem(storageKey)
  }
}

export function useProjectReferencesMigration() {
  const { reloadProjectIds } = useProjects()
  const [migrationStatus, setMigrationStatus] = useState<ProjectReferencesMigrationStatus>(() => (
    hasLegacyProjectsEntry() ? { status: 'needed' } : { status: 'completed' }
  ))
  const inFlightRef = useRef<Promise<void> | null>(null)

  const migrateProjects = useCallback(async () => {
    if (inFlightRef.current) return inFlightRef.current

    const runMigration = async () => {
      if (!hasLegacyProjectsEntry()) {
        reloadProjectIds()
        setMigrationStatus({ status: 'completed' })
        return
      }

      setMigrationStatus({ status: 'inProgress', ratio: 0 })
      await yieldToUi()

      try {
        const legacyProjects = readLegacyProjects()
        const total = legacyProjects.length

        for (const [index, project] of legacyProjects.entries()) {
          writeRawProject(project.projectId, project.projectData)
          setMigrationStatus({
            status: 'inProgress',
            ratio: total === 0 ? 1 : (index + 1) / total,
          })
          await yieldToUi()
        }

        writeProjectIds(legacyProjects.map(project => project.projectId))
        deleteLegacyProjectsEntry()
        reloadProjectIds()
        setMigrationStatus({ status: 'completed' })
      } catch (error) {
        logger.error(`Failed to migrate project references: ${error}`)
        setMigrationStatus({ status: 'needed' })
      }
    }

    inFlightRef.current = runMigration().finally(() => {
      inFlightRef.current = null
    })

    return inFlightRef.current
  }, [reloadProjectIds])

  return { migrationStatus, migrateProjects }
}
