import {
  deleteAllProjectEntries,
  deleteProjectIdsEntry,
  hasLegacyProjectsEntry,
  readLegacyProjects,
  readProjectsFromReferences,
  writeLegacyProjects,
} from '../hooks/useProjectReferencesMigration'

export type RollbackProjectReferencesMigrationOptions = {
  reload?: boolean
}

export type RollbackProjectReferencesMigrationResult =
  | {
      status: 'alreadyLegacy'
      projectCount: number
    }
  | {
      status: 'rolledBack'
      projectCount: number
      projectIds: string[]
    }

function rollbackProjectReferencesMigration(
  options: RollbackProjectReferencesMigrationOptions = {},
): RollbackProjectReferencesMigrationResult {
  if (hasLegacyProjectsEntry()) {
    const legacyProjects = readLegacyProjects()
    if (options.reload) {
      window.location.reload()
    }
    return {
      status: 'alreadyLegacy',
      projectCount: legacyProjects.length,
    }
  }

  const projects = readProjectsFromReferences()
  const restoredProjects = writeLegacyProjects(projects)
  deleteProjectIdsEntry()
  deleteAllProjectEntries()

  if (options.reload) {
    window.location.reload()
  }

  return {
    status: 'rolledBack',
    projectCount: restoredProjects.length,
    projectIds: restoredProjects.map(project => project.id),
  }
}

export function installProjectStorageDevtools(): void {
  if (!import.meta.env.DEV) return

  window.__ltxProjectStorageDebug = {
    rollbackProjectReferencesMigration,
  }
}
