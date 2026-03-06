import { app, ipcMain } from 'electron'
import fs from 'fs'
import path from 'path'
import { logger } from '../logger'

const PROJECT_SCHEMA_VERSION = 1 as const

type ProjectLike = {
  id: string
  updatedAt?: number
}

type ProjectFileV1 = {
  projectSchemaVersion: 1
  project: ProjectLike & Record<string, unknown>
}

function projectsDir(): string {
  return path.join(app.getPath('userData'), 'projects')
}

function ensureProjectsDir(): string {
  const dir = projectsDir()
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true })
  }
  return dir
}

function encodeProjectId(id: string): string {
  // Deterministic and filesystem-safe.
  return encodeURIComponent(id).replace(/%/g, '_')
}

function projectPathForId(id: string): string {
  const safe = encodeProjectId(id)
  return path.join(ensureProjectsDir(), `p-${safe}.json`)
}

function writeJsonAtomic(filePath: string, jsonText: string): void {
  const dir = path.dirname(filePath)
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true })

  const tmp = `${filePath}.tmp-${process.pid}-${Date.now()}`
  const bak = `${filePath}.bak-${process.pid}-${Date.now()}`

  fs.writeFileSync(tmp, jsonText, 'utf-8')

  try {
    if (fs.existsSync(filePath)) {
      // Windows rename semantics: cannot overwrite existing file.
      fs.renameSync(filePath, bak)
    }
    fs.renameSync(tmp, filePath)
    if (fs.existsSync(bak)) fs.rmSync(bak, { force: true })
  } catch (e) {
    try {
      if (fs.existsSync(tmp)) fs.rmSync(tmp, { force: true })
    } catch {
      // ignore
    }
    try {
      // Attempt rollback if we moved original aside.
      if (!fs.existsSync(filePath) && fs.existsSync(bak)) {
        fs.renameSync(bak, filePath)
      }
    } catch {
      // ignore
    }
    throw e
  }
}

function toProjectFileV1(project: ProjectLike & Record<string, unknown>): ProjectFileV1 {
  return { projectSchemaVersion: PROJECT_SCHEMA_VERSION, project }
}

function migrateProjectFile(payload: unknown): ProjectLike & Record<string, unknown> {
  // v0: raw project object (no wrapper).
  if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
    const record = payload as Record<string, unknown>
    if (record.projectSchemaVersion === 1 && record.project && typeof record.project === 'object') {
      return record.project as ProjectLike & Record<string, unknown>
    }

    if (typeof record.id === 'string') {
      return record as ProjectLike & Record<string, unknown>
    }
  }
  throw new Error('Invalid project file payload')
}

function listProjectFiles(): string[] {
  const dir = ensureProjectsDir()
  let entries: string[] = []
  try {
    entries = fs.readdirSync(dir)
  } catch {
    return []
  }
  return entries
    .filter((name) => name.startsWith('p-') && name.endsWith('.json'))
    .map((name) => path.join(dir, name))
}

function readProjectFile(filePath: string): (ProjectLike & Record<string, unknown>) | null {
  try {
    const raw = fs.readFileSync(filePath, 'utf-8')
    const parsed = JSON.parse(raw) as unknown
    return migrateProjectFile(parsed)
  } catch (e) {
    logger.error(`[projects] Failed to read project file ${filePath}: ${e}`)
    return null
  }
}

export function registerProjectHandlers(): void {
  ipcMain.handle('projects-get-dir', () => {
    return ensureProjectsDir()
  })

  ipcMain.handle('projects-load-all', () => {
    const projects = listProjectFiles()
      .map(readProjectFile)
      .filter((p): p is ProjectLike & Record<string, unknown> => p !== null)
      .sort((a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0))

    return projects
  })

  ipcMain.handle('projects-save', (_event, project: unknown) => {
    if (!project || typeof project !== 'object') {
      throw new Error('Invalid project payload')
    }
    const record = project as Record<string, unknown>
    const id = record.id
    if (typeof id !== 'string' || !id) {
      throw new Error('Invalid project id')
    }

    const filePath = projectPathForId(id)
    writeJsonAtomic(filePath, JSON.stringify(toProjectFileV1(record as ProjectLike & Record<string, unknown>), null, 2))
    return { ok: true as const }
  })

  ipcMain.handle('projects-replace-all', (_event, projects: unknown) => {
    if (!Array.isArray(projects)) {
      throw new Error('Invalid projects payload')
    }

    const expected = new Set<string>()

    for (const item of projects) {
      if (!item || typeof item !== 'object') continue
      const record = item as Record<string, unknown>
      const id = record.id
      if (typeof id !== 'string' || !id) continue

      const fp = projectPathForId(id)
      expected.add(path.resolve(fp))
      writeJsonAtomic(fp, JSON.stringify(toProjectFileV1(record as ProjectLike & Record<string, unknown>), null, 2))
    }

    // Delete removed projects (only our managed p-*.json files).
    for (const fp of listProjectFiles()) {
      const resolved = path.resolve(fp)
      if (!expected.has(resolved)) {
        try {
          fs.rmSync(resolved, { force: true })
        } catch (e) {
          logger.warn(`[projects] Failed to delete project file ${resolved}: ${e}`)
        }
      }
    }

    return { ok: true as const }
  })

  ipcMain.handle('projects-delete', (_event, projectId: string) => {
    if (typeof projectId !== 'string' || !projectId) {
      throw new Error('Invalid project id')
    }
    const fp = projectPathForId(projectId)
    try {
      fs.rmSync(fp, { force: true })
    } catch (e) {
      logger.warn(`[projects] Failed to delete project ${projectId}: ${e}`)
    }
    return { ok: true as const }
  })
}

