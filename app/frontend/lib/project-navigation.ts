import { generatePath, redirect, type LoaderFunctionArgs } from 'react-router'
import { paths } from '../paths.ts'
import { recordRecentProjectId } from './home-recent-projects-preference.ts'
import { readProject } from './project-storage.ts'

export function openProject(
  projectId: string,
  deps: {
    activateProject: (id: string) => void
    setCurrentTab: (tab: 'gen-space' | 'video-editor') => void
    navigate: (to: string) => void
  },
) {
  recordRecentProjectId(projectId)
  deps.activateProject(projectId)
  deps.setCurrentTab('gen-space')
  deps.navigate(generatePath(paths.project, { projectId }))
}

export type ProjectRouteDecision = 'activate' | 'projects' | 'ready'

export function decideProjectRoute(input: {
  projectId: string | undefined
  activeProjectId: string | null
  projectExists: boolean
}): ProjectRouteDecision {
  if (!input.projectId || !input.projectExists) return 'projects'
  if (input.activeProjectId === input.projectId) return 'ready'
  return 'activate'
}

/** Missing or unknown id → project list. Loading the document stays in `Project`. */
export function projectLoader({ params }: LoaderFunctionArgs) {
  const projectId = params.projectId
  if (!projectId || !readProject(projectId)) {
    return redirect(paths.projects)
  }
  return null
}
