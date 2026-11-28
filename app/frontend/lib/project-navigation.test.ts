import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { writeProject } from './project-storage.ts'
import { createDefaultTimeline, normalizeProject } from '../types/project-model.ts'
import { decideProjectRoute, openProject, projectLoader } from './project-navigation.ts'

describe('openProject', () => {
  it('activates, resets the tab, then navigates — in that order', () => {
    installMemoryLocalStorage()
    writeProject('project-1', normalizeProject({
      id: 'project-1',
      name: 'One',
      createdAt: 1,
      updatedAt: 1,
      assets: [],
      timelines: [createDefaultTimeline()],
    }))
    const calls: string[] = []
    openProject('project-1', {
      activateProject: (id) => calls.push(`activate:${id}`),
      setCurrentTab: (tab) => calls.push(`tab:${tab}`),
      navigate: (to) => calls.push(`nav:${to}`),
    })
    assert.deepEqual(calls, [
      'activate:project-1',
      'tab:gen-space',
      'nav:/projects/project-1',
    ])
  })
})

describe('decideProjectRoute', () => {
  it('goes to the project list when the id is missing or the project does not exist', () => {
    assert.equal(
      decideProjectRoute({ projectId: undefined, activeProjectId: 'p1', projectExists: false }),
      'projects',
    )
    assert.equal(
      decideProjectRoute({ projectId: 'missing', activeProjectId: 'p1', projectExists: false }),
      'projects',
    )
  })

  it('activates when the URL id exists but is not the loaded document', () => {
    assert.equal(
      decideProjectRoute({ projectId: 'p2', activeProjectId: 'p1', projectExists: true }),
      'activate',
    )
    assert.equal(
      decideProjectRoute({ projectId: 'p2', activeProjectId: null, projectExists: true }),
      'activate',
    )
  })

  it('is ready when the loaded document already matches the URL', () => {
    assert.equal(
      decideProjectRoute({ projectId: 'p1', activeProjectId: 'p1', projectExists: true }),
      'ready',
    )
  })
})

function loaderArgs(projectId?: string) {
  return {
    params: projectId ? { projectId } : {},
    request: new Request('https://desktop.local/'),
  }
}

function locationOf(result: unknown): string | null {
  return result instanceof Response ? result.headers.get('Location') : null
}

function installMemoryLocalStorage() {
  const store = new Map<string, string>()
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: {
      getItem: (key: string) => store.get(key) ?? null,
      setItem: (key: string, value: string) => { store.set(key, value) },
      removeItem: (key: string) => { store.delete(key) },
      clear: () => store.clear(),
    },
  })
}

describe('projectLoader', () => {
  it('redirects to the project list when the id is missing or the project is not stored', () => {
    installMemoryLocalStorage()
    const missingId = projectLoader(loaderArgs())
    const missingProject = projectLoader(loaderArgs('missing'))
    assert.equal(locationOf(missingId), '/projects')
    assert.equal(locationOf(missingProject), '/projects')
  })

  it('lets the route render when the project is stored', () => {
    installMemoryLocalStorage()
    writeProject('p1', normalizeProject({
      id: 'p1',
      name: 'One',
      createdAt: 1,
      updatedAt: 1,
      assets: [],
      timelines: [createDefaultTimeline()],
    }))
    assert.equal(projectLoader(loaderArgs('p1')), null)
  })
})
