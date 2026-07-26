import React, { createContext, useCallback, useContext, useState } from 'react'
import { hasLegacyProjectsEntry } from '../hooks/useProjectReferencesMigration'
import { createDefaultTimeline, normalizeProject, type Project, type Asset, type AssetTake, type ProjectTab } from '../types/project-model'
import {
  deleteProjectEntry,
  readProject,
  readProjectIds,
  writeProject,
  writeProjectIds,
} from '../lib/project-storage'

interface ProjectContextType {
  currentTab: ProjectTab
  setCurrentTab: (tab: ProjectTab) => void

  projectIds: string[]
  activeProject: Project | null
  getProject: (id: string) => Project | null
  setProject: (id: string, project: Project) => void
  createProject: (name: string) => Project
  deleteProject: (id: string) => void
  renameProject: (id: string, name: string) => void
  activateProject: (id: string) => void
  clearActiveProject: () => void
  reloadProjectIds: () => void

  addAsset: (projectId: string, asset: Omit<Asset, 'id' | 'createdAt'>) => Asset
  deleteAsset: (projectId: string, assetId: string) => void
  updateAsset: (projectId: string, assetId: string, updates: Partial<Asset>) => void
  addTakeToAsset: (projectId: string, assetId: string, take: AssetTake) => void
  deleteTakeFromAsset: (projectId: string, assetId: string, takeIndex: number) => void
  setAssetActiveTake: (projectId: string, assetId: string, takeIndex: number) => void
  toggleFavorite: (projectId: string, assetId: string) => void

  genSpaceEditImagePath: string | null
  setGenSpaceEditImagePath: (path: string | null) => void
  genSpaceEditMode: 'image' | 'video' | null
  setGenSpaceEditMode: (mode: 'image' | 'video' | null) => void
  genSpaceAudioPath: string | null
  setGenSpaceAudioPath: (path: string | null) => void
  genSpaceRetakeSource: GenSpaceRetakeSource | null
  setGenSpaceRetakeSource: (source: GenSpaceRetakeSource | null) => void
  pendingRetakeUpdate: PendingRetakeUpdate | null
  setPendingRetakeUpdate: (update: PendingRetakeUpdate | null) => void
  genSpaceIcLoraSource: GenSpaceIcLoraSource | null
  setGenSpaceIcLoraSource: (source: GenSpaceIcLoraSource | null) => void
  pendingIcLoraUpdate: PendingIcLoraUpdate | null
  setPendingIcLoraUpdate: (update: PendingIcLoraUpdate | null) => void
}

export interface GenSpaceRetakeSource {
  videoPath: string
  clipId?: string
  assetId?: string
  linkedClipIds?: string[]
  duration?: number
}

export interface PendingRetakeUpdate {
  assetId: string
  clipIds: string[]
  newTakeIndex: number
}

export interface GenSpaceIcLoraSource {
  videoPath: string
  clipId?: string
  assetId?: string
  linkedClipIds?: string[]
}

export interface PendingIcLoraUpdate {
  assetId: string
  clipIds: string[]
  newTakeIndex: number
}

const ProjectContext = createContext<ProjectContextType | null>(null)

function loadInitialProjectIds(): string[] {
  if (hasLegacyProjectsEntry()) return []
  return readProjectIds()
}

export function ProjectProvider({ children }: { children: React.ReactNode }) {
  const [currentTab, setCurrentTab] = useState<ProjectTab>('gen-space')
  const [projectIds, setProjectIds] = useState<string[]>(() => loadInitialProjectIds())
  const [activeProject, setActiveProject] = useState<Project | null>(null)
  const [projectRevision, setProjectRevision] = useState(0)
  const [genSpaceEditImagePath, setGenSpaceEditImagePath] = useState<string | null>(null)
  const [genSpaceEditMode, setGenSpaceEditMode] = useState<'image' | 'video' | null>(null)
  const [genSpaceAudioPath, setGenSpaceAudioPath] = useState<string | null>(null)
  const [genSpaceRetakeSource, setGenSpaceRetakeSource] = useState<GenSpaceRetakeSource | null>(null)
  const [pendingRetakeUpdate, setPendingRetakeUpdate] = useState<PendingRetakeUpdate | null>(null)
  const [genSpaceIcLoraSource, setGenSpaceIcLoraSource] = useState<GenSpaceIcLoraSource | null>(null)
  const [pendingIcLoraUpdate, setPendingIcLoraUpdate] = useState<PendingIcLoraUpdate | null>(null)

  const bumpProjectRevision = useCallback(() => {
    setProjectRevision(prev => prev + 1)
  }, [])

  const getProject = useCallback((id: string): Project | null => readProject(id), [projectRevision])

  const reloadProjectIds = useCallback(() => {
    const nextProjectIds = hasLegacyProjectsEntry() ? [] : readProjectIds()
    setProjectIds(nextProjectIds)
    setActiveProject(prev => (
      prev && nextProjectIds.includes(prev.id) ? prev : null
    ))
    bumpProjectRevision()
  }, [bumpProjectRevision])

  const activateProject = useCallback((id: string) => {
    setActiveProject(readProject(id))
  }, [])

  const clearActiveProject = useCallback(() => {
    setActiveProject(null)
  }, [])

  const persistProject = useCallback((projectId: string, project: Project): Project => {
    const persistedProject = writeProject(projectId, normalizeProject({ ...project, id: projectId }))
    setActiveProject(prev => (prev?.id === projectId ? persistedProject : prev))
    bumpProjectRevision()
    return persistedProject
  }, [bumpProjectRevision])

  const mutateProject = useCallback((projectId: string, updater: (project: Project) => Project): Project | null => {
    const project = readProject(projectId)
    if (!project) return null
    return persistProject(projectId, updater(project))
  }, [persistProject])

  const setProject = useCallback((projectId: string, project: Project) => {
    persistProject(projectId, project)
  }, [persistProject])

  const createProject = useCallback((name: string): Project => {
    const defaultTimeline = createDefaultTimeline('Timeline 1')
    const newProject = normalizeProject({
      id: `project-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      name,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      assets: [],
      timelines: [defaultTimeline],
      activeTimelineId: defaultTimeline.id,
    })

    const persistedProject = writeProject(newProject.id, newProject)
    const nextProjectIds = [persistedProject.id, ...readProjectIds().filter(id => id !== persistedProject.id)]
    writeProjectIds(nextProjectIds)
    setProjectIds(nextProjectIds)
    bumpProjectRevision()
    return persistedProject
  }, [bumpProjectRevision])

  const deleteProject = useCallback((id: string) => {
    const nextProjectIds = readProjectIds().filter(projectId => projectId !== id)
    writeProjectIds(nextProjectIds)
    setProjectIds(nextProjectIds)
    deleteProjectEntry(id)
    setActiveProject(prev => (prev?.id === id ? null : prev))
    bumpProjectRevision()
  }, [bumpProjectRevision])

  const renameProject = useCallback((id: string, name: string) => {
    mutateProject(id, project => ({
      ...project,
      name,
      updatedAt: Date.now(),
    }))
  }, [mutateProject])

  const addAsset = useCallback((projectId: string, assetData: Omit<Asset, 'id' | 'createdAt'>): Asset => {
    const newAsset: Asset = {
      ...assetData,
      id: `asset-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      createdAt: Date.now(),
    }

    const persistedProject = mutateProject(projectId, project => ({
      ...project,
      assets: [newAsset, ...project.assets],
      updatedAt: Date.now(),
    }))
    if (!persistedProject) {
      throw new Error(`Cannot add asset: project ${projectId} was not found`)
    }

    return newAsset
  }, [mutateProject])

  const deleteAsset = useCallback((projectId: string, assetId: string) => {
    mutateProject(projectId, project => ({
      ...project,
      assets: project.assets.filter(asset => asset.id !== assetId),
      updatedAt: Date.now(),
    }))
  }, [mutateProject])

  const updateAsset = useCallback((projectId: string, assetId: string, updates: Partial<Asset>) => {
    mutateProject(projectId, project => ({
      ...project,
      assets: project.assets.map(asset => (
        asset.id === assetId ? { ...asset, ...updates } : asset
      )),
      updatedAt: Date.now(),
    }))
  }, [mutateProject])

  const addTakeToAsset = useCallback((projectId: string, assetId: string, take: AssetTake) => {
    mutateProject(projectId, project => ({
      ...project,
      assets: project.assets.map(asset => {
        if (asset.id !== assetId) return asset

        const existingTakes: AssetTake[] = asset.takes || [{
          path: asset.path,
          bigThumbnailPath: asset.bigThumbnailPath,
          smallThumbnailPath: asset.smallThumbnailPath,
          width: asset.width,
          height: asset.height,
          createdAt: asset.createdAt,
        }]
        const newTakes = [...existingTakes, take]
        const newIndex = newTakes.length - 1

        return {
          ...asset,
          takes: newTakes,
          activeTakeIndex: newIndex,
          path: take.path,
          bigThumbnailPath: take.bigThumbnailPath,
          smallThumbnailPath: take.smallThumbnailPath,
          width: take.width,
          height: take.height,
        }
      }),
      updatedAt: Date.now(),
    }))
  }, [mutateProject])

  const deleteTakeFromAsset = useCallback((projectId: string, assetId: string, takeIndex: number) => {
    mutateProject(projectId, project => ({
      ...project,
      assets: project.assets.map(asset => {
        if (asset.id !== assetId || !asset.takes || asset.takes.length <= 1) return asset

        const newTakes = asset.takes.filter((_, index) => index !== takeIndex)
        let newActiveIdx = asset.activeTakeIndex ?? newTakes.length - 1
        if (newActiveIdx >= newTakes.length) newActiveIdx = newTakes.length - 1
        if (newActiveIdx < 0) newActiveIdx = 0
        const activeTake = newTakes[newActiveIdx]

        return {
          ...asset,
          takes: newTakes,
          activeTakeIndex: newActiveIdx,
          path: activeTake.path,
          bigThumbnailPath: activeTake.bigThumbnailPath,
          smallThumbnailPath: activeTake.smallThumbnailPath,
          width: activeTake.width,
          height: activeTake.height,
        }
      }),
      updatedAt: Date.now(),
    }))
  }, [mutateProject])

  const setAssetActiveTake = useCallback((projectId: string, assetId: string, takeIndex: number) => {
    mutateProject(projectId, project => ({
      ...project,
      assets: project.assets.map(asset => {
        if (asset.id !== assetId || !asset.takes) return asset

        const nextIndex = Math.max(0, Math.min(takeIndex, asset.takes.length - 1))
        const take = asset.takes[nextIndex]

        return {
          ...asset,
          activeTakeIndex: nextIndex,
          path: take.path,
          bigThumbnailPath: take.bigThumbnailPath,
          smallThumbnailPath: take.smallThumbnailPath,
          width: take.width,
          height: take.height,
        }
      }),
      updatedAt: Date.now(),
    }))
  }, [mutateProject])

  const toggleFavorite = useCallback((projectId: string, assetId: string) => {
    mutateProject(projectId, project => ({
      ...project,
      assets: project.assets.map(asset => (
        asset.id === assetId ? { ...asset, favorite: !asset.favorite } : asset
      )),
      updatedAt: Date.now(),
    }))
  }, [mutateProject])

  return (
    <ProjectContext.Provider value={{
      currentTab,
      setCurrentTab,
      projectIds,
      activeProject,
      getProject,
      setProject,
      createProject,
      deleteProject,
      renameProject,
      activateProject,
      clearActiveProject,
      reloadProjectIds,
      addAsset,
      deleteAsset,
      updateAsset,
      addTakeToAsset,
      deleteTakeFromAsset,
      setAssetActiveTake,
      toggleFavorite,
      genSpaceEditImagePath,
      setGenSpaceEditImagePath,
      genSpaceEditMode,
      setGenSpaceEditMode,
      genSpaceAudioPath,
      setGenSpaceAudioPath,
      genSpaceRetakeSource,
      setGenSpaceRetakeSource,
      pendingRetakeUpdate,
      setPendingRetakeUpdate,
      genSpaceIcLoraSource,
      setGenSpaceIcLoraSource,
      pendingIcLoraUpdate,
      setPendingIcLoraUpdate,
    }}>
      {children}
    </ProjectContext.Provider>
  )
}

export function useProjects() {
  const context = useContext(ProjectContext)
  if (!context) {
    throw new Error('useProjects must be used within a ProjectProvider')
  }
  return context
}
