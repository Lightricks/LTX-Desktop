import { useEffect, useState } from 'react'
import { ArrowLeft, Sparkles, Film } from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import { LtxLogo } from '../components/LtxLogo'
import { Button } from '../components/ui/button'
import { GenSpace } from './GenSpace'
import { VideoEditor } from './VideoEditor'
import type { ProjectTab } from '../types/project'
import {
  hasVisualAssetMetadataForMigration,
  runVisualAssetMetadataMigration,
} from '../lib/project-asset-metadata-migration'

export function Project() {
  const {
    currentProject,
    currentProjectId,
    currentTab,
    setCurrentProject,
    setCurrentTab,
    goHome,
    updateAsset,
    pendingRetakeUpdate,
    setPendingRetakeUpdate,
    pendingIcLoraUpdate,
    setPendingIcLoraUpdate,
  } = useProjects()
  const [assetMetadataMigrationProgress, setAssetMetadataMigrationProgress] = useState({ running: false, total: 0, completed: 0 })
  const [upgradePassProjectId, setUpgradePassProjectId] = useState<string | null>(null)

  useEffect(() => {
    if (!currentProjectId || !currentProject) return

    let cancelled = false

    const runAssetMetadataMigration = async () => {
      for await (const event of runVisualAssetMetadataMigration(currentProject.assets, window.electronAPI)) {
        if (cancelled) return

        if (event.kind === 'progress') {
          setAssetMetadataMigrationProgress({ running: true, total: event.total, completed: event.completed })
          continue
        }

        for (const update of event.updates) {
          updateAsset(currentProjectId, update.assetId, update.updates)
        }

        setAssetMetadataMigrationProgress({ running: false, total: 0, completed: 0 })
        setUpgradePassProjectId(currentProjectId)
      }
    }

    void runAssetMetadataMigration()

    return () => {
      cancelled = true
    }
  }, [currentProjectId, currentProject?.id, updateAsset])

  useEffect(() => {
    if (currentTab !== 'video-editor') return
    if (pendingRetakeUpdate) setPendingRetakeUpdate(null)
    if (pendingIcLoraUpdate) setPendingIcLoraUpdate(null)
  }, [
    currentTab,
    pendingRetakeUpdate,
    setPendingRetakeUpdate,
    pendingIcLoraUpdate,
    setPendingIcLoraUpdate,
  ])
  
  if (!currentProject) {
    return (
      <div className="h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <p className="text-zinc-400 mb-4">Project not found</p>
          <Button onClick={goHome}>Go Home</Button>
        </div>
      </div>
    )
  }
  
  const tabs: { id: ProjectTab; label: string; icon: React.ReactNode }[] = [
    { id: 'gen-space', label: 'Gen Space', icon: <Sparkles className="h-4 w-4" /> },
    { id: 'video-editor', label: 'Video Editor', icon: <Film className="h-4 w-4" /> },
  ]
  const shouldShowAssetMetadataMigrationProgressScreen = assetMetadataMigrationProgress.running
    || (upgradePassProjectId !== currentProjectId && hasVisualAssetMetadataForMigration(currentProject.assets))

  if (shouldShowAssetMetadataMigrationProgressScreen) {
    const progressPct = assetMetadataMigrationProgress.total > 0
      ? (assetMetadataMigrationProgress.completed / assetMetadataMigrationProgress.total) * 100
      : 0

    return (
      <div className="h-screen bg-background flex items-center justify-center">
        <div className="w-[360px]">
          <p className="text-center text-sm text-zinc-300 mb-4">
            Preparing your project assets...
          </p>
          <div className="h-2 w-full rounded-full bg-zinc-800 overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-150"
              style={{ width: `${Math.max(0, Math.min(100, progressPct))}%` }}
            />
          </div>
        </div>
      </div>
    )
  }
  
  return (
    <div className="h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="flex items-center px-4 py-3 border-b border-zinc-800">
        <div className="flex-1 flex items-center gap-4">
          {/* Back button and logo */}
          <button 
            onClick={goHome}
            className="p-2 rounded-lg hover:bg-zinc-800 transition-colors"
          >
            <ArrowLeft className="h-5 w-5 text-zinc-400" />
          </button>
          
          <LtxLogo className="h-5 w-auto text-white" />
          
          {/* Project name */}
          <span className="text-white font-medium">{currentProject.name}</span>
        </div>
        
        {/* Center - Tabs */}
        <div className="flex items-center gap-1 bg-zinc-900 rounded-lg p-1">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setCurrentTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                currentTab === tab.id
                  ? 'bg-zinc-800 text-white'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
        
        {/* Right spacer - equal to left to keep tabs centered */}
        <div className="flex-1" />
      </header>
      
      <main className="flex-1 overflow-hidden relative">
        {currentTab === 'gen-space' ? (
          <GenSpace />
        ) : (
          <VideoEditor
            key={currentProject.id}
            currentProject={currentProject}
            setCurrentProject={setCurrentProject}
            pendingRetakeUpdate={pendingRetakeUpdate}
            pendingIcLoraUpdate={pendingIcLoraUpdate}
          />
        )}
      </main>
    </div>
  )
}
