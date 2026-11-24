import { useCallback, useEffect, useLayoutEffect, useState, type ReactNode } from 'react'
import { Sparkles, Film } from 'lucide-react'
import { Link, Navigate, useParams } from 'react-router'
import { Text } from '@ds/Text/Text'
// eslint-disable-next-line no-restricted-imports
import { useQuickSearch } from '@/ltx-io/components/QuickSearch/QuickSearchContext'
import { useProjects } from '../contexts/ProjectContext'
import { recordRecentProjectId } from '../lib/home-recent-projects-preference'
import { decideProjectRoute } from '../lib/project-navigation'
import { paths } from '../paths'
import { GenSpace } from './GenSpace'
import { VideoEditor } from './VideoEditor'
import type { ProjectTab } from '../types/project-model'
import {
  hasVisualAssetMetadataForMigration,
  runVisualAssetMetadataMigration,
} from '../lib/project-asset-metadata-migration'
import styles from './Project.module.scss'

export function Project() {
  const { projectId } = useParams()
  const { open: openQuickSearch } = useQuickSearch()
  const {
    activeProject,
    currentTab,
    getProject,
    activateProject,
    clearActiveProject,
    setProject,
    setCurrentTab,
    updateAsset,
    pendingRetakeUpdate,
    setPendingRetakeUpdate,
    pendingIcLoraUpdate,
    setPendingIcLoraUpdate,
  } = useProjects()
  const resolvedProject = projectId ? getProject(projectId) : null
  const decision = decideProjectRoute({
    projectId,
    activeProjectId: activeProject?.id ?? null,
    projectExists: resolvedProject !== null,
  })
  const [assetMetadataMigrationProgress, setAssetMetadataMigrationProgress] = useState({ running: false, total: 0, completed: 0 })

  useLayoutEffect(() => {
    if (decision !== 'activate' || !projectId) return
    recordRecentProjectId(projectId)
    activateProject(projectId)
  }, [activateProject, decision, projectId])

  useLayoutEffect(() => {
    return () => clearActiveProject()
  }, [clearActiveProject])
  const [upgradePassProjectId, setUpgradePassProjectId] = useState<string | null>(null)
  const activeProjectId = activeProject?.id ?? null
  const activeProjectAssets = activeProject?.assets ?? null
  const needsAssetMetadataMigration = activeProjectAssets
    ? hasVisualAssetMetadataForMigration(activeProjectAssets)
    : false

  const handleSaveActiveProject = useCallback((project: typeof activeProject extends null ? never : NonNullable<typeof activeProject>) => {
    if (!activeProjectId) return
    setProject(activeProjectId, project)
  }, [activeProjectId, setProject])

  useEffect(() => {
    if (!activeProjectId || !activeProjectAssets || !needsAssetMetadataMigration) return

    let cancelled = false

    const runAssetMetadataMigration = async () => {
      for await (const event of runVisualAssetMetadataMigration(activeProjectAssets, window.electronAPI)) {
        if (cancelled) return

        if (event.kind === 'progress') {
          setAssetMetadataMigrationProgress({ running: true, total: event.total, completed: event.completed })
          continue
        }

        for (const update of event.updates) {
          updateAsset(activeProjectId, update.assetId, update.updates)
        }

        setAssetMetadataMigrationProgress({ running: false, total: 0, completed: 0 })
        setUpgradePassProjectId(activeProjectId)
      }
    }

    void runAssetMetadataMigration()

    return () => {
      cancelled = true
    }
  }, [activeProjectAssets, activeProjectId, needsAssetMetadataMigration, updateAsset])

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

  if (decision === 'projects') {
    return <Navigate to={paths.projects} replace />
  }

  if (decision === 'activate' || !activeProject || activeProject.id !== projectId) {
    return null
  }

  const tabs: { id: ProjectTab; label: string; icon: ReactNode }[] = [
    { id: 'gen-space', label: 'Gen Space', icon: <Sparkles className={styles.tabIcon} /> },
    { id: 'video-editor', label: 'Video Editor', icon: <Film className={styles.tabIcon} /> },
  ]
  const shouldShowAssetMetadataMigrationProgressScreen = assetMetadataMigrationProgress.running
    || (upgradePassProjectId !== activeProjectId && needsAssetMetadataMigration)

  if (shouldShowAssetMetadataMigrationProgressScreen) {
    const progressPct = assetMetadataMigrationProgress.total > 0
      ? (assetMetadataMigrationProgress.completed / assetMetadataMigrationProgress.total) * 100
      : 0

    return (
      <div className={styles.migration}>
        <div className={styles.migrationInner}>
          <Text as="p" variant="body" size="lg" align="center" className={styles.migrationCopy}>
            Preparing your project assets...
          </Text>
          <div className={styles.migrationTrack}>
            <div
              className={styles.migrationFill}
              style={{ width: `${Math.max(0, Math.min(100, progressPct))}%` }}
            />
          </div>
        </div>
      </div>
    )
  }
  const isMac = window.electronAPI.platform === 'darwin'

  return (
    <div className={styles.root}>
      <header className={`${styles.header} ${isMac ? 'window-drag' : ''}`}>
        <div className={styles.titleSlot}>
          <Text as="span" variant="label" size="xl" shouldTruncate className={styles.title}>
            {activeProject.name}
          </Text>
        </div>

        <div className={styles.tabs} role="tablist" aria-label="Project views">
          {tabs.map(tab => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={currentTab === tab.id}
              onClick={() => setCurrentTab(tab.id)}
              className={`${styles.tab} ${currentTab === tab.id ? styles.tabSelected : ''} window-no-drag`}
            >
              {tab.icon}
              <Text as="span" variant="label" size="lg">
                {tab.label}
              </Text>
            </button>
          ))}
        </div>

        <div className={styles.titleSlot} />
      </header>

      {currentTab === 'gen-space' && (
        <Text as="p" variant="body" size="md" className={styles.deprecation}>
          Gen Space is retiring soon. Visit <Link to={paths.home}>Home</Link> or{' '}
          <button type="button" onClick={openQuickSearch}>Explore Tools</button> to keep generating with all workflows.
        </Text>
      )}

      <div className={styles.main}>
        {currentTab === 'gen-space' ? (
          <GenSpace />
        ) : (
          <VideoEditor
            key={activeProject.id}
            currentProject={activeProject}
            saveProject={handleSaveActiveProject}
            pendingRetakeUpdate={pendingRetakeUpdate}
            pendingIcLoraUpdate={pendingIcLoraUpdate}
          />
        )}
      </div>
    </div>
  )
}
