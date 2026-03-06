import { useState, useEffect } from 'react'
import { Plus, Folder, FolderOpen, MoreVertical, Trash2, Pencil, Sparkles, Image, UserCircle, Palette, ImageIcon, Braces, BookOpen } from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import { LtxLogo } from '../components/LtxLogo'
import { Button } from '../components/ui/button'
import type { Project } from '../types/project'

function formatDate(timestamp: number): string {
  const date = new Date(timestamp)
  return date.toLocaleDateString('en-US', { 
    month: 'short', 
    day: 'numeric', 
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

function ProjectCard({ project, onOpen, onDelete, onRename, onSetAssetFolder }: { 
  project: Project
  onOpen: () => void
  onDelete: () => void
  onRename: () => void
  onSetAssetFolder: () => void
}) {
  const [showMenu, setShowMenu] = useState(false)
  const [imgError, setImgError] = useState(false)
  
  // Get thumbnail: use stored thumbnail, or first asset's URL as fallback
  const thumbnailUrl = project.thumbnail || (project.assets.length > 0 ? project.assets[0].url : null)
  // For videos, try to find the first image asset for a better thumbnail
  const bestThumbnail = project.assets.find(a => a.type === 'image')?.url || thumbnailUrl
  
  return (
    <div 
      className="group relative bg-zinc-900 rounded-lg overflow-hidden border border-zinc-800 hover:border-zinc-700 transition-colors cursor-pointer"
      onClick={onOpen}
    >
      {/* Thumbnail */}
      <div className="aspect-video bg-zinc-800 flex items-center justify-center relative overflow-hidden">
        {bestThumbnail && !imgError ? (
          project.assets.find(a => a.type === 'video' && a.url === bestThumbnail) ? (
            <video 
              src={bestThumbnail} 
              className="w-full h-full object-cover" 
              muted 
              preload="metadata"
              onError={() => setImgError(true)}
            />
          ) : (
            <img 
              src={bestThumbnail} 
              alt={project.name} 
              className="w-full h-full object-cover" 
              onError={() => setImgError(true)}
            />
          )
        ) : (
          <Folder className="h-12 w-12 text-zinc-600" />
        )}
        {/* Hover overlay */}
        <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity" />
      </div>
      
      {/* Info */}
      <div className="p-3">
        <h3 className="font-medium text-white truncate">{project.name}</h3>
        <p className="text-xs text-zinc-500 mt-1">{formatDate(project.updatedAt)}</p>
      </div>
      
      {/* Menu button */}
      <button
        onClick={(e) => {
          e.stopPropagation()
          setShowMenu(!showMenu)
        }}
        className="absolute top-2 right-2 p-1.5 rounded bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-black/70"
      >
        <MoreVertical className="h-4 w-4 text-white" />
      </button>
      
      {/* Dropdown menu */}
      {showMenu && (
        <div 
          className="absolute top-10 right-2 bg-zinc-800 rounded-lg shadow-lg border border-zinc-700 py-1 z-10 min-w-[120px]"
          onClick={(e) => e.stopPropagation()}
        >
          <button
            onClick={() => { onRename(); setShowMenu(false) }}
            className="w-full px-3 py-2 text-left text-sm text-zinc-300 hover:bg-zinc-700 flex items-center gap-2"
          >
            <Pencil className="h-4 w-4" />
            Rename
          </button>
          <button
            onClick={() => { onSetAssetFolder(); setShowMenu(false) }}
            className="w-full px-3 py-2 text-left text-sm text-zinc-300 hover:bg-zinc-700 flex items-center gap-2"
          >
            <FolderOpen className="h-4 w-4" />
            Asset Folder
          </button>
          <button
            onClick={() => { onDelete(); setShowMenu(false) }}
            className="w-full px-3 py-2 text-left text-sm text-red-400 hover:bg-zinc-700 flex items-center gap-2"
          >
            <Trash2 className="h-4 w-4" />
            Delete
          </button>
        </div>
      )}
    </div>
  )
}

export function Home() {
  const { projects, createProject, deleteProject, renameProject, updateProject, openProject, openPlayground, openGallery, openCharacters, openStyles, openReferences, openWildcards, openPromptLibrary } = useProjects()
  const [isCreating, setIsCreating] = useState(false)
  const [newProjectName, setNewProjectName] = useState('')
  const [newProjectAssetPath, setNewProjectAssetPath] = useState('')
  const [defaultDownloadsPath, setDefaultDownloadsPath] = useState('')
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [assetFolderProjectId, setAssetFolderProjectId] = useState<string | null>(null)
  const [assetFolderPath, setAssetFolderPath] = useState('')

  useEffect(() => {
    window.electronAPI?.getDownloadsPath().then(p => {
      setDefaultDownloadsPath(p)
    }).catch(() => {})
  }, [])

  const getDefaultAssetPath = (name: string) => {
    if (!defaultDownloadsPath) return ''
    const sep = defaultDownloadsPath.includes('\\') ? '\\' : '/'
    return `${defaultDownloadsPath}${sep}Directors Desktop Assets${sep}${name}`
  }
  
  const handleCreateProject = () => {
    if (newProjectName.trim()) {
      const assetPath = newProjectAssetPath.trim() || getDefaultAssetPath(newProjectName.trim())
      const project = createProject(newProjectName.trim(), assetPath || undefined)
      setNewProjectName('')
      setNewProjectAssetPath('')
      setIsCreating(false)
      openProject(project.id)
    }
  }

  const handleBrowseAssetFolder = async (setter: (v: string) => void) => {
    const dir = await window.electronAPI?.showOpenDirectoryDialog({ title: 'Select Asset Folder' })
    if (dir) setter(dir)
  }

  const handleSaveAssetFolder = () => {
    if (assetFolderProjectId && assetFolderPath.trim()) {
      updateProject(assetFolderProjectId, { assetSavePath: assetFolderPath.trim() })
    }
    setAssetFolderProjectId(null)
    setAssetFolderPath('')
  }
  
  const handleRenameProject = (id: string, currentName: string) => {
    setRenamingId(id)
    setRenameValue(currentName)
  }
  
  const submitRename = () => {
    if (renamingId && renameValue.trim()) {
      renameProject(renamingId, renameValue.trim())
    }
    setRenamingId(null)
    setRenameValue('')
  }
  
  return (
    <div className="h-screen bg-background flex">
      {/* Sidebar */}
      <aside className="w-64 border-r border-zinc-800 flex flex-col">
        <div className="p-6">
          <LtxLogo className="h-6 w-auto text-white" />
        </div>
        
        <nav className="flex-1 px-3 overflow-y-auto">
          <button className="w-full px-3 py-2 rounded-lg bg-zinc-800 text-white text-left text-sm font-medium flex items-center gap-2">
            <Folder className="h-4 w-4" />
            Home
          </button>

          {/* CREATE */}
          <div className="mt-6">
            <h4 className="px-3 text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">
              Create
            </h4>
            <button
              onClick={openPlayground}
              className="w-full px-3 py-2 rounded-lg text-zinc-400 hover:bg-zinc-800 hover:text-white text-left text-sm flex items-center gap-2 transition-colors"
            >
              <Sparkles className="h-4 w-4" />
              Playground
            </button>
          </div>

          {/* EDIT */}
          <div className="mt-5">
            <h4 className="px-3 text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">
              Edit
            </h4>
            {projects.slice(0, 5).map(project => (
              <button
                key={project.id}
                onClick={() => openProject(project.id)}
                className="w-full px-3 py-2 rounded-lg text-zinc-400 hover:bg-zinc-800 hover:text-white text-left text-sm flex items-center gap-2 transition-colors truncate"
              >
                <Folder className="h-4 w-4 flex-shrink-0" />
                <span className="truncate">{project.name}</span>
              </button>
            ))}
          </div>

          {/* LIBRARY */}
          <div className="mt-5">
            <h4 className="px-3 text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">
              Library
            </h4>
            <button
              onClick={openGallery}
              className="w-full px-3 py-2 rounded-lg text-zinc-400 hover:bg-zinc-800 hover:text-white text-left text-sm flex items-center gap-2 transition-colors"
            >
              <Image className="h-4 w-4" />
              Gallery
            </button>
            <button
              onClick={openCharacters}
              className="w-full px-3 py-2 rounded-lg text-zinc-400 hover:bg-zinc-800 hover:text-white text-left text-sm flex items-center gap-2 transition-colors"
            >
              <UserCircle className="h-4 w-4" />
              Characters
            </button>
            <button
              onClick={openStyles}
              className="w-full px-3 py-2 rounded-lg text-zinc-400 hover:bg-zinc-800 hover:text-white text-left text-sm flex items-center gap-2 transition-colors"
            >
              <Palette className="h-4 w-4" />
              Styles
            </button>
            <button
              onClick={openReferences}
              className="w-full px-3 py-2 rounded-lg text-zinc-400 hover:bg-zinc-800 hover:text-white text-left text-sm flex items-center gap-2 transition-colors"
            >
              <ImageIcon className="h-4 w-4" />
              References
            </button>
          </div>

          {/* TOOLS */}
          <div className="mt-5">
            <h4 className="px-3 text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">
              Tools
            </h4>
            <button
              onClick={openWildcards}
              className="w-full px-3 py-2 rounded-lg text-zinc-400 hover:bg-zinc-800 hover:text-white text-left text-sm flex items-center gap-2 transition-colors"
            >
              <Braces className="h-4 w-4" />
              Wildcards
            </button>
            <button
              onClick={openPromptLibrary}
              className="w-full px-3 py-2 rounded-lg text-zinc-400 hover:bg-zinc-800 hover:text-white text-left text-sm flex items-center gap-2 transition-colors"
            >
              <BookOpen className="h-4 w-4" />
              Prompt Library
            </button>
          </div>
        </nav>
        
        <div className="p-4 border-t border-zinc-800">
          <button
            onClick={() => setIsCreating(true)}
            className="w-full px-3 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium flex items-center justify-center gap-2 transition-colors"
          >
            <Plus className="h-4 w-4" />
            New Project
          </button>
        </div>
      </aside>
      
      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        {/* Header Banner with video background */}
        <div className="relative h-72 overflow-hidden">
          <video
            src="./hero-video.mp4"
            autoPlay
            loop
            muted
            playsInline
            className="absolute inset-0 w-full h-full object-cover"
          />
          {/* Dark overlay for text readability */}
          <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/30 to-black/10" />
          <div className="absolute bottom-6 left-8 z-10">
            <h1 className="text-3xl font-bold text-white mb-2 drop-shadow-lg">Director's Desktop</h1>
            <p className="text-zinc-200 drop-shadow-md">Create and manage your video projects</p>
          </div>
        </div>
        
        {/* Projects Grid */}
        <div className="p-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-white">Projects</h2>
          </div>
          
          {projects.length === 0 ? (
            <div className="text-center py-16">
              <Folder className="h-16 w-16 text-zinc-700 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-zinc-400 mb-2">No projects yet</h3>
              <p className="text-zinc-500 mb-6">Create your first project to get started</p>
              <Button 
                onClick={() => setIsCreating(true)}
                className="bg-blue-600 hover:bg-blue-500"
              >
                <Plus className="h-4 w-4 mr-2" />
                Create Project
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {projects.map(project => (
                <ProjectCard
                  key={project.id}
                  project={project}
                  onOpen={() => openProject(project.id)}
                  onDelete={() => {
                    if (confirm(`Delete "${project.name}"?`)) {
                      deleteProject(project.id)
                    }
                  }}
                  onRename={() => handleRenameProject(project.id, project.name)}
                  onSetAssetFolder={() => {
                    setAssetFolderProjectId(project.id)
                    setAssetFolderPath(project.assetSavePath || getDefaultAssetPath(project.name))
                  }}
                />
              ))}
            </div>
          )}
        </div>
      </main>
      
      {/* Create Project Modal */}
      {isCreating && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-zinc-900 rounded-xl p-6 w-full max-w-md border border-zinc-800">
            <h2 className="text-xl font-semibold text-white mb-4">Create New Project</h2>
            <input
              type="text"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              placeholder="Project name"
              className="w-full px-4 py-3 rounded-lg bg-zinc-800 border border-zinc-700 text-white placeholder:text-zinc-500 focus:outline-none focus:border-blue-500"
              autoFocus
              onKeyDown={(e) => e.key === 'Enter' && handleCreateProject()}
            />
            <div className="mt-4">
              <label className="text-xs text-zinc-500 uppercase tracking-wider font-semibold mb-1.5 block">Asset Save Folder</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={newProjectAssetPath || (newProjectName.trim() ? getDefaultAssetPath(newProjectName.trim()) : '')}
                  onChange={(e) => setNewProjectAssetPath(e.target.value)}
                  placeholder={newProjectName.trim() ? getDefaultAssetPath(newProjectName.trim()) : 'Downloads/Directors Desktop Assets/...'}
                  className="flex-1 px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-300 text-sm placeholder:text-zinc-600 focus:outline-none focus:border-blue-500 truncate"
                />
                <Button
                  variant="outline"
                  size="sm"
                  className="border-zinc-700 flex-shrink-0"
                  onClick={() => handleBrowseAssetFolder(setNewProjectAssetPath)}
                >
                  <FolderOpen className="h-4 w-4" />
                </Button>
              </div>
              <p className="text-[10px] text-zinc-600 mt-1">Where generated assets will be saved</p>
            </div>
            <div className="flex gap-3 mt-6">
              <Button
                variant="outline"
                onClick={() => { setIsCreating(false); setNewProjectName(''); setNewProjectAssetPath('') }}
                className="flex-1 border-zinc-700"
              >
                Cancel
              </Button>
              <Button
                onClick={handleCreateProject}
                disabled={!newProjectName.trim()}
                className="flex-1 bg-blue-600 hover:bg-blue-500"
              >
                Create
              </Button>
            </div>
          </div>
        </div>
      )}
      
      {/* Rename Modal */}
      {renamingId && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-zinc-900 rounded-xl p-6 w-full max-w-md border border-zinc-800">
            <h2 className="text-xl font-semibold text-white mb-4">Rename Project</h2>
            <input
              type="text"
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
              placeholder="Project name"
              className="w-full px-4 py-3 rounded-lg bg-zinc-800 border border-zinc-700 text-white placeholder:text-zinc-500 focus:outline-none focus:border-blue-500"
              autoFocus
              onKeyDown={(e) => e.key === 'Enter' && submitRename()}
            />
            <div className="flex gap-3 mt-6">
              <Button
                variant="outline"
                onClick={() => { setRenamingId(null); setRenameValue('') }}
                className="flex-1 border-zinc-700"
              >
                Cancel
              </Button>
              <Button
                onClick={submitRename}
                disabled={!renameValue.trim()}
                className="flex-1 bg-blue-600 hover:bg-blue-500"
              >
                Save
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Asset Folder Modal */}
      {assetFolderProjectId && (() => {
        const proj = projects.find(p => p.id === assetFolderProjectId)
        return (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
            <div className="bg-zinc-900 rounded-xl p-6 w-full max-w-md border border-zinc-800">
              <h2 className="text-xl font-semibold text-white mb-1">Asset Save Folder</h2>
              <p className="text-xs text-zinc-500 mb-4">Where generated assets for "{proj?.name}" will be saved</p>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={assetFolderPath}
                  onChange={(e) => setAssetFolderPath(e.target.value)}
                  placeholder="Select a folder..."
                  className="flex-1 px-3 py-2.5 rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-300 text-sm placeholder:text-zinc-600 focus:outline-none focus:border-blue-500 truncate"
                  autoFocus
                  onKeyDown={(e) => e.key === 'Enter' && handleSaveAssetFolder()}
                />
                <Button
                  variant="outline"
                  className="border-zinc-700 flex-shrink-0"
                  onClick={() => handleBrowseAssetFolder(setAssetFolderPath)}
                >
                  <FolderOpen className="h-4 w-4" />
                </Button>
              </div>
              <div className="flex gap-3 mt-6">
                <Button
                  variant="outline"
                  onClick={() => { setAssetFolderProjectId(null); setAssetFolderPath('') }}
                  className="flex-1 border-zinc-700"
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleSaveAssetFolder}
                  disabled={!assetFolderPath.trim()}
                  className="flex-1 bg-blue-600 hover:bg-blue-500"
                >
                  Save
                </Button>
              </div>
            </div>
          </div>
        )
      })()}
    </div>
  )
}
