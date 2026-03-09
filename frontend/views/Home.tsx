import { useState, useEffect, useCallback } from 'react'
import { Plus, Folder, FolderOpen, MoreVertical, Trash2, Pencil, Sparkles, Image, UserCircle, Palette, ImageIcon, Braces, BookOpen, LogOut, LogIn, Key } from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import { useAppSettings } from '../contexts/AppSettingsContext'
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
  const { refreshSettings } = useAppSettings()
  const [isCreating, setIsCreating] = useState(false)
  const [newProjectName, setNewProjectName] = useState('')
  const [newProjectAssetPath, setNewProjectAssetPath] = useState('')
  const [defaultDownloadsPath, setDefaultDownloadsPath] = useState('')
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [assetFolderProjectId, setAssetFolderProjectId] = useState<string | null>(null)
  const [assetFolderPath, setAssetFolderPath] = useState('')
  const [paletteUser, setPaletteUser] = useState<{ name?: string; email?: string } | null>(null)
  const [paletteConnected, setPaletteConnected] = useState(false)
  const [creditBalance, setCreditBalance] = useState<number | null>(null)
  const [showSignInMenu, setShowSignInMenu] = useState(false)
  const [apiKeyInput, setApiKeyInput] = useState('')
  const [showApiKeyInput, setShowApiKeyInput] = useState(false)
  const [connectError, setConnectError] = useState<string | null>(null)
  const [loginEmail, setLoginEmail] = useState('')
  const [loginPassword, setLoginPassword] = useState('')
  const [loginLoading, setLoginLoading] = useState(false)

  const fetchPaletteStatus = useCallback(async () => {
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const res = await fetch(`${backendUrl}/api/sync/status`)
      if (res.ok) {
        const data = await res.json()
        setPaletteConnected(data.connected ?? false)
        setPaletteUser(data.user ?? null)
        if (data.connected) {
          // Fetch credits when connected
          const creditsRes = await fetch(`${backendUrl}/api/sync/credits`)
          if (creditsRes.ok) {
            const creditsData = await creditsRes.json()
            setCreditBalance(creditsData.balance_cents ?? creditsData.balance ?? null)
          }
        } else {
          setCreditBalance(null)
        }
      }
    } catch { /* ignore */ }
  }, [])

  const handleDisconnect = async () => {
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      await fetch(`${backendUrl}/api/sync/disconnect`, { method: 'POST' })
      setPaletteConnected(false)
      setPaletteUser(null)
      void refreshSettings()
    } catch { /* ignore */ }
  }

  const handleConnectWithApiKey = async () => {
    const trimmed = apiKeyInput.trim()
    if (!trimmed) return
    setConnectError(null)
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const res = await fetch(`${backendUrl}/api/sync/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: trimmed }),
      })
      if (res.ok) {
        const data = await res.json()
        if (data.connected) {
          setPaletteConnected(true)
          setPaletteUser(data.user ?? null)
          setApiKeyInput('')
          setShowApiKeyInput(false)
          setShowSignInMenu(false)
          void refreshSettings()
        } else {
          setConnectError(data.error || 'Connection failed. The Palette API may not be available yet.')
        }
      } else {
        setConnectError('Connection failed. Please check your key and try again.')
      }
    } catch {
      setConnectError('Could not reach the server. Please try again.')
    }
  }

  const handleEmailLogin = async () => {
    const email = loginEmail.trim()
    if (!email || !loginPassword) return
    setLoginLoading(true)
    setConnectError(null)
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const res = await fetch(`${backendUrl}/api/sync/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password: loginPassword }),
      })
      const data = await res.json()
      if (data.connected) {
        setPaletteConnected(true)
        setPaletteUser(data.user ?? null)
        setLoginEmail('')
        setLoginPassword('')
        setShowSignInMenu(false)
        void refreshSettings()
      } else {
        setConnectError(data.error || 'Login failed')
      }
    } catch {
      setConnectError('Could not reach the server. Please try again.')
    } finally {
      setLoginLoading(false)
    }
  }

  useEffect(() => {
    window.electronAPI?.getDownloadsPath().then(p => {
      setDefaultDownloadsPath(p)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    void fetchPaletteStatus()
  }, [fetchPaletteStatus])

  // Listen for deep link auth callbacks (from browser login)
  useEffect(() => {
    const cleanup = window.electronAPI.onPaletteAuthCallback(async (data) => {
      if (data.token) {
        try {
          const backendUrl = await window.electronAPI.getBackendUrl()
          const res = await fetch(`${backendUrl}/api/sync/connect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token: data.token }),
          })
          if (res.ok) {
            const result = await res.json()
            if (result.connected) {
              setPaletteConnected(true)
              setPaletteUser(result.user ?? null)
              setShowSignInMenu(false)
              void refreshSettings()
            }
          }
        } catch { /* ignore */ }
      }
    })
    return cleanup
  }, [refreshSettings])

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

        {/* ACCOUNT section */}
        <div className="mt-auto border-t border-zinc-800">
          {paletteConnected ? (
            <div className="p-3">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
                  {(paletteUser?.name || paletteUser?.email || '?')[0].toUpperCase()}
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm text-white font-medium truncate">{paletteUser?.name || 'Connected'}</p>
                  {paletteUser?.email && (
                    <p className="text-[10px] text-zinc-500 truncate">{paletteUser.email}</p>
                  )}
                </div>
              </div>
              {creditBalance !== null && (
                <div className="flex items-center gap-1.5 px-2 py-1.5 rounded-md bg-zinc-800 mb-2">
                  <div className="w-2 h-2 rounded-full bg-emerald-500" />
                  <span className="text-xs text-zinc-300 font-medium">
                    Credits: {(creditBalance / 100).toFixed(2)}
                  </span>
                </div>
              )}
              <button
                onClick={handleDisconnect}
                className="w-full px-2 py-1.5 rounded-md text-zinc-400 hover:bg-zinc-800 hover:text-red-400 text-xs flex items-center gap-2 transition-colors"
              >
                <LogOut className="h-3.5 w-3.5" />
                Sign Out
              </button>
            </div>
          ) : (
            <div className="p-3">
              <p className="text-[10px] text-zinc-600 mb-2 px-1">Connect to Director's Palette for cloud gallery, characters, credits, and sync.</p>

              {showSignInMenu ? (
                <div className="space-y-1.5">
                  {/* Email / Password login */}
                  <input
                    type="email"
                    value={loginEmail}
                    onChange={(e) => { setLoginEmail(e.target.value); setConnectError(null) }}
                    onKeyDown={(e) => { if (e.key === 'Enter') document.getElementById('home-palette-password')?.focus() }}
                    placeholder="Email address"
                    className="w-full px-2 py-1.5 rounded-md bg-zinc-800 border border-zinc-700 text-xs text-white placeholder:text-zinc-600 focus:outline-none focus:border-blue-500"
                  />
                  <input
                    id="home-palette-password"
                    type="password"
                    value={loginPassword}
                    onChange={(e) => { setLoginPassword(e.target.value); setConnectError(null) }}
                    onKeyDown={(e) => { if (e.key === 'Enter') void handleEmailLogin() }}
                    placeholder="Password"
                    className="w-full px-2 py-1.5 rounded-md bg-zinc-800 border border-zinc-700 text-xs text-white placeholder:text-zinc-600 focus:outline-none focus:border-blue-500"
                  />
                  <button
                    onClick={() => void handleEmailLogin()}
                    disabled={!loginEmail.trim() || !loginPassword || loginLoading}
                    className="w-full px-3 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium flex items-center justify-center gap-2 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <LogIn className="h-3.5 w-3.5" />
                    {loginLoading ? 'Signing in...' : 'Sign In'}
                  </button>
                  <button
                    onClick={() => setShowApiKeyInput(!showApiKeyInput)}
                    className="w-full px-3 py-1.5 rounded-md text-zinc-400 hover:bg-zinc-800 hover:text-white text-xs flex items-center gap-2 transition-colors"
                  >
                    <Key className="h-3.5 w-3.5" />
                    Use API Key
                  </button>
                  {showApiKeyInput && (
                    <div className="space-y-1">
                      <div className="flex gap-1">
                        <input
                          type="password"
                          value={apiKeyInput}
                          onChange={(e) => { setApiKeyInput(e.target.value); setConnectError(null) }}
                          onKeyDown={(e) => e.key === 'Enter' && handleConnectWithApiKey()}
                          placeholder="Paste API key..."
                          className="flex-1 min-w-0 px-2 py-1.5 rounded-md bg-zinc-800 border border-zinc-700 text-xs text-white placeholder:text-zinc-600 focus:outline-none focus:border-blue-500"
                        />
                        <button
                          onClick={handleConnectWithApiKey}
                          disabled={!apiKeyInput.trim()}
                          className="px-2 py-1.5 rounded-md bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium disabled:opacity-40 disabled:cursor-not-allowed"
                        >
                          Go
                        </button>
                      </div>
                    </div>
                  )}
                  {connectError && (
                    <p className="text-[10px] text-red-400 px-1">{connectError}</p>
                  )}
                  <button
                    onClick={() => { setShowSignInMenu(false); setShowApiKeyInput(false); setConnectError(null) }}
                    className="w-full px-3 py-1 text-[10px] text-zinc-600 hover:text-zinc-400 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setShowSignInMenu(true)}
                  className="w-full px-3 py-2 rounded-lg border border-zinc-700 hover:border-zinc-500 text-zinc-300 text-xs font-medium flex items-center justify-center gap-2 transition-colors"
                >
                  <LogIn className="h-3.5 w-3.5" />
                  Sign In to Director's Palette
                </button>
              )}
            </div>
          )}

          <div className="px-3 pb-3">
            <button
              onClick={() => setIsCreating(true)}
              className="w-full px-3 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium flex items-center justify-center gap-2 transition-colors"
            >
              <Plus className="h-4 w-4" />
              New Project
            </button>
          </div>
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
