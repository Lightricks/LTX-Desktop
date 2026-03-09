import { useState, useEffect, useCallback } from 'react'
import { ArrowLeft, Plus, Pencil, Trash2, UserCircle, X } from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import { LtxLogo } from '../components/LtxLogo'
import { Button } from '../components/ui/button'
import { logger } from '../lib/logger'

interface Character {
  id: string
  name: string
  role: string
  description: string
  reference_image_paths: string[]
  created_at: string
}

export function Characters() {
  const { goHome } = useProjects()
  const [characters, setCharacters] = useState<Character[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [editingCharacter, setEditingCharacter] = useState<Character | null>(null)
  const [formName, setFormName] = useState('')
  const [formRole, setFormRole] = useState('')
  const [formDescription, setFormDescription] = useState('')
  const [formImages, setFormImages] = useState<string[]>([])
  const [saving, setSaving] = useState(false)

  const fetchCharacters = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const res = await fetch(`${backendUrl}/api/library/characters`)
      if (!res.ok) throw new Error(`Failed to fetch characters: ${res.status}`)
      const data = (await res.json()) as { characters: Character[] }
      setCharacters(data.characters ?? [])
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to load characters'
      logger.error(msg)
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchCharacters()
  }, [fetchCharacters])

  const openCreate = () => {
    setEditingCharacter(null)
    setFormName('')
    setFormRole('')
    setFormDescription('')
    setFormImages([])
    setIsModalOpen(true)
  }

  const openEdit = (char: Character) => {
    setEditingCharacter(char)
    setFormName(char.name)
    setFormRole(char.role)
    setFormDescription(char.description)
    setFormImages([...char.reference_image_paths])
    setIsModalOpen(true)
  }

  const handleSave = async () => {
    if (!formName.trim()) return
    setSaving(true)
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const body = {
        name: formName.trim(),
        role: formRole.trim(),
        description: formDescription.trim(),
        reference_image_paths: formImages,
      }
      if (editingCharacter) {
        const res = await fetch(`${backendUrl}/api/library/characters/${editingCharacter.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        if (!res.ok) throw new Error(`Update failed: ${res.status}`)
      } else {
        const res = await fetch(`${backendUrl}/api/library/characters`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        if (!res.ok) throw new Error(`Create failed: ${res.status}`)
      }
      setIsModalOpen(false)
      void fetchCharacters()
    } catch (e) {
      logger.error(`Failed to save character: ${e}`)
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (char: Character) => {
    if (!confirm(`Delete character "${char.name}"?`)) return
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const res = await fetch(`${backendUrl}/api/library/characters/${char.id}`, { method: 'DELETE' })
      if (!res.ok) throw new Error(`Delete failed: ${res.status}`)
      setCharacters(prev => prev.filter(c => c.id !== char.id))
    } catch (e) {
      logger.error(`Failed to delete character: ${e}`)
    }
  }

  const handleAddImage = async () => {
    try {
      const files = await window.electronAPI.showOpenFileDialog({
        title: 'Select Reference Image',
        filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'webp'] }],
        properties: ['openFile'],
      })
      if (files && files.length > 0) {
        setFormImages(prev => [...prev, ...files])
      }
    } catch (e) {
      logger.error(`Failed to open file dialog: ${e}`)
    }
  }

  return (
    <div className="h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="flex items-center gap-3 px-6 py-4 border-b border-zinc-800 shrink-0">
        <button
          onClick={goHome}
          className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
        </button>
        <LtxLogo className="h-5 w-auto text-white" />
        <span className="text-zinc-500 text-sm">/</span>
        <h1 className="text-lg font-semibold text-white">Characters</h1>
        <div className="ml-auto">
          <Button onClick={openCreate} className="bg-blue-600 hover:bg-blue-500" size="sm">
            <Plus className="h-3.5 w-3.5 mr-1.5" />
            Add Character
          </Button>
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="h-8 w-8 border-2 border-zinc-600 border-t-blue-500 rounded-full animate-spin" />
          </div>
        ) : error ? (
          <div className="text-center py-16">
            <p className="text-zinc-400 mb-4">{error}</p>
            <Button variant="outline" onClick={() => void fetchCharacters()} className="border-zinc-700">
              Retry
            </Button>
          </div>
        ) : characters.length === 0 ? (
          <div className="text-center py-16">
            <div className="w-16 h-16 rounded-full bg-zinc-800 flex items-center justify-center mx-auto mb-4">
              <UserCircle className="h-8 w-8 text-zinc-600" />
            </div>
            <h3 className="text-lg font-medium text-zinc-400 mb-2">No characters yet</h3>
            <p className="text-zinc-500 mb-6">Create characters with reference images for consistent generation</p>
            <Button onClick={openCreate} className="bg-blue-600 hover:bg-blue-500">
              <Plus className="h-4 w-4 mr-2" />
              Add Character
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {characters.map(char => (
              <div
                key={char.id}
                className="group bg-zinc-900 rounded-lg border border-zinc-800 hover:border-zinc-600 transition-all overflow-hidden"
              >
                {/* Reference images */}
                <div className="aspect-video bg-zinc-800 flex items-center justify-center overflow-hidden">
                  {char.reference_image_paths.length > 0 ? (
                    <div className="grid grid-cols-2 w-full h-full">
                      {char.reference_image_paths.slice(0, 4).map((img, i) => (
                        <img
                          key={i}
                          src={img}
                          alt={`${char.name} ref ${i + 1}`}
                          className="w-full h-full object-cover"
                        />
                      ))}
                    </div>
                  ) : (
                    <UserCircle className="h-12 w-12 text-zinc-600" />
                  )}
                </div>

                <div className="p-3">
                  <h3 className="text-sm font-semibold text-white">{char.name}</h3>
                  {char.role && (
                    <span className="inline-block mt-1 text-[10px] bg-blue-500/20 text-blue-400 rounded px-1.5 py-0.5 font-medium">
                      {char.role}
                    </span>
                  )}
                  {char.description && (
                    <p className="text-xs text-zinc-500 mt-1.5 line-clamp-2">{char.description}</p>
                  )}

                  <div className="flex items-center gap-1 mt-3 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => openEdit(char)}
                      className="p-1.5 rounded text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </button>
                    <button
                      onClick={() => void handleDelete(char)}
                      className="p-1.5 rounded text-zinc-400 hover:text-red-400 hover:bg-zinc-800 transition-colors"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Create/Edit Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-zinc-900 rounded-xl p-6 w-full max-w-md border border-zinc-800">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-white">
                {editingCharacter ? 'Edit Character' : 'Add Character'}
              </h2>
              <button onClick={() => setIsModalOpen(false)} className="text-zinc-400 hover:text-white">
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="text-xs text-zinc-500 uppercase tracking-wider font-semibold mb-1.5 block">Name</label>
                <input
                  type="text"
                  value={formName}
                  onChange={e => setFormName(e.target.value)}
                  placeholder="Character name"
                  className="w-full px-3 py-2.5 rounded-lg bg-zinc-800 border border-zinc-700 text-white text-sm placeholder:text-zinc-500 focus:outline-none focus:border-blue-500"
                  autoFocus
                />
              </div>

              <div>
                <label className="text-xs text-zinc-500 uppercase tracking-wider font-semibold mb-1.5 block">Role</label>
                <input
                  type="text"
                  value={formRole}
                  onChange={e => setFormRole(e.target.value)}
                  placeholder="e.g. Protagonist, Narrator"
                  className="w-full px-3 py-2.5 rounded-lg bg-zinc-800 border border-zinc-700 text-white text-sm placeholder:text-zinc-500 focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="text-xs text-zinc-500 uppercase tracking-wider font-semibold mb-1.5 block">Description</label>
                <textarea
                  value={formDescription}
                  onChange={e => setFormDescription(e.target.value)}
                  placeholder="Physical appearance, personality traits..."
                  rows={3}
                  className="w-full px-3 py-2.5 rounded-lg bg-zinc-800 border border-zinc-700 text-white text-sm placeholder:text-zinc-500 focus:outline-none focus:border-blue-500 resize-none"
                />
              </div>

              <div>
                <label className="text-xs text-zinc-500 uppercase tracking-wider font-semibold mb-1.5 block">Reference Images</label>
                <div className="flex flex-wrap gap-2 mb-2">
                  {formImages.map((img, i) => (
                    <div key={i} className="relative w-16 h-16 rounded-lg overflow-hidden border border-zinc-700">
                      <img src={img} alt="" className="w-full h-full object-cover" />
                      <button
                        onClick={() => setFormImages(prev => prev.filter((_, idx) => idx !== i))}
                        className="absolute top-0.5 right-0.5 bg-black/70 rounded p-0.5"
                      >
                        <X className="h-3 w-3 text-white" />
                      </button>
                    </div>
                  ))}
                </div>
                <Button variant="outline" size="sm" className="border-zinc-700" onClick={() => void handleAddImage()}>
                  <Plus className="h-3.5 w-3.5 mr-1" />
                  Add Image
                </Button>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <Button
                variant="outline"
                onClick={() => setIsModalOpen(false)}
                className="flex-1 border-zinc-700"
              >
                Cancel
              </Button>
              <Button
                onClick={() => void handleSave()}
                disabled={!formName.trim() || saving}
                className="flex-1 bg-blue-600 hover:bg-blue-500"
              >
                {saving ? 'Saving...' : editingCharacter ? 'Update' : 'Create'}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
