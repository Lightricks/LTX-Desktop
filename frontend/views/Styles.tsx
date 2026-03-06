import { useState, useEffect, useCallback } from 'react'
import { ArrowLeft, Plus, Pencil, Trash2, Palette, X } from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import { LtxLogo } from '../components/LtxLogo'
import { Button } from '../components/ui/button'
import { logger } from '../lib/logger'

interface Style {
  id: string
  name: string
  description: string
  reference_image?: string
  created_at: string
}

export function Styles() {
  const { goHome } = useProjects()
  const [styles, setStyles] = useState<Style[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [editingStyle, setEditingStyle] = useState<Style | null>(null)
  const [formName, setFormName] = useState('')
  const [formDescription, setFormDescription] = useState('')
  const [formImage, setFormImage] = useState('')
  const [saving, setSaving] = useState(false)

  const fetchStyles = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const res = await fetch(`${backendUrl}/api/library/styles`)
      if (!res.ok) throw new Error(`Failed to fetch styles: ${res.status}`)
      const data = (await res.json()) as { styles: Style[] }
      setStyles(data.styles ?? [])
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to load styles'
      logger.error(msg)
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchStyles()
  }, [fetchStyles])

  const openCreate = () => {
    setEditingStyle(null)
    setFormName('')
    setFormDescription('')
    setFormImage('')
    setIsModalOpen(true)
  }

  const openEdit = (style: Style) => {
    setEditingStyle(style)
    setFormName(style.name)
    setFormDescription(style.description)
    setFormImage(style.reference_image ?? '')
    setIsModalOpen(true)
  }

  const handleSave = async () => {
    if (!formName.trim()) return
    setSaving(true)
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const body = {
        name: formName.trim(),
        description: formDescription.trim(),
        reference_image: formImage || undefined,
      }
      if (editingStyle) {
        const res = await fetch(`${backendUrl}/api/library/styles/${editingStyle.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        if (!res.ok) throw new Error(`Update failed: ${res.status}`)
      } else {
        const res = await fetch(`${backendUrl}/api/library/styles`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        if (!res.ok) throw new Error(`Create failed: ${res.status}`)
      }
      setIsModalOpen(false)
      void fetchStyles()
    } catch (e) {
      logger.error(`Failed to save style: ${e}`)
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (style: Style) => {
    if (!confirm(`Delete style "${style.name}"?`)) return
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const res = await fetch(`${backendUrl}/api/library/styles/${style.id}`, { method: 'DELETE' })
      if (!res.ok) throw new Error(`Delete failed: ${res.status}`)
      setStyles(prev => prev.filter(s => s.id !== style.id))
    } catch (e) {
      logger.error(`Failed to delete style: ${e}`)
    }
  }

  const handleSelectImage = async () => {
    try {
      const files = await window.electronAPI.showOpenFileDialog({
        title: 'Select Style Reference Image',
        filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'webp'] }],
        properties: ['openFile'],
      })
      if (files && files.length > 0) {
        setFormImage(files[0])
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
        <h1 className="text-lg font-semibold text-white">Styles</h1>
        <div className="ml-auto">
          <Button onClick={openCreate} className="bg-blue-600 hover:bg-blue-500" size="sm">
            <Plus className="h-3.5 w-3.5 mr-1.5" />
            Add Style
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
            <Button variant="outline" onClick={() => void fetchStyles()} className="border-zinc-700">
              Retry
            </Button>
          </div>
        ) : styles.length === 0 ? (
          <div className="text-center py-16">
            <div className="w-16 h-16 rounded-full bg-zinc-800 flex items-center justify-center mx-auto mb-4">
              <Palette className="h-8 w-8 text-zinc-600" />
            </div>
            <h3 className="text-lg font-medium text-zinc-400 mb-2">No styles yet</h3>
            <p className="text-zinc-500 mb-6">Save visual styles with reference images for reuse</p>
            <Button onClick={openCreate} className="bg-blue-600 hover:bg-blue-500">
              <Plus className="h-4 w-4 mr-2" />
              Add Style
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {styles.map(style => (
              <div
                key={style.id}
                className="group bg-zinc-900 rounded-lg border border-zinc-800 hover:border-zinc-600 transition-all overflow-hidden"
              >
                <div className="aspect-video bg-zinc-800 flex items-center justify-center overflow-hidden">
                  {style.reference_image ? (
                    <img
                      src={style.reference_image}
                      alt={style.name}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <Palette className="h-12 w-12 text-zinc-600" />
                  )}
                </div>

                <div className="p-3">
                  <h3 className="text-sm font-semibold text-white">{style.name}</h3>
                  {style.description && (
                    <p className="text-xs text-zinc-500 mt-1 line-clamp-2">{style.description}</p>
                  )}

                  <div className="flex items-center gap-1 mt-3 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => openEdit(style)}
                      className="p-1.5 rounded text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </button>
                    <button
                      onClick={() => void handleDelete(style)}
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
                {editingStyle ? 'Edit Style' : 'Add Style'}
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
                  placeholder="Style name"
                  className="w-full px-3 py-2.5 rounded-lg bg-zinc-800 border border-zinc-700 text-white text-sm placeholder:text-zinc-500 focus:outline-none focus:border-blue-500"
                  autoFocus
                />
              </div>

              <div>
                <label className="text-xs text-zinc-500 uppercase tracking-wider font-semibold mb-1.5 block">Description</label>
                <textarea
                  value={formDescription}
                  onChange={e => setFormDescription(e.target.value)}
                  placeholder="Describe the visual style..."
                  rows={3}
                  className="w-full px-3 py-2.5 rounded-lg bg-zinc-800 border border-zinc-700 text-white text-sm placeholder:text-zinc-500 focus:outline-none focus:border-blue-500 resize-none"
                />
              </div>

              <div>
                <label className="text-xs text-zinc-500 uppercase tracking-wider font-semibold mb-1.5 block">Reference Image</label>
                {formImage ? (
                  <div className="relative w-full aspect-video rounded-lg overflow-hidden border border-zinc-700 mb-2">
                    <img src={formImage} alt="" className="w-full h-full object-cover" />
                    <button
                      onClick={() => setFormImage('')}
                      className="absolute top-2 right-2 bg-black/70 rounded p-1"
                    >
                      <X className="h-3.5 w-3.5 text-white" />
                    </button>
                  </div>
                ) : null}
                <Button variant="outline" size="sm" className="border-zinc-700" onClick={() => void handleSelectImage()}>
                  <Plus className="h-3.5 w-3.5 mr-1" />
                  {formImage ? 'Change Image' : 'Add Image'}
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
                {saving ? 'Saving...' : editingStyle ? 'Update' : 'Create'}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
