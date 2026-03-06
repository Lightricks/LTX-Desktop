import { useState, useEffect, useCallback } from 'react'
import { ArrowLeft, Plus, Trash2, ImageIcon, X } from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import { LtxLogo } from '../components/LtxLogo'
import { Button } from '../components/ui/button'
import { logger } from '../lib/logger'

type Category = 'all' | 'people' | 'places' | 'props' | 'other'

interface Reference {
  id: string
  name: string
  category: Exclude<Category, 'all'>
  image_url: string
  created_at: string
}

export function References() {
  const { goHome } = useProjects()
  const [references, setReferences] = useState<Reference[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [category, setCategory] = useState<Category>('all')
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [formName, setFormName] = useState('')
  const [formCategory, setFormCategory] = useState<Exclude<Category, 'all'>>('people')
  const [formImage, setFormImage] = useState('')
  const [saving, setSaving] = useState(false)

  const fetchReferences = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const query = category !== 'all' ? `?category=${category}` : ''
      const res = await fetch(`${backendUrl}/api/library/references${query}`)
      if (!res.ok) throw new Error(`Failed to fetch references: ${res.status}`)
      const data = (await res.json()) as { references: Reference[] }
      setReferences(data.references ?? [])
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to load references'
      logger.error(msg)
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [category])

  useEffect(() => {
    void fetchReferences()
  }, [fetchReferences])

  const openCreate = () => {
    setFormName('')
    setFormCategory('people')
    setFormImage('')
    setIsModalOpen(true)
  }

  const handleSave = async () => {
    if (!formName.trim() || !formImage) return
    setSaving(true)
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const res = await fetch(`${backendUrl}/api/library/references`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: formName.trim(),
          category: formCategory,
          image_url: formImage,
        }),
      })
      if (!res.ok) throw new Error(`Create failed: ${res.status}`)
      setIsModalOpen(false)
      void fetchReferences()
    } catch (e) {
      logger.error(`Failed to save reference: ${e}`)
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (ref: Reference) => {
    if (!confirm(`Delete reference "${ref.name}"?`)) return
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const res = await fetch(`${backendUrl}/api/library/references/${ref.id}`, { method: 'DELETE' })
      if (!res.ok) throw new Error(`Delete failed: ${res.status}`)
      setReferences(prev => prev.filter(r => r.id !== ref.id))
    } catch (e) {
      logger.error(`Failed to delete reference: ${e}`)
    }
  }

  const handleSelectImage = async () => {
    try {
      const files = await window.electronAPI.showOpenFileDialog({
        title: 'Select Reference Image',
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

  const categories: { label: string; value: Category }[] = [
    { label: 'All', value: 'all' },
    { label: 'People', value: 'people' },
    { label: 'Places', value: 'places' },
    { label: 'Props', value: 'props' },
    { label: 'Other', value: 'other' },
  ]

  const categoryColors: Record<string, string> = {
    people: 'bg-blue-500/20 text-blue-400',
    places: 'bg-green-500/20 text-green-400',
    props: 'bg-amber-500/20 text-amber-400',
    other: 'bg-zinc-500/20 text-zinc-400',
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
        <h1 className="text-lg font-semibold text-white">References</h1>

        <div className="ml-auto flex items-center gap-2">
          {/* Category filters */}
          <div className="flex items-center bg-zinc-900 rounded-lg border border-zinc-800 p-0.5">
            {categories.map(c => (
              <button
                key={c.value}
                onClick={() => setCategory(c.value)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  category === c.value
                    ? 'bg-zinc-700 text-white'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                {c.label}
              </button>
            ))}
          </div>

          <Button onClick={openCreate} className="bg-blue-600 hover:bg-blue-500" size="sm">
            <Plus className="h-3.5 w-3.5 mr-1.5" />
            Add Reference
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
            <Button variant="outline" onClick={() => void fetchReferences()} className="border-zinc-700">
              Retry
            </Button>
          </div>
        ) : references.length === 0 ? (
          <div className="text-center py-16">
            <div className="w-16 h-16 rounded-full bg-zinc-800 flex items-center justify-center mx-auto mb-4">
              <ImageIcon className="h-8 w-8 text-zinc-600" />
            </div>
            <h3 className="text-lg font-medium text-zinc-400 mb-2">No references yet</h3>
            <p className="text-zinc-500 mb-6">Add reference images organized by category</p>
            <Button onClick={openCreate} className="bg-blue-600 hover:bg-blue-500">
              <Plus className="h-4 w-4 mr-2" />
              Add Reference
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
            {references.map(ref => (
              <div
                key={ref.id}
                className="group relative bg-zinc-900 rounded-lg border border-zinc-800 hover:border-zinc-600 transition-all overflow-hidden"
              >
                <div className="aspect-square bg-zinc-800 flex items-center justify-center overflow-hidden">
                  <img
                    src={ref.image_url}
                    alt={ref.name}
                    className="w-full h-full object-cover"
                  />
                </div>

                <div className="p-2.5">
                  <p className="text-xs text-white font-medium truncate">{ref.name}</p>
                  <span className={`inline-block mt-1 text-[10px] rounded px-1.5 py-0.5 font-medium ${categoryColors[ref.category] ?? categoryColors.other}`}>
                    {ref.category}
                  </span>
                </div>

                <button
                  onClick={() => void handleDelete(ref)}
                  className="absolute top-2 right-2 p-1.5 rounded bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-500/80"
                >
                  <Trash2 className="h-3.5 w-3.5 text-white" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Add Reference Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-zinc-900 rounded-xl p-6 w-full max-w-md border border-zinc-800">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-white">Add Reference</h2>
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
                  placeholder="Reference name"
                  className="w-full px-3 py-2.5 rounded-lg bg-zinc-800 border border-zinc-700 text-white text-sm placeholder:text-zinc-500 focus:outline-none focus:border-blue-500"
                  autoFocus
                />
              </div>

              <div>
                <label className="text-xs text-zinc-500 uppercase tracking-wider font-semibold mb-1.5 block">Category</label>
                <div className="flex gap-2">
                  {(['people', 'places', 'props', 'other'] as const).map(cat => (
                    <button
                      key={cat}
                      onClick={() => setFormCategory(cat)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors capitalize ${
                        formCategory === cat
                          ? 'bg-blue-600/20 border-blue-500 text-blue-400'
                          : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:border-zinc-600'
                      }`}
                    >
                      {cat}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-xs text-zinc-500 uppercase tracking-wider font-semibold mb-1.5 block">Image</label>
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
                  {formImage ? 'Change Image' : 'Select Image'}
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
                disabled={!formName.trim() || !formImage || saving}
                className="flex-1 bg-blue-600 hover:bg-blue-500"
              >
                {saving ? 'Saving...' : 'Create'}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
