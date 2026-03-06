import { useState, useEffect, useCallback } from 'react'
import { ArrowLeft, Image as ImageIcon, Film, Trash2, CloudUpload, Download, X, ChevronLeft, ChevronRight } from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import { LtxLogo } from '../components/LtxLogo'
import { Button } from '../components/ui/button'
import { logger } from '../lib/logger'

type FilterType = 'all' | 'images' | 'videos'

interface GalleryItem {
  id: string
  filename: string
  type: 'image' | 'video'
  url: string
  thumbnail?: string
  model?: string
  file_size?: number
  created_at: string
}

interface GalleryResponse {
  items: GalleryItem[]
  total: number
  page: number
  per_page: number
}

function formatFileSize(bytes?: number): string {
  if (!bytes) return '--'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr)
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

export function Gallery() {
  const { goHome } = useProjects()
  const [filter, setFilter] = useState<FilterType>('all')
  const [items, setItems] = useState<GalleryItem[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [previewItem, setPreviewItem] = useState<GalleryItem | null>(null)

  const perPage = 50

  const fetchGallery = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const res = await fetch(`${backendUrl}/api/gallery/local?page=${page}&per_page=${perPage}&type=${filter}`)
      if (!res.ok) throw new Error(`Failed to fetch gallery: ${res.status}`)
      const data = (await res.json()) as GalleryResponse
      setItems(data.items)
      setTotal(data.total)
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to load gallery'
      logger.error(msg)
      setError(msg)
      setItems([])
    } finally {
      setLoading(false)
    }
  }, [page, filter])

  useEffect(() => {
    void fetchGallery()
  }, [fetchGallery])

  const handleDelete = async (item: GalleryItem) => {
    if (!confirm(`Delete "${item.filename}"?`)) return
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const res = await fetch(`${backendUrl}/api/gallery/local/${item.id}`, { method: 'DELETE' })
      if (!res.ok) throw new Error(`Delete failed: ${res.status}`)
      setItems(prev => prev.filter(i => i.id !== item.id))
      setTotal(prev => prev - 1)
    } catch (e) {
      logger.error(`Failed to delete gallery item: ${e}`)
    }
  }

  const totalPages = Math.ceil(total / perPage)

  const filters: { label: string; value: FilterType; icon: React.ReactNode }[] = [
    { label: 'All', value: 'all', icon: null },
    { label: 'Images', value: 'images', icon: <ImageIcon className="h-3.5 w-3.5" /> },
    { label: 'Videos', value: 'videos', icon: <Film className="h-3.5 w-3.5" /> },
  ]

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
        <h1 className="text-lg font-semibold text-white">Gallery</h1>

        <div className="ml-auto flex items-center gap-2">
          {/* Filter tabs */}
          <div className="flex items-center bg-zinc-900 rounded-lg border border-zinc-800 p-0.5">
            {filters.map(f => (
              <button
                key={f.value}
                onClick={() => { setFilter(f.value); setPage(1) }}
                className={`px-3 py-1.5 rounded-md text-xs font-medium flex items-center gap-1.5 transition-colors ${
                  filter === f.value
                    ? 'bg-zinc-700 text-white'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                {f.icon}
                {f.label}
              </button>
            ))}
          </div>

          <Button
            variant="outline"
            size="sm"
            className="border-zinc-700 text-zinc-400 cursor-not-allowed opacity-50"
            disabled
            title="Cloud sync not connected"
          >
            <CloudUpload className="h-3.5 w-3.5 mr-1.5" />
            Push to Cloud
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
            <Button variant="outline" onClick={() => void fetchGallery()} className="border-zinc-700">
              Retry
            </Button>
          </div>
        ) : items.length === 0 ? (
          <div className="text-center py-16">
            <div className="w-16 h-16 rounded-full bg-zinc-800 flex items-center justify-center mx-auto mb-4">
              <ImageIcon className="h-8 w-8 text-zinc-600" />
            </div>
            <h3 className="text-lg font-medium text-zinc-400 mb-2">No items yet</h3>
            <p className="text-zinc-500">Generated images and videos will appear here</p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
              {items.map(item => (
                <div
                  key={item.id}
                  className="group relative bg-zinc-900 rounded-lg overflow-hidden border border-zinc-800 hover:border-zinc-600 transition-all cursor-pointer hover:shadow-lg hover:shadow-black/20"
                  onClick={() => setPreviewItem(item)}
                >
                  {/* Thumbnail */}
                  <div className="aspect-video bg-zinc-800 flex items-center justify-center overflow-hidden">
                    {item.thumbnail || item.type === 'image' ? (
                      <img
                        src={item.thumbnail || item.url}
                        alt={item.filename}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <Film className="h-8 w-8 text-zinc-600" />
                    )}
                    {item.type === 'video' && (
                      <div className="absolute top-2 left-2 bg-black/60 rounded px-1.5 py-0.5 text-[10px] font-medium text-white flex items-center gap-1">
                        <Film className="h-3 w-3" />
                        Video
                      </div>
                    )}
                  </div>

                  {/* Info */}
                  <div className="p-2.5">
                    <p className="text-xs text-white font-medium truncate">{item.filename}</p>
                    <div className="flex items-center gap-2 mt-1.5">
                      {item.model && (
                        <span className="text-[10px] bg-blue-500/20 text-blue-400 rounded px-1.5 py-0.5 font-medium">
                          {item.model}
                        </span>
                      )}
                      <span className="text-[10px] text-zinc-500">{formatFileSize(item.file_size)}</span>
                      <span className="text-[10px] text-zinc-500">{formatDate(item.created_at)}</span>
                    </div>
                  </div>

                  {/* Delete button */}
                  <button
                    onClick={(e) => { e.stopPropagation(); void handleDelete(item) }}
                    className="absolute top-2 right-2 p-1.5 rounded bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-500/80"
                  >
                    <Trash2 className="h-3.5 w-3.5 text-white" />
                  </button>
                </div>
              ))}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-3 mt-6">
                <Button
                  variant="outline"
                  size="sm"
                  className="border-zinc-700"
                  disabled={page <= 1}
                  onClick={() => setPage(p => p - 1)}
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <span className="text-sm text-zinc-400">
                  Page {page} of {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  className="border-zinc-700"
                  disabled={page >= totalPages}
                  onClick={() => setPage(p => p + 1)}
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            )}
          </>
        )}
      </div>

      {/* Preview Lightbox */}
      {previewItem && (
        <div
          className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50"
          onClick={() => setPreviewItem(null)}
        >
          <div
            className="relative max-w-4xl max-h-[85vh] w-full mx-4"
            onClick={e => e.stopPropagation()}
          >
            <button
              onClick={() => setPreviewItem(null)}
              className="absolute -top-10 right-0 p-2 text-zinc-400 hover:text-white transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
            {previewItem.type === 'video' ? (
              <video
                src={previewItem.url}
                controls
                autoPlay
                className="w-full max-h-[80vh] rounded-lg"
              />
            ) : (
              <img
                src={previewItem.url}
                alt={previewItem.filename}
                className="w-full max-h-[80vh] object-contain rounded-lg"
              />
            )}
            <div className="mt-3 flex items-center justify-between">
              <div>
                <p className="text-sm text-white font-medium">{previewItem.filename}</p>
                <p className="text-xs text-zinc-500 mt-0.5">
                  {formatFileSize(previewItem.file_size)} &middot; {formatDate(previewItem.created_at)}
                  {previewItem.model && ` \u00B7 ${previewItem.model}`}
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                className="border-zinc-700"
                onClick={() => {
                  if (previewItem.url) {
                    void window.electronAPI.showItemInFolder(previewItem.url)
                  }
                }}
              >
                <Download className="h-3.5 w-3.5 mr-1.5" />
                Show in Folder
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
