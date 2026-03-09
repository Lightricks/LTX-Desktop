import { useState, useEffect, useCallback, useMemo } from 'react'
import { ArrowLeft, Plus, Copy, Check, BookOpen, Search, X } from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import { LtxLogo } from '../components/LtxLogo'
import { Button } from '../components/ui/button'
import { logger } from '../lib/logger'

type SortMode = 'most-used' | 'recent' | 'alphabetical'

interface SavedPrompt {
  id: string
  text: string
  tags: string[]
  used_count: number
  last_used_at: string | null
  created_at: string
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr)
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

export function PromptLibrary() {
  const { goHome } = useProjects()
  const [prompts, setPrompts] = useState<SavedPrompt[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [sortMode, setSortMode] = useState<SortMode>('most-used')
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [formText, setFormText] = useState('')
  const [formTags, setFormTags] = useState('')
  const [saving, setSaving] = useState(false)

  const fetchPrompts = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const res = await fetch(`${backendUrl}/api/prompts`)
      if (!res.ok) throw new Error(`Failed to fetch prompts: ${res.status}`)
      const data = (await res.json()) as { prompts: SavedPrompt[] }
      setPrompts(data.prompts ?? [])
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to load prompts'
      logger.error(msg)
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchPrompts()
  }, [fetchPrompts])

  const handleCopy = async (prompt: SavedPrompt) => {
    try {
      await navigator.clipboard.writeText(prompt.text)
      setCopiedId(prompt.id)
      setTimeout(() => setCopiedId(null), 2000)
    } catch (e) {
      logger.error(`Failed to copy to clipboard: ${e}`)
    }
  }

  const handleSave = async () => {
    if (!formText.trim()) return
    setSaving(true)
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const tags = formTags.split(',').map(t => t.trim()).filter(Boolean)
      const res = await fetch(`${backendUrl}/api/prompts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: formText.trim(), tags }),
      })
      if (!res.ok) throw new Error(`Save failed: ${res.status}`)
      setIsModalOpen(false)
      setFormText('')
      setFormTags('')
      void fetchPrompts()
    } catch (e) {
      logger.error(`Failed to save prompt: ${e}`)
    } finally {
      setSaving(false)
    }
  }

  const filtered = useMemo(() => {
    let result = [...prompts]
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase()
      result = result.filter(p =>
        p.text.toLowerCase().includes(q) ||
        p.tags.some(t => t.toLowerCase().includes(q))
      )
    }
    switch (sortMode) {
      case 'most-used':
        result.sort((a, b) => b.used_count - a.used_count)
        break
      case 'recent':
        result.sort((a, b) => new Date(b.last_used_at ?? '').getTime() - new Date(a.last_used_at ?? '').getTime())
        break
      case 'alphabetical':
        result.sort((a, b) => a.text.localeCompare(b.text))
        break
    }
    return result
  }, [prompts, searchQuery, sortMode])

  const sortOptions: { label: string; value: SortMode }[] = [
    { label: 'Most Used', value: 'most-used' },
    { label: 'Recent', value: 'recent' },
    { label: 'A-Z', value: 'alphabetical' },
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
        <h1 className="text-lg font-semibold text-white">Prompt Library</h1>

        <div className="ml-auto flex items-center gap-2">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-500" />
            <input
              type="text"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Search prompts..."
              className="pl-8 pr-3 py-1.5 w-56 rounded-lg bg-zinc-900 border border-zinc-800 text-sm text-white placeholder:text-zinc-500 focus:outline-none focus:border-blue-500"
            />
          </div>

          {/* Sort */}
          <div className="flex items-center bg-zinc-900 rounded-lg border border-zinc-800 p-0.5">
            {sortOptions.map(s => (
              <button
                key={s.value}
                onClick={() => setSortMode(s.value)}
                className={`px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  sortMode === s.value
                    ? 'bg-zinc-700 text-white'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>

          <Button onClick={() => setIsModalOpen(true)} className="bg-blue-600 hover:bg-blue-500" size="sm">
            <Plus className="h-3.5 w-3.5 mr-1.5" />
            Save Prompt
          </Button>
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-4xl mx-auto">
          {loading ? (
            <div className="flex items-center justify-center h-64">
              <div className="h-8 w-8 border-2 border-zinc-600 border-t-blue-500 rounded-full animate-spin" />
            </div>
          ) : error ? (
            <div className="text-center py-16">
              <p className="text-zinc-400 mb-4">{error}</p>
              <Button variant="outline" onClick={() => void fetchPrompts()} className="border-zinc-700">
                Retry
              </Button>
            </div>
          ) : filtered.length === 0 ? (
            <div className="text-center py-16">
              <div className="w-16 h-16 rounded-full bg-zinc-800 flex items-center justify-center mx-auto mb-4">
                <BookOpen className="h-8 w-8 text-zinc-600" />
              </div>
              <h3 className="text-lg font-medium text-zinc-400 mb-2">
                {prompts.length === 0 ? 'No saved prompts yet' : 'No matching prompts'}
              </h3>
              <p className="text-zinc-500 mb-6">
                {prompts.length === 0
                  ? 'Save your best prompts for quick reuse'
                  : 'Try a different search term'}
              </p>
              {prompts.length === 0 && (
                <Button onClick={() => setIsModalOpen(true)} className="bg-blue-600 hover:bg-blue-500">
                  <Plus className="h-4 w-4 mr-2" />
                  Save Prompt
                </Button>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              {filtered.map(prompt => (
                <div
                  key={prompt.id}
                  className="group bg-zinc-900 rounded-lg border border-zinc-800 hover:border-zinc-600 transition-all p-4 cursor-pointer"
                  onClick={() => void handleCopy(prompt)}
                >
                  <div className="flex items-start gap-3">
                    <p className="text-sm text-zinc-200 flex-1 line-clamp-3">{prompt.text}</p>
                    <button
                      className="shrink-0 p-1.5 rounded text-zinc-500 hover:text-white hover:bg-zinc-800 transition-colors"
                    >
                      {copiedId === prompt.id ? (
                        <Check className="h-4 w-4 text-green-400" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </button>
                  </div>

                  <div className="flex items-center gap-2 mt-3">
                    {prompt.tags.map(tag => (
                      <span
                        key={tag}
                        className="text-[10px] bg-blue-500/20 text-blue-400 rounded px-1.5 py-0.5 font-medium"
                      >
                        {tag}
                      </span>
                    ))}
                    <div className="ml-auto flex items-center gap-3 text-[10px] text-zinc-500">
                      <span>Used {prompt.used_count}x</span>
                      <span>Last: {prompt.last_used_at ? formatDate(prompt.last_used_at) : 'Never'}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Save Prompt Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-zinc-900 rounded-xl p-6 w-full max-w-lg border border-zinc-800">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-white">Save Prompt</h2>
              <button onClick={() => setIsModalOpen(false)} className="text-zinc-400 hover:text-white">
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="text-xs text-zinc-500 uppercase tracking-wider font-semibold mb-1.5 block">Prompt</label>
                <textarea
                  value={formText}
                  onChange={e => setFormText(e.target.value)}
                  placeholder="Enter your prompt text..."
                  rows={4}
                  className="w-full px-3 py-2.5 rounded-lg bg-zinc-800 border border-zinc-700 text-white text-sm placeholder:text-zinc-500 focus:outline-none focus:border-blue-500 resize-none"
                  autoFocus
                />
              </div>

              <div>
                <label className="text-xs text-zinc-500 uppercase tracking-wider font-semibold mb-1.5 block">Tags (comma separated)</label>
                <input
                  type="text"
                  value={formTags}
                  onChange={e => setFormTags(e.target.value)}
                  placeholder="e.g. landscape, cinematic, dark"
                  className="w-full px-3 py-2.5 rounded-lg bg-zinc-800 border border-zinc-700 text-white text-sm placeholder:text-zinc-500 focus:outline-none focus:border-blue-500"
                />
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
                disabled={!formText.trim() || saving}
                className="flex-1 bg-blue-600 hover:bg-blue-500"
              >
                {saving ? 'Saving...' : 'Save'}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
