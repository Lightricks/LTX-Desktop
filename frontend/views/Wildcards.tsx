import { useState, useEffect, useCallback } from 'react'
import { ArrowLeft, Plus, Pencil, Trash2, Braces, X, Shuffle, ChevronDown, ChevronRight } from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import { LtxLogo } from '../components/LtxLogo'
import { Button } from '../components/ui/button'
import { logger } from '../lib/logger'

interface Wildcard {
  id: string
  name: string
  values: string[]
  created_at: string
}

export function Wildcards() {
  const { goHome } = useProjects()
  const [wildcards, setWildcards] = useState<Wildcard[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [editingWildcard, setEditingWildcard] = useState<Wildcard | null>(null)
  const [formName, setFormName] = useState('')
  const [formValues, setFormValues] = useState('')
  const [saving, setSaving] = useState(false)
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())
  const [testPrompt, setTestPrompt] = useState('')
  const [testResult, setTestResult] = useState('')

  const fetchWildcards = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const res = await fetch(`${backendUrl}/api/wildcards`)
      if (!res.ok) throw new Error(`Failed to fetch wildcards: ${res.status}`)
      const data = (await res.json()) as { wildcards: Wildcard[] }
      setWildcards(data.wildcards ?? [])
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to load wildcards'
      logger.error(msg)
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchWildcards()
  }, [fetchWildcards])

  const openCreate = () => {
    setEditingWildcard(null)
    setFormName('')
    setFormValues('')
    setIsModalOpen(true)
  }

  const openEdit = (wc: Wildcard) => {
    setEditingWildcard(wc)
    setFormName(wc.name)
    setFormValues(wc.values.join('\n'))
    setIsModalOpen(true)
  }

  const handleSave = async () => {
    if (!formName.trim() || !formValues.trim()) return
    setSaving(true)
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const values = formValues.split('\n').map(v => v.trim()).filter(Boolean)
      const body = { name: formName.trim(), values }
      if (editingWildcard) {
        const res = await fetch(`${backendUrl}/api/wildcards/${editingWildcard.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        if (!res.ok) throw new Error(`Update failed: ${res.status}`)
      } else {
        const res = await fetch(`${backendUrl}/api/wildcards`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        if (!res.ok) throw new Error(`Create failed: ${res.status}`)
      }
      setIsModalOpen(false)
      void fetchWildcards()
    } catch (e) {
      logger.error(`Failed to save wildcard: ${e}`)
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (wc: Wildcard) => {
    if (!confirm(`Delete wildcard "_${wc.name}_"?`)) return
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const res = await fetch(`${backendUrl}/api/wildcards/${wc.id}`, { method: 'DELETE' })
      if (!res.ok) throw new Error(`Delete failed: ${res.status}`)
      setWildcards(prev => prev.filter(w => w.id !== wc.id))
    } catch (e) {
      logger.error(`Failed to delete wildcard: ${e}`)
    }
  }

  const toggleExpanded = (id: string) => {
    setExpandedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const expandTestPrompt = () => {
    if (!testPrompt.trim()) return
    let result = testPrompt
    for (const wc of wildcards) {
      const pattern = new RegExp(`_${wc.name}_`, 'g')
      result = result.replace(pattern, () => {
        const idx = Math.floor(Math.random() * wc.values.length)
        return wc.values[idx] ?? wc.name
      })
    }
    setTestResult(result)
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
        <h1 className="text-lg font-semibold text-white">Wildcards</h1>
        <div className="ml-auto">
          <Button onClick={openCreate} className="bg-blue-600 hover:bg-blue-500" size="sm">
            <Plus className="h-3.5 w-3.5 mr-1.5" />
            Add Wildcard
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
              <Button variant="outline" onClick={() => void fetchWildcards()} className="border-zinc-700">
                Retry
              </Button>
            </div>
          ) : (
            <>
              {/* Wildcard list */}
              {wildcards.length === 0 ? (
                <div className="text-center py-16">
                  <div className="w-16 h-16 rounded-full bg-zinc-800 flex items-center justify-center mx-auto mb-4">
                    <Braces className="h-8 w-8 text-zinc-600" />
                  </div>
                  <h3 className="text-lg font-medium text-zinc-400 mb-2">No wildcards yet</h3>
                  <p className="text-zinc-500 mb-6">Create wildcards to randomize parts of your prompts</p>
                  <Button onClick={openCreate} className="bg-blue-600 hover:bg-blue-500">
                    <Plus className="h-4 w-4 mr-2" />
                    Add Wildcard
                  </Button>
                </div>
              ) : (
                <div className="space-y-2 mb-8">
                  {wildcards.map(wc => {
                    const isExpanded = expandedIds.has(wc.id)
                    return (
                      <div
                        key={wc.id}
                        className="bg-zinc-900 rounded-lg border border-zinc-800 hover:border-zinc-700 transition-colors"
                      >
                        <div className="flex items-center px-4 py-3">
                          <button
                            onClick={() => toggleExpanded(wc.id)}
                            className="mr-2 text-zinc-500 hover:text-white transition-colors"
                          >
                            {isExpanded ? (
                              <ChevronDown className="h-4 w-4" />
                            ) : (
                              <ChevronRight className="h-4 w-4" />
                            )}
                          </button>
                          <code className="text-sm font-mono text-blue-400 font-medium">_{wc.name}_</code>
                          <span className="text-xs text-zinc-500 ml-3">{wc.values.length} values</span>
                          <div className="ml-auto flex items-center gap-1">
                            <button
                              onClick={() => openEdit(wc)}
                              className="p-1.5 rounded text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
                            >
                              <Pencil className="h-3.5 w-3.5" />
                            </button>
                            <button
                              onClick={() => void handleDelete(wc)}
                              className="p-1.5 rounded text-zinc-400 hover:text-red-400 hover:bg-zinc-800 transition-colors"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        </div>
                        {isExpanded && (
                          <div className="px-4 pb-3 pt-0">
                            <div className="flex flex-wrap gap-1.5 pl-6">
                              {wc.values.map((v, i) => (
                                <span
                                  key={i}
                                  className="text-xs bg-zinc-800 text-zinc-300 rounded px-2 py-1 border border-zinc-700"
                                >
                                  {v}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}

              {/* Test area */}
              {wildcards.length > 0 && (
                <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
                  <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                    <Shuffle className="h-4 w-4 text-zinc-400" />
                    Test Wildcards
                  </h3>
                  <textarea
                    value={testPrompt}
                    onChange={e => setTestPrompt(e.target.value)}
                    placeholder="Type a prompt with _wildcard_names_ to test..."
                    rows={2}
                    className="w-full px-3 py-2.5 rounded-lg bg-zinc-800 border border-zinc-700 text-white text-sm placeholder:text-zinc-500 focus:outline-none focus:border-blue-500 resize-none mb-3"
                  />
                  <div className="flex items-center gap-3">
                    <Button
                      onClick={expandTestPrompt}
                      variant="outline"
                      size="sm"
                      className="border-zinc-700"
                      disabled={!testPrompt.trim()}
                    >
                      <Shuffle className="h-3.5 w-3.5 mr-1.5" />
                      Expand
                    </Button>
                    {testResult && (
                      <p className="text-sm text-green-400 flex-1">{testResult}</p>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Create/Edit Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-zinc-900 rounded-xl p-6 w-full max-w-md border border-zinc-800">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-white">
                {editingWildcard ? 'Edit Wildcard' : 'Add Wildcard'}
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
                  placeholder="e.g. color, mood, setting"
                  className="w-full px-3 py-2.5 rounded-lg bg-zinc-800 border border-zinc-700 text-white text-sm placeholder:text-zinc-500 focus:outline-none focus:border-blue-500"
                  autoFocus
                />
                <p className="text-[10px] text-zinc-600 mt-1">
                  Use as <code className="text-blue-400">_{formName || 'name'}_</code> in prompts
                </p>
              </div>

              <div>
                <label className="text-xs text-zinc-500 uppercase tracking-wider font-semibold mb-1.5 block">Values (one per line)</label>
                <textarea
                  value={formValues}
                  onChange={e => setFormValues(e.target.value)}
                  placeholder={"red\nblue\ngreen\ngolden"}
                  rows={6}
                  className="w-full px-3 py-2.5 rounded-lg bg-zinc-800 border border-zinc-700 text-white text-sm font-mono placeholder:text-zinc-500 focus:outline-none focus:border-blue-500 resize-none"
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
                disabled={!formName.trim() || !formValues.trim() || saving}
                className="flex-1 bg-blue-600 hover:bg-blue-500"
              >
                {saving ? 'Saving...' : editingWildcard ? 'Update' : 'Create'}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
