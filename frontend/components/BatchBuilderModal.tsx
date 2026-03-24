import { useState, useRef } from 'react'
import { X, Plus, Trash2, Copy, Upload, Grid3X3, List, FileText, Play, AlertCircle } from 'lucide-react'
import type { BatchSubmitRequest, BatchJobItem, SweepAxis } from '@/types/batch'
import { parseCSV, parseJSON, parseRange } from '@/lib/batch-import'
import { useBatch } from '@/hooks/use-batch'

interface BatchBuilderModalProps {
  isOpen: boolean
  onClose: () => void
}

type TabId = 'list' | 'import' | 'grid'

interface ListRow {
  id: string
  type: 'image' | 'video'
  model: string
  prompt: string
  loraPath: string
  loraWeight: number
}

interface GridAxis {
  id: string
  param: string
  valuesInput: string
}

const PARAM_OPTIONS = [
  { value: 'loraWeight', label: 'LoRA Weight' },
  { value: 'loraPath', label: 'LoRA Path' },
  { value: 'prompt', label: 'Prompt' },
  { value: 'numSteps', label: 'Steps' },
  { value: 'seed', label: 'Seed' },
  { value: 'model', label: 'Model' },
]

let rowIdCounter = 0
function nextRowId(): string {
  return `row_${++rowIdCounter}`
}

export function BatchBuilderModal({ isOpen, onClose }: BatchBuilderModalProps) {
  const [activeTab, setActiveTab] = useState<TabId>('list')
  const [target, setTarget] = useState<'local' | 'cloud'>('local')
  const [pipelineEnabled, setPipelineEnabled] = useState(false)
  const batch = useBatch()

  // List tab state
  const [rows, setRows] = useState<ListRow[]>([
    { id: nextRowId(), type: 'image', model: 'flux-klein-9b', prompt: '', loraPath: '', loraWeight: 1.0 },
  ])

  // Import tab state
  const [importText, setImportText] = useState('')
  const [importError, setImportError] = useState<string | null>(null)
  const [importedItems, setImportedItems] = useState<BatchJobItem[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Grid tab state
  const [gridBasePrompt, setGridBasePrompt] = useState('')
  const [gridBaseModel, setGridBaseModel] = useState('z-image-turbo')
  const [gridAxes, setGridAxes] = useState<GridAxis[]>([
    { id: nextRowId(), param: 'loraWeight', valuesInput: '0.3-1.0:8' },
  ])

  if (!isOpen) return null

  const addRow = () => {
    setRows(prev => [...prev, {
      id: nextRowId(), type: 'image', model: 'flux-klein-9b', prompt: '', loraPath: '', loraWeight: 1.0,
    }])
  }

  const removeRow = (id: string) => {
    setRows(prev => prev.filter(r => r.id !== id))
  }

  const duplicateRow = (id: string) => {
    setRows(prev => {
      const idx = prev.findIndex(r => r.id === id)
      if (idx < 0) return prev
      const copy = { ...prev[idx], id: nextRowId() }
      const next = [...prev]
      next.splice(idx + 1, 0, copy)
      return next
    })
  }

  const updateRow = (id: string, field: keyof ListRow, value: string | number) => {
    setRows(prev => prev.map(r => r.id === id ? { ...r, [field]: value } : r))
  }

  const handleImportParse = (text: string) => {
    setImportText(text)
    setImportError(null)
    setImportedItems([])
    if (!text.trim()) return
    try {
      const items = text.trim().startsWith('{') || text.trim().startsWith('[')
        ? parseJSON(text)
        : parseCSV(text)
      setImportedItems(items)
    } catch (err) {
      setImportError(err instanceof Error ? err.message : 'Parse error')
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => handleImportParse(reader.result as string)
    reader.readAsText(file)
  }

  const addAxis = () => {
    if (gridAxes.length >= 3) return
    setGridAxes(prev => [...prev, { id: nextRowId(), param: 'numSteps', valuesInput: '4, 8, 12' }])
  }

  const removeAxis = (id: string) => {
    setGridAxes(prev => prev.filter(a => a.id !== id))
  }

  const updateAxis = (id: string, field: keyof GridAxis, value: string) => {
    setGridAxes(prev => prev.map(a => a.id === id ? { ...a, [field]: value } : a))
  }

  const getGridTotalJobs = (): number => {
    return gridAxes.reduce((total, axis) => {
      const values = parseRange(axis.valuesInput)
      return total * Math.max(values.length, 1)
    }, 1)
  }

  const handleSubmit = async () => {
    let request: BatchSubmitRequest

    if (activeTab === 'list') {
      const jobs: BatchJobItem[] = rows.filter(r => r.prompt.trim()).map(r => ({
        type: r.type,
        model: r.model,
        params: {
          prompt: r.prompt,
          ...(r.loraPath ? { loraPath: r.loraPath, loraWeight: r.loraWeight } : {}),
        },
      }))
      if (pipelineEnabled) {
        request = {
          mode: 'pipeline',
          target,
          pipeline: {
            steps: jobs.flatMap(j => [
              { type: 'image' as const, model: j.model, params: j.params, auto_prompt: false },
              { type: 'video' as const, model: 'ltx-fast', params: {}, auto_prompt: true },
            ]),
          },
        }
      } else {
        request = { mode: 'list', target, jobs }
      }
    } else if (activeTab === 'import') {
      request = { mode: 'list', target, jobs: importedItems }
    } else {
      const axes: SweepAxis[] = gridAxes.map(a => ({
        param: a.param,
        values: a.param === 'prompt'
          ? a.valuesInput.split(',').map(v => v.trim())
          : parseRange(a.valuesInput),
        mode: a.param === 'prompt' ? 'search_replace' as const : 'replace' as const,
        ...(a.param === 'prompt' ? { search: gridBasePrompt.split(' ')[0] } : {}),
      }))
      request = {
        mode: 'sweep',
        target,
        sweep: {
          base_type: 'image',
          base_model: gridBaseModel,
          base_params: { prompt: gridBasePrompt },
          axes,
        },
      }
    }

    await batch.submit(request)
    onClose()
  }

  const tabs: { id: TabId; label: string; icon: React.ReactNode }[] = [
    { id: 'list', label: 'List', icon: <List className="w-4 h-4" /> },
    { id: 'import', label: 'Import', icon: <FileText className="w-4 h-4" /> },
    { id: 'grid', label: 'Grid Sweep', icon: <Grid3X3 className="w-4 h-4" /> },
  ]

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div
        className="w-[900px] max-h-[85vh] rounded-xl border flex flex-col overflow-hidden"
        style={{
          background: 'oklch(0.18 0.02 290)',
          borderColor: 'oklch(0.32 0.03 290)',
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: 'oklch(0.32 0.03 290)' }}>
          <h2 className="text-lg font-semibold" style={{ color: 'oklch(0.92 0.02 290)' }}>Batch Generation</h2>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/10 transition-colors">
            <X className="w-5 h-5" style={{ color: 'oklch(0.65 0.04 290)' }} />
          </button>
        </div>

        {/* Tabs + Target */}
        <div className="flex items-center justify-between px-6 py-3 border-b" style={{ borderColor: 'oklch(0.32 0.03 290)' }}>
          <div className="flex gap-1">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors"
                style={{
                  background: activeTab === tab.id ? 'oklch(0.6 0.2 290 / 0.2)' : 'transparent',
                  color: activeTab === tab.id ? 'oklch(0.75 0.15 290)' : 'oklch(0.65 0.04 290)',
                }}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs" style={{ color: 'oklch(0.65 0.04 290)' }}>Target:</span>
            <select
              value={target}
              onChange={e => setTarget(e.target.value as 'local' | 'cloud')}
              className="text-sm rounded-lg px-2 py-1 border"
              style={{
                background: 'oklch(0.22 0.025 290)',
                borderColor: 'oklch(0.32 0.03 290)',
                color: 'oklch(0.92 0.02 290)',
              }}
            >
              <option value="local">Local GPU</option>
              <option value="cloud">Cloud API</option>
            </select>
          </div>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {activeTab === 'list' && (
            <div className="space-y-3">
              {/* Table header */}
              <div className="grid grid-cols-[80px_120px_1fr_180px_80px_60px] gap-2 text-xs font-medium" style={{ color: 'oklch(0.65 0.04 290)' }}>
                <span>Type</span>
                <span>Model</span>
                <span>Prompt</span>
                <span>LoRA Path</span>
                <span>LoRA Wt</span>
                <span></span>
              </div>
              {rows.map(row => (
                <div key={row.id} className="grid grid-cols-[80px_120px_1fr_180px_80px_60px] gap-2 items-center">
                  <select
                    value={row.type}
                    onChange={e => updateRow(row.id, 'type', e.target.value)}
                    className="text-sm rounded-lg px-2 py-1.5 border"
                    style={{ background: 'oklch(0.22 0.025 290)', borderColor: 'oklch(0.32 0.03 290)', color: 'oklch(0.92 0.02 290)' }}
                  >
                    <option value="image">Image</option>
                    <option value="video">Video</option>
                  </select>
                  <input
                    value={row.model}
                    onChange={e => updateRow(row.id, 'model', e.target.value)}
                    className="text-sm rounded-lg px-2 py-1.5 border"
                    style={{ background: 'oklch(0.22 0.025 290)', borderColor: 'oklch(0.32 0.03 290)', color: 'oklch(0.92 0.02 290)' }}
                  />
                  <input
                    value={row.prompt}
                    onChange={e => updateRow(row.id, 'prompt', e.target.value)}
                    placeholder="Enter prompt..."
                    className="text-sm rounded-lg px-2 py-1.5 border"
                    style={{ background: 'oklch(0.22 0.025 290)', borderColor: 'oklch(0.32 0.03 290)', color: 'oklch(0.92 0.02 290)' }}
                  />
                  <input
                    value={row.loraPath}
                    onChange={e => updateRow(row.id, 'loraPath', e.target.value)}
                    placeholder="Optional"
                    className="text-sm rounded-lg px-2 py-1.5 border"
                    style={{ background: 'oklch(0.22 0.025 290)', borderColor: 'oklch(0.32 0.03 290)', color: 'oklch(0.92 0.02 290)' }}
                  />
                  <input
                    type="number"
                    value={row.loraWeight}
                    onChange={e => updateRow(row.id, 'loraWeight', Number(e.target.value))}
                    step={0.1}
                    min={0}
                    max={2}
                    className="text-sm rounded-lg px-2 py-1.5 border"
                    style={{ background: 'oklch(0.22 0.025 290)', borderColor: 'oklch(0.32 0.03 290)', color: 'oklch(0.92 0.02 290)' }}
                  />
                  <div className="flex gap-1">
                    <button onClick={() => duplicateRow(row.id)} className="p-1 rounded hover:bg-white/10" title="Duplicate">
                      <Copy className="w-3.5 h-3.5" style={{ color: 'oklch(0.65 0.04 290)' }} />
                    </button>
                    <button onClick={() => removeRow(row.id)} className="p-1 rounded hover:bg-white/10" title="Remove">
                      <Trash2 className="w-3.5 h-3.5" style={{ color: 'oklch(0.65 0.04 290)' }} />
                    </button>
                  </div>
                </div>
              ))}
              <button
                onClick={addRow}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors hover:bg-white/10"
                style={{ color: 'oklch(0.75 0.15 290)' }}
              >
                <Plus className="w-4 h-4" />
                Add Row
              </button>

              {/* Pipeline toggle */}
              <label className="flex items-center gap-2 pt-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={pipelineEnabled}
                  onChange={e => setPipelineEnabled(e.target.checked)}
                  className="rounded"
                />
                <span className="text-sm" style={{ color: 'oklch(0.75 0.04 290)' }}>
                  Also generate video from each image (i2v pipeline)
                </span>
              </label>
            </div>
          )}

          {activeTab === 'import' && (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm border transition-colors hover:bg-white/10"
                  style={{ borderColor: 'oklch(0.32 0.03 290)', color: 'oklch(0.75 0.15 290)' }}
                >
                  <Upload className="w-4 h-4" />
                  Upload CSV/JSON
                </button>
                <input ref={fileInputRef} type="file" accept=".csv,.json" className="hidden" onChange={handleFileUpload} />
                <span className="text-xs" style={{ color: 'oklch(0.65 0.04 290)' }}>
                  Or paste below
                </span>
              </div>
              <textarea
                value={importText}
                onChange={e => handleImportParse(e.target.value)}
                placeholder={'prompt,type,model,loraWeight\n"a cute cat",image,flux-klein-9b,0.8\n"a happy dog",image,flux-klein-9b,1.0'}
                rows={8}
                className="w-full text-sm rounded-lg px-3 py-2 border font-mono resize-none"
                style={{ background: 'oklch(0.22 0.025 290)', borderColor: 'oklch(0.32 0.03 290)', color: 'oklch(0.92 0.02 290)' }}
              />
              {importError && (
                <div className="flex items-center gap-2 text-sm px-3 py-2 rounded-lg" style={{ background: 'oklch(0.25 0.08 25)', color: 'oklch(0.75 0.15 25)' }}>
                  <AlertCircle className="w-4 h-4" />
                  {importError}
                </div>
              )}
              {importedItems.length > 0 && (
                <div className="text-sm" style={{ color: 'oklch(0.75 0.15 290)' }}>
                  Parsed {importedItems.length} job{importedItems.length !== 1 ? 's' : ''}
                </div>
              )}
            </div>
          )}

          {activeTab === 'grid' && (
            <div className="space-y-4">
              {/* Base settings */}
              <div className="space-y-2">
                <label className="text-sm font-medium" style={{ color: 'oklch(0.75 0.04 290)' }}>Base Prompt</label>
                <input
                  value={gridBasePrompt}
                  onChange={e => setGridBasePrompt(e.target.value)}
                  placeholder="Enter base prompt..."
                  className="w-full text-sm rounded-lg px-3 py-2 border"
                  style={{ background: 'oklch(0.22 0.025 290)', borderColor: 'oklch(0.32 0.03 290)', color: 'oklch(0.92 0.02 290)' }}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium" style={{ color: 'oklch(0.75 0.04 290)' }}>Model</label>
                <input
                  value={gridBaseModel}
                  onChange={e => setGridBaseModel(e.target.value)}
                  className="w-48 text-sm rounded-lg px-3 py-2 border"
                  style={{ background: 'oklch(0.22 0.025 290)', borderColor: 'oklch(0.32 0.03 290)', color: 'oklch(0.92 0.02 290)' }}
                />
              </div>

              {/* Axes */}
              <div className="space-y-3">
                <label className="text-sm font-medium" style={{ color: 'oklch(0.75 0.04 290)' }}>Sweep Axes</label>
                {gridAxes.map(axis => (
                  <div key={axis.id} className="flex items-center gap-2">
                    <select
                      value={axis.param}
                      onChange={e => updateAxis(axis.id, 'param', e.target.value)}
                      className="text-sm rounded-lg px-2 py-1.5 border w-40"
                      style={{ background: 'oklch(0.22 0.025 290)', borderColor: 'oklch(0.32 0.03 290)', color: 'oklch(0.92 0.02 290)' }}
                    >
                      {PARAM_OPTIONS.map(opt => (
                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                      ))}
                    </select>
                    <input
                      value={axis.valuesInput}
                      onChange={e => updateAxis(axis.id, 'valuesInput', e.target.value)}
                      placeholder="0.3-1.0:8 or 4, 8, 12"
                      className="flex-1 text-sm rounded-lg px-2 py-1.5 border"
                      style={{ background: 'oklch(0.22 0.025 290)', borderColor: 'oklch(0.32 0.03 290)', color: 'oklch(0.92 0.02 290)' }}
                    />
                    <button onClick={() => removeAxis(axis.id)} className="p-1 rounded hover:bg-white/10">
                      <Trash2 className="w-3.5 h-3.5" style={{ color: 'oklch(0.65 0.04 290)' }} />
                    </button>
                  </div>
                ))}
                {gridAxes.length < 3 && (
                  <button
                    onClick={addAxis}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors hover:bg-white/10"
                    style={{ color: 'oklch(0.75 0.15 290)' }}
                  >
                    <Plus className="w-4 h-4" />
                    Add Axis
                  </button>
                )}
              </div>

              {/* Preview */}
              <div
                className="px-3 py-2 rounded-lg text-sm"
                style={{ background: 'oklch(0.22 0.025 290)', color: 'oklch(0.75 0.15 290)' }}
              >
                {gridAxes.map(a => `${a.param}: ${parseRange(a.valuesInput).length} values`).join(' x ')} = <strong>{getGridTotalJobs()} jobs</strong>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t" style={{ borderColor: 'oklch(0.32 0.03 290)' }}>
          <div className="text-sm" style={{ color: 'oklch(0.65 0.04 290)' }}>
            {activeTab === 'list' && `${rows.filter(r => r.prompt.trim()).length} job${rows.filter(r => r.prompt.trim()).length !== 1 ? 's' : ''}${pipelineEnabled ? ' (x2 with pipeline)' : ''}`}
            {activeTab === 'import' && `${importedItems.length} job${importedItems.length !== 1 ? 's' : ''}`}
            {activeTab === 'grid' && `${getGridTotalJobs()} jobs`}
          </div>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-sm transition-colors hover:bg-white/10"
              style={{ color: 'oklch(0.65 0.04 290)' }}
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
              style={{ background: 'oklch(0.6 0.2 290)', color: 'oklch(0.98 0.01 290)' }}
            >
              <Play className="w-4 h-4" />
              Generate Batch
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
