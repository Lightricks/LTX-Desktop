import type { BatchJobItem } from '@/types/batch'

function parseCSVLine(line: string): string[] {
  const result: string[] = []
  let current = ''
  let inQuotes = false
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"'
        i++
      } else {
        inQuotes = !inQuotes
      }
    } else if (ch === ',' && !inQuotes) {
      result.push(current)
      current = ''
    } else {
      current += ch
    }
  }
  result.push(current)
  return result
}

function inferType(value: string): unknown {
  if (/^-?\d+(\.\d+)?$/.test(value)) return Number(value)
  if (value.toLowerCase() === 'true') return true
  if (value.toLowerCase() === 'false') return false
  return value
}

export function parseCSV(text: string): BatchJobItem[] {
  const lines = text.trim().split('\n')
  if (lines.length < 2) return []
  const headers = lines[0].split(',').map(h => h.trim().toLowerCase())
  const promptIdx = headers.indexOf('prompt')
  if (promptIdx === -1) throw new Error('CSV must have a "prompt" column')

  return lines.slice(1).filter(l => l.trim()).map(line => {
    const cols = parseCSVLine(line)
    const params: Record<string, unknown> = {}
    headers.forEach((h, i) => {
      if (h !== 'type' && h !== 'model' && cols[i]?.trim()) {
        params[h] = inferType(cols[i].trim())
      }
    })
    return {
      type: (cols[headers.indexOf('type')]?.trim() as 'video' | 'image') || 'image',
      model: cols[headers.indexOf('model')]?.trim() || 'zit',
      params,
    }
  })
}

export function parseJSON(text: string): BatchJobItem[] {
  const data = JSON.parse(text)
  const defaults = data.defaults || {}
  return (data.jobs || []).map((job: Record<string, unknown>) => ({
    type: job.type || defaults.type || 'image',
    model: job.model || defaults.model || 'zit',
    params: { ...defaults, ...(job.params as Record<string, unknown> || {}), prompt: job.prompt || '' },
  }))
}

export function parseRange(input: string): number[] {
  const match = input.match(/^([\d.]+)-([\d.]+):(\d+)$/)
  if (!match) {
    return input.split(',').map(v => Number(v.trim())).filter(n => !isNaN(n))
  }
  const start = Number(match[1])
  const end = Number(match[2])
  const count = Number(match[3])
  if (count < 2) return [start]
  const step = (end - start) / (count - 1)
  return Array.from({ length: count }, (_, i) => Math.round((start + i * step) * 1000) / 1000)
}
