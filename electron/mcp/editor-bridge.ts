import { randomUUID } from 'crypto'
import { getMainWindow } from '../window'
import { handle } from '../ipc/typed-handle'

// Main → renderer RPC for editor tools. The editor's state lives in a zustand
// store inside the renderer, so timeline reads/edits are forwarded there
// ('mcp-editor-request') and answered via the mcpEditorResponse IPC. Going
// through the live store (never writing project files directly) keeps edits
// on the same autosave/undo path as manual edits — no stale-snapshot clobbers.

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> }
const pending = new Map<string, Pending>()

export function registerEditorBridge(): void {
  handle('mcpEditorResponse', ({ id, ok, result, error }) => {
    const p = pending.get(id)
    if (!p) return
    pending.delete(id)
    clearTimeout(p.timer)
    if (ok) p.resolve(result)
    else p.reject(new Error(error || 'Editor request failed'))
  })
}

export function callEditor<T = unknown>(tool: string, args: Record<string, unknown> = {}, timeoutMs = 60000): Promise<T> {
  const win = getMainWindow()
  if (!win || win.isDestroyed()) return Promise.reject(new Error('RiX window is not open'))
  const id = randomUUID()
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id)
      reject(new Error(
        `RiX editor did not answer "${tool}" within ${Math.round(timeoutMs / 1000)}s. ` +
        'Open a project in RiX and visit the Video Editor tab once so the editor is loaded.',
      ))
    }, timeoutMs)
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject, timer })
    win.webContents.send('mcp-editor-request', { id, tool, args })
  })
}
