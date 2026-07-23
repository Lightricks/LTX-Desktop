// Local generation can starve the backend's own event loop for tens of seconds (see
// electron/python-backend.ts) — this stops the liveness monitor from mistaking "busy" for
// "hung" and killing the process mid-generation.
export async function withGenerationActive<T>(fn: () => Promise<T>): Promise<T> {
  void window.electronAPI.notifyGenerationActive({ active: true })
  try {
    return await fn()
  } finally {
    void window.electronAPI.notifyGenerationActive({ active: false })
  }
}
