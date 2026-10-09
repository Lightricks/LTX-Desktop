/** No-op. Vite aliases this module to a local overlay when present. */
export function attachOptionalPreload(
  _api: Record<string, unknown>,
  _ipcRenderer: { invoke: (channel: string, input?: unknown) => Promise<unknown> },
): void {}
