/// <reference types="vite/client" />

import type { ElectronAPI } from '../shared/electron-api-schema'

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
}
