/// <reference types="vite/client" />
/// <reference types="vite-plugin-svgr/client" />

import type { ElectronAPI } from '../shared/electron-api-schema'
import type {
  RollbackProjectReferencesMigrationOptions,
  RollbackProjectReferencesMigrationResult,
} from './lib/project-storage-devtools'

declare global {
  interface Window {
    electronAPI: ElectronAPI
    __ltxProjectStorageDebug?: {
      rollbackProjectReferencesMigration: (
        options?: RollbackProjectReferencesMigrationOptions
      ) => RollbackProjectReferencesMigrationResult
    }
  }
}
