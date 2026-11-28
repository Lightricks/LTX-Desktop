import { forkProductName } from '../../shared/fork-identity.ts'

// A paired browser can run on a different platform from its Desktop host.
const platform = globalThis.window?.electronAPI?.platform
export const desktopProductName = platform ? forkProductName(platform) : 'LTX Desktop'
