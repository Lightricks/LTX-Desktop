import { createRequire } from 'node:module'

/** Python one-liner: strip CWD from the Windows DLL search path. No-op elsewhere. */
export const PY_REMOVE_CWD_FROM_DLL_SEARCH =
  "import sys;(sys.platform=='win32')and __import__('ctypes').WinDLL('kernel32',use_last_error=True).SetDllDirectoryW('');"

export function removeCwdFromDllSearchPath(): void {
  if (process.platform !== 'win32') {
    return
  }

  const require = createRequire(import.meta.url)
  const koffi = require('koffi') as {
    load: (name: string) => { func: (sig: string) => (...args: unknown[]) => unknown }
  }
  const kernel32 = koffi.load('kernel32.dll')
  const setDllDirectoryW = kernel32.func('int SetDllDirectoryW(str16)')
  const ok = setDllDirectoryW('')
  if (!ok) {
    console.error('[LTX Desktop] SetDllDirectoryW("") failed')
  }
}

removeCwdFromDllSearchPath()
