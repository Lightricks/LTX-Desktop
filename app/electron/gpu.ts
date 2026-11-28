import { execFile } from 'child_process'
import { promisify } from 'util'
import { logger } from './logger'
import { getAuthToken, getBackendUrl, getPythonPath } from './python-backend'
import { PY_REMOVE_CWD_FROM_DLL_SEARCH } from './win-dll-search'

const runPython = promisify(execFile)

// PyTorch exposes ROCm devices through its torch.cuda API.
export async function checkGPU(): Promise<{ available: boolean; name?: string; vram?: number }> {
  try {
    const url = getBackendUrl()
    if (!url) throw new Error('Backend URL not available yet')
    // Try to get GPU info from the backend API first (more reliable)
    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    const token = getAuthToken()
    if (token) headers['Authorization'] = `Bearer ${token}`
    const response = await fetch(`${url}/api/gpu-info`, {
      method: 'GET',
      headers,
    })

    if (response.ok) {
      const data = await response.json()
      return {
        available: data.gpu_available ?? data.cuda_available ?? false,
        name: data.gpu_name,
        vram: data.vram_gb
      }
    }
  } catch (error) {
    logger.warn(`Backend GPU check failed, trying direct check: ${error}`)
  }

  // Fallback: try direct Python check
  try {
    const pythonPath = getPythonPath()
    const probe = [
      PY_REMOVE_CWD_FROM_DLL_SEARCH,
      'import json, sys, torch',
      "gpu = torch.cuda.is_available() and (sys.platform != 'win32' or bool(torch.version.hip))",
      "mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()",
      "name = torch.cuda.get_device_name(0) if gpu else ('Apple Silicon (MPS)' if mps else None)",
      'vram = torch.cuda.get_device_properties(0).total_memory // (1024**3) if gpu else None',
      "print(json.dumps({'available': gpu or mps, 'name': name, 'vram': vram}))",
    ].join('\n')
    const { stdout } = await runPython(pythonPath, ['-c', probe], {
      encoding: 'utf-8',
      timeout: 30000,
      windowsHide: true,
    })
    const info = JSON.parse(stdout.trim())
    return { available: info.available === true, name: info.name ?? undefined, vram: info.vram ?? undefined }
  } catch (error) {
    logger.error(`Direct GPU check also failed: ${error}`)
    return { available: false }
  }
}
