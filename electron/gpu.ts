import { execSync } from 'child_process'
import { logger } from './logger'
import { getAuthToken, getBackendUrl, getPythonPath } from './python-backend'

// Check if NVIDIA GPU is available
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
    const result = execSync(`"${pythonPath}" -c "import torch; cuda=torch.cuda.is_available(); mps=hasattr(torch.backends,'mps') and torch.backends.mps.is_available(); print(cuda or mps); print(torch.cuda.get_device_name(0) if cuda else ('Apple Silicon (MPS)' if mps else '')); print(torch.cuda.get_device_properties(0).total_memory // (1024**3) if cuda else 0)"`, {
      encoding: 'utf-8',
      timeout: 30000,
      windowsHide: true
    }).trim().split('\n')

    return {
      available: result[0] === 'True',
      name: result[1] || undefined,
      vram: parseInt(result[2]) || undefined
    }
  } catch (error) {
    logger.error(`Direct GPU check also failed: ${error}`)
    return { available: false }
  }
}
