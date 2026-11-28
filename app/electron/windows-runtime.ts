import fs from 'node:fs'
import path from 'node:path'

type WindowsRuntime =
  | { version: 1; backend: 'rocm'; deviceName: string; hipOrdinal?: number }
  | { version: 1; backend: 'cuda'; deviceName: string; cudaUuid: string }
  | { version: 1; backend: 'api'; deviceName: string }

const cudaId = /^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

export function readWindowsRuntime(resourcesDir: string): WindowsRuntime {
  const receipt = path.join(resourcesDir, 'gpu-runtime.json')
  if (!fs.existsSync(receipt)) {
    // AMD-only installations before the universal installer recorded readiness in Python.
    if (fs.existsSync(path.join(resourcesDir, 'python', 'ltx-amd-runtime.json'))) {
      return { version: 1, backend: 'rocm', deviceName: 'AMD Radeon' }
    }
    throw new Error('GPU runtime selection is missing. Run the SpaceSpeak Windows installer again.')
  }
  const item = JSON.parse(fs.readFileSync(receipt, 'utf8'))
  if (item?.version === 1 && typeof item.deviceName === 'string') {
    if (item.backend === 'api') return item
    if (item.backend === 'cuda' && typeof item.cudaUuid === 'string' && cudaId.test(item.cudaUuid)) return item
    if (item.backend === 'rocm' && Number.isSafeInteger(item.hipOrdinal) && item.hipOrdinal >= 0) return item
  }
  throw new Error('Invalid GPU runtime selection. Run the SpaceSpeak Windows installer again.')
}

export function windowsRuntimeReady(resourcesDir: string): boolean {
  try {
    const runtime = readWindowsRuntime(resourcesDir)
    const pythonDir = path.join(resourcesDir, 'python')
    return fs.existsSync(path.join(pythonDir, 'python.exe'))
      && (runtime.backend !== 'rocm' || fs.existsSync(path.join(pythonDir, 'ltx-amd-runtime.json')))
  } catch {
    return false
  }
}

export function windowsRuntimeEnvironment(runtime: WindowsRuntime): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {
    LTX_GPU_RUNTIME: runtime.backend,
    LTX_REQUIRE_ROCM: runtime.backend === 'rocm' ? '1' : '0',
  }
  if (runtime.backend !== 'rocm' || runtime.hipOrdinal !== undefined) {
    env.ROCR_VISIBLE_DEVICES = undefined
    env.HIP_VISIBLE_DEVICES = undefined
    env.GPU_DEVICE_ORDINAL = undefined
  }
  if (runtime.backend === 'cuda') env.CUDA_VISIBLE_DEVICES = runtime.cudaUuid
  if (runtime.backend === 'api') env.CUDA_VISIBLE_DEVICES = '-1'
  if (runtime.backend === 'rocm' && runtime.hipOrdinal !== undefined) {
    for (const key of Object.keys(process.env)) {
      if (/^HSA_OVERRIDE_GFX_VERSION(?:_\d+)?$/i.test(key)) env[key] = undefined
    }
    env.HIP_VISIBLE_DEVICES = String(runtime.hipOrdinal)
    env.CUDA_VISIBLE_DEVICES = String(runtime.hipOrdinal)
  }
  return env
}
