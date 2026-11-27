import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'node:test'
import { readWindowsRuntime, windowsRuntimeEnvironment, windowsRuntimeReady } from './windows-runtime.ts'

test('NVIDIA uses the selected UUID and does not require a ROCm marker', (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ltx-gpu-'))
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const cudaUuid = 'GPU-e2bac309-278a-3ded-85ac-4c3ae8b708f8'
  fs.mkdirSync(path.join(dir, 'python'))
  fs.writeFileSync(path.join(dir, 'python/python.exe'), '')
  fs.writeFileSync(path.join(dir, 'gpu-runtime.json'), JSON.stringify({ version: 1, backend: 'cuda', deviceName: 'RTX', cudaUuid }))
  assert.equal(windowsRuntimeReady(dir), true)
  const env = windowsRuntimeEnvironment(readWindowsRuntime(dir))
  assert.equal(env.CUDA_VISIBLE_DEVICES, cudaUuid)
  assert.equal(env.LTX_REQUIRE_ROCM, '0')
  fs.writeFileSync(path.join(dir, 'gpu-runtime.json'), JSON.stringify({ version: 1, backend: 'api', deviceName: 'none' }))
  assert.equal(windowsRuntimeEnvironment(readWindowsRuntime(dir)).CUDA_VISIBLE_DEVICES, '-1')
})

test('legacy AMD marker, explicit HIP device and invalid receipts', (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ltx-gpu-'))
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  assert.equal(windowsRuntimeReady(dir), false)
  fs.mkdirSync(path.join(dir, 'python'))
  fs.writeFileSync(path.join(dir, 'python/python.exe'), '')
  fs.writeFileSync(path.join(dir, 'python/ltx-amd-runtime.json'), '{}')
  assert.equal(windowsRuntimeReady(dir), true)
  assert.equal(windowsRuntimeEnvironment(readWindowsRuntime(dir)).LTX_REQUIRE_ROCM, '1')
  fs.writeFileSync(path.join(dir, 'gpu-runtime.json'), JSON.stringify({ version: 1, backend: 'rocm', deviceName: 'Radeon', hipOrdinal: 1 }))
  assert.equal(windowsRuntimeEnvironment(readWindowsRuntime(dir)).HIP_VISIBLE_DEVICES, '1')
  fs.writeFileSync(path.join(dir, 'gpu-runtime.json'), '{"backend":"rocm","version":1}')
  assert.equal(windowsRuntimeReady(dir), false)
})
