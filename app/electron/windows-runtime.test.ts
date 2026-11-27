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
  const inherited = 'HSA_OVERRIDE_GFX_VERSION_2'
  const oldValue = process.env[inherited]
  process.env[inherited] = '11.5.1'
  t.after(() => {
    if (oldValue === undefined) delete process.env[inherited]
    else process.env[inherited] = oldValue
  })
  for (const model of ['8040S', '8050S', '8060S']) {
    const receipt = { version: 1, backend: 'rocm', deviceName: `AMD Radeon ${model}`, hipOrdinal: 2 }
    fs.writeFileSync(path.join(dir, 'gpu-runtime.json'), JSON.stringify(receipt))
    assert.deepEqual(readWindowsRuntime(dir), receipt)
    assert.equal(windowsRuntimeReady(dir), true)
    const env = windowsRuntimeEnvironment(readWindowsRuntime(dir))
    assert.equal(env.HIP_VISIBLE_DEVICES, '2')
    assert.equal(env.CUDA_VISIBLE_DEVICES, '2')
    assert.equal(env.LTX_GPU_RUNTIME, 'rocm')
    assert.equal(Object.hasOwn(env, inherited), true)
    assert.equal(env[inherited], undefined)
  }
  fs.writeFileSync(path.join(dir, 'gpu-runtime.json'), '{"backend":"rocm","version":1}')
  assert.equal(windowsRuntimeReady(dir), false)
})
