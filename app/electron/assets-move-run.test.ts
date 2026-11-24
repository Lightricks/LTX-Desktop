import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, beforeEach, describe, it } from 'node:test'
import { runMove, type MoveDeps, type MoveRequest, type MoveStatePatch } from './assets-move-run.ts'
import { ASSETS_MARKER_FILENAME, preflightMove } from './assets-relocation.ts'

const A = '11111111-1111-4111-8111-111111111111'
const B = '22222222-2222-4222-8222-222222222222'

let root: string
let from: string
let to: string

beforeEach(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'ltx-move-'))
  from = path.join(root, 'old')
  to = path.join(root, 'new')
  fs.mkdirSync(from)
  fs.mkdirSync(to)
  fs.writeFileSync(path.join(from, `${A}.png`), 'hello')
  fs.writeFileSync(path.join(from, `${A}-thumb.jpg`), 'th')
  fs.writeFileSync(path.join(from, 'notes.txt'), 'mine')
})

afterEach(() => {
  fs.rmSync(root, { recursive: true, force: true })
})

const replan = () =>
  preflightMove({ from, to, protectedDirs: [], blockedRoots: [], appFolders: [], freeBytes: async () => 1_000_000 })

async function request(overrides: Partial<MoveRequest> = {}): Promise<MoveRequest> {
  const first = await replan()
  assert.equal(first.ok, true)
  if (!first.ok) throw new Error(first.reason)
  return { plan: first.plan, toCustom: true, previousDir: undefined, replan, ...overrides }
}

function fakes(overrides: Partial<MoveDeps> = {}) {
  const calls: string[] = []
  const phases: MoveStatePatch[] = []
  const deps: MoveDeps = {
    stopBackend: async () => void calls.push('stop'),
    startBackend: async () => void calls.push('start'),
    saveAssetsDir: (dir) => void calls.push(`save:${dir ?? 'default'}`),
    emit: (patch) => void phases.push(patch),
    isCancelled: () => false,
    logError: () => {},
    ...overrides,
  }
  return { deps, calls, phases }
}

const list = (dir: string) => fs.readdirSync(dir).sort()

describe('runMove', () => {
  it('copies, saves the folder, restarts, and deletes only the asset files in the old folder', async () => {
    const { deps, calls, phases } = fakes()

    const result = await runMove(await request(), deps)

    assert.equal(result, 'moved')
    assert.deepEqual(calls, ['stop', 'save:' + to, 'start'])
    assert.deepEqual(list(to), [ASSETS_MARKER_FILENAME, `${A}-thumb.jpg`, `${A}.png`].sort())
    assert.deepEqual(list(from), ['notes.txt'])
    assert.equal(phases[phases.length - 1].result?.ok, true)
  })

  it('copies a file that was added after the confirmation', async () => {
    const { deps } = fakes()
    const req = await request()
    fs.writeFileSync(path.join(from, `${B}.png`), 'late')

    await runMove(req, deps)

    assert.equal(fs.readFileSync(path.join(to, `${B}.png`), 'utf8'), 'late')
    assert.deepEqual(list(from), ['notes.txt'])
  })

  it('goes back to the old folder when the backend does not start on the new one', async () => {
    let starts = 0
    const { deps, calls, phases } = fakes({
      startBackend: async () => {
        calls.push('start')
        if (++starts === 1) throw new Error('boom')
      },
    })

    const result = await runMove(await request({ previousDir: from }), deps)

    assert.equal(result, 'failed')
    assert.deepEqual(calls, ['stop', 'save:' + to, 'start', `save:${from}`, 'start'])
    assert.deepEqual(list(to), [])
    assert.deepEqual(list(from), [`${A}-thumb.jpg`, `${A}.png`, 'notes.txt'].sort())
    assert.equal(phases[phases.length - 1].result?.ok, false)
  })

  it('removes the copies and keeps the old folder when the user cancels', async () => {
    const { deps, calls, phases } = fakes({ isCancelled: () => true })

    const result = await runMove(await request(), deps)

    assert.equal(result, 'cancelled')
    assert.deepEqual(calls, ['stop', 'start'])
    assert.deepEqual(list(to), [])
    assert.deepEqual(list(from), [`${A}-thumb.jpg`, `${A}.png`, 'notes.txt'].sort())
    assert.equal(phases[phases.length - 1].result?.ok, false)
  })

  it('honors a cancel that comes in while the last file is copied', async () => {
    let cancel = false
    const { deps, calls } = fakes({
      isCancelled: () => cancel,
      emit: (patch) => {
        if (patch.phase === 'copying' && patch.filesDone === 2) cancel = true
      },
    })

    const result = await runMove(await request(), deps)

    assert.equal(result, 'cancelled')
    assert.deepEqual(calls, ['stop', 'start'])
    assert.deepEqual(list(to), [])
    assert.deepEqual(list(from), [`${A}-thumb.jpg`, `${A}.png`, 'notes.txt'].sort())
  })

  it('restarts and changes nothing when the new plan is rejected', async () => {
    const { deps, calls } = fakes()
    const req = await request({
      replan: async () => ({ ok: false, reason: 'The current assets folder was not found.' }),
    })

    const result = await runMove(req, deps)

    assert.equal(result, 'failed')
    assert.deepEqual(calls, ['stop', 'start'])
    assert.deepEqual(list(to), [])
    assert.deepEqual(list(from), [`${A}-thumb.jpg`, `${A}.png`, 'notes.txt'].sort())
  })

  it('does not start a second backend when the first one does not stop', async () => {
    const { deps, calls, phases } = fakes({
      stopBackend: async () => {
        calls.push('stop')
        throw new Error('timeout')
      },
    })

    const result = await runMove(await request(), deps)

    assert.equal(result, 'failed')
    assert.deepEqual(calls, ['stop'])
    assert.match(phases[phases.length - 1].result?.message ?? '', /did not stop/)
  })

  it('moves back to the default folder without a marker', async () => {
    const { deps, calls } = fakes()

    const result = await runMove(await request({ toCustom: false, previousDir: from }), deps)

    assert.equal(result, 'moved')
    assert.deepEqual(calls, ['stop', 'save:default', 'start'])
    assert.equal(fs.existsSync(path.join(to, ASSETS_MARKER_FILENAME)), false)
  })
})
