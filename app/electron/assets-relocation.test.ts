import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, beforeEach, describe, it } from 'node:test'
import {
  ASSETS_MARKER_FILENAME,
  checkCustomAssetsDir,
  copyFiles,
  deleteMovedFiles,
  preflightMove,
  releaseFolder,
  writeMarker,
  type PreflightOptions,
} from './assets-relocation.ts'

const A = '11111111-1111-4111-8111-111111111111'
const B = '22222222-2222-4222-8222-222222222222'

let root: string
let from: string
let to: string

beforeEach(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'ltx-assets-'))
  from = path.join(root, 'old')
  to = path.join(root, 'new')
  fs.mkdirSync(from)
  fs.mkdirSync(to)
})

afterEach(() => {
  fs.rmSync(root, { recursive: true, force: true })
})

function write(dir: string, name: string, content: string): void {
  fs.writeFileSync(path.join(dir, name), content)
}

function options(overrides: Partial<PreflightOptions> = {}): PreflightOptions {
  return {
    from,
    to,
    protectedDirs: [],
    blockedRoots: [],
    appFolders: [],
    freeBytes: async () => 1_000_000,
    ...overrides,
  }
}

async function plan(overrides: Partial<PreflightOptions> = {}) {
  const result = await preflightMove(options(overrides))
  assert.equal(result.ok, true, result.ok ? '' : result.reason)
  if (!result.ok) throw new Error(result.reason)
  return result.plan
}

async function reason(overrides: Partial<PreflightOptions> = {}): Promise<string> {
  const result = await preflightMove(options(overrides))
  assert.equal(result.ok, false)
  return result.ok ? '' : result.reason
}

describe('preflightMove', () => {
  it('lists the files, skips the in-flight folder and the marker, and counts the bytes', async () => {
    write(from, `${A}.png`, '12345')
    write(from, `${A}-thumb.jpg`, '12')
    write(from, ASSETS_MARKER_FILENAME, '')
    fs.mkdirSync(path.join(from, '.in-flight'))
    write(path.join(from, '.in-flight'), 'x.mp4', 'busy')

    const result = await plan()

    assert.deepEqual(result.files.map((f) => f.name).sort(), [`${A}-thumb.jpg`, `${A}.png`])
    assert.equal(result.totalBytes, 7)
  })

  it('leaves foreign files, partial copies and the marker out of the list', async () => {
    write(from, `${A}.png`, '1')
    write(from, `${B}.png.ltx-partial`, '2')
    write(from, 'notes.txt', 'mine')
    write(from, 'photo.png', 'mine')
    write(from, ASSETS_MARKER_FILENAME, '')

    const result = await plan()

    assert.deepEqual(result.files.map((f) => f.name), [`${A}.png`])
  })

  it('rejects a missing old folder: it is not an empty library', async () => {
    fs.rmSync(from, { recursive: true })
    assert.match(await reason(), /was not found/)
  })

  it('moves the content of an asset file that is a link and deletes only the link', async (t) => {
    write(root, 'real.png', 'real')
    try {
      fs.symlinkSync(path.join(root, 'real.png'), path.join(from, `${A}.png`))
    } catch {
      t.skip('symbolic links are not allowed here')
      return
    }
    const result = await plan()
    assert.deepEqual(result.files, [{ name: `${A}.png`, size: 4 }])

    await copyFiles(result, { onProgress: () => {}, isCancelled: () => false })
    await deleteMovedFiles(result)

    assert.equal(fs.readFileSync(path.join(to, `${A}.png`), 'utf8'), 'real')
    assert.equal(fs.existsSync(path.join(from, `${A}.png`)), false)
    assert.equal(fs.readFileSync(path.join(root, 'real.png'), 'utf8'), 'real')
  })

  it('rejects the same folder', async () => {
    assert.match(await reason({ to: from }), /uses now/)
  })

  it('rejects a folder inside the old folder and a parent of the old folder', async () => {
    const inside = path.join(from, 'inner')
    fs.mkdirSync(inside)
    assert.match(await reason({ to: inside }), /must not contain/)
    assert.match(await reason({ to: root }), /must not contain/)
  })

  it('rejects a drive root, a network path, a protected folder and a system folder', async () => {
    assert.match(await reason({ to: path.parse(root).root }), /drive root/)
    assert.match(await reason({ to: '\\\\server\\share\\assets' }), /Network/)
    const data = path.join(root, 'data')
    fs.mkdirSync(path.join(data, 'sub'), { recursive: true })
    assert.match(await reason({ to: data, protectedDirs: [path.join(data, 'sub')] }), /app or personal/)
    const system = path.join(root, 'system')
    fs.mkdirSync(path.join(system, 'inner'), { recursive: true })
    assert.match(await reason({ to: path.join(system, 'inner'), blockedRoots: [system] }), /system folder/)
  })

  it('names the app folders when the target is inside one', async () => {
    const appData = path.join(root, 'app-data')
    fs.mkdirSync(path.join(appData, 'models'), { recursive: true })
    assert.match(await reason({ to: path.join(appData, 'models'), appFolders: [appData] }), /inside the app folders/)
  })

  it('rejects a new folder inside the old folder when the path goes through a link', async (t) => {
    const link = path.join(root, 'link')
    try {
      fs.symlinkSync(root, link)
    } catch {
      t.skip('symbolic links are not allowed here')
      return
    }
    // The folder does not exist yet, so the check must resolve the link through its parent.
    assert.match(await reason({ to: path.join(link, 'old', 'inner') }), /must not contain/)
    assert.match(
      await reason({ to: path.join(link, 'data', 'new'), appFolders: [path.join(root, 'data')] }),
      /inside the app folders/,
    )
  })

  it('rejects a move that does not fit and shows the sizes', async () => {
    write(from, `${A}.png`, 'x'.repeat(4096))
    assert.match(await reason({ freeBytes: async () => 1024 }), /Not enough space: 4.0 KB needed, 1.0 KB free/)
  })

  it('allows a foreign folder and leaves its files alone', async () => {
    write(to, 'notes.txt', 'mine')
    write(from, `${A}.png`, '1')
    await plan()
    assert.equal(fs.readFileSync(path.join(to, 'notes.txt'), 'utf8'), 'mine')
  })

})

describe('copyFiles', () => {
  it('copies files with the same content and reports progress', async () => {
    write(from, `${A}.png`, 'hello')
    write(from, `${B}.jpg`, 'world!')
    const result = await plan()
    const seen: number[] = []

    const copied = await copyFiles(result, {
      onProgress: (p) => seen.push(p.bytesDone),
      isCancelled: () => false,
    })

    assert.equal(copied.status, 'done')
    assert.equal(fs.readFileSync(path.join(to, `${A}.png`), 'utf8'), 'hello')
    assert.equal(fs.readFileSync(path.join(to, `${B}.jpg`), 'utf8'), 'world!')
    assert.equal(seen[seen.length - 1], 11)
    assert.deepEqual(fs.readdirSync(to).sort(), [`${A}.png`, `${B}.jpg`])
  })

  it('replaces a file with the same name in the target', async () => {
    write(from, `${A}.png`, 'hello')
    write(to, `${A}.png`, 'old')
    const result = await plan()

    await copyFiles(result, { onProgress: () => {}, isCancelled: () => false })

    assert.equal(fs.readFileSync(path.join(to, `${A}.png`), 'utf8'), 'hello')
  })

  it('removes everything it created when it is cancelled', async () => {
    write(from, `${A}.png`, 'hello')
    write(from, `${B}.png`, 'world')
    const result = await plan()
    let cancel = false

    const copied = await copyFiles(result, {
      onProgress: (p) => {
        if (p.filesDone === 1) cancel = true
      },
      isCancelled: () => cancel,
    })

    assert.equal(copied.status, 'cancelled')
    assert.deepEqual(fs.readdirSync(to), [])
    assert.deepEqual(fs.readdirSync(from).sort(), [`${A}.png`, `${B}.png`])
  })

  it('keeps a file that was in the target before when a later file is cancelled', async () => {
    write(from, `${A}.png`, 'hello')
    write(from, `${B}.png`, 'world')
    write(to, `${A}.png`, 'old')
    const result = await plan()
    let cancel = false

    const copied = await copyFiles(result, {
      onProgress: (p) => {
        if (p.filesDone === 1) cancel = true
      },
      isCancelled: () => cancel,
    })

    assert.equal(copied.status, 'cancelled')
    assert.deepEqual(fs.readdirSync(to), [`${A}.png`])
  })

  it('removes the partial copy and throws when the source cannot be read', async () => {
    write(from, `${A}.png`, 'hello')
    const result = await plan()
    fs.rmSync(path.join(from, `${A}.png`))

    await assert.rejects(() => copyFiles(result, { onProgress: () => {}, isCancelled: () => false }))
    assert.deepEqual(fs.readdirSync(to), [])
  })
})

describe('deleteMovedFiles', () => {
  it('deletes the old asset files and keeps everything else', async () => {
    write(from, `${A}.png`, 'hello')
    write(from, 'notes.txt', 'mine')
    fs.mkdirSync(path.join(from, '.in-flight'))
    const result = await plan()

    const failed = await deleteMovedFiles(result)

    assert.deepEqual(failed, [])
    assert.deepEqual(fs.readdirSync(from).sort(), ['.in-flight', 'notes.txt'])
  })
})

describe('marker', () => {
  it('writeMarker tells whether it created the marker, and releaseFolder removes it', async () => {
    assert.equal(await writeMarker(to), true)
    assert.equal(await writeMarker(to), false)
    fs.mkdirSync(path.join(to, '.in-flight'))
    await releaseFolder(to)
    assert.deepEqual(fs.readdirSync(to), [])
  })

  it('releaseFolder keeps an in-flight folder that is not empty', async () => {
    fs.mkdirSync(path.join(to, '.in-flight'))
    write(path.join(to, '.in-flight'), 'x.mp4', 'busy')
    await releaseFolder(to)
    assert.deepEqual(fs.readdirSync(to), ['.in-flight'])
  })
})

describe('marker contract', () => {
  it('matches shared/assets-marker.json', () => {
    const contract = JSON.parse(
      fs.readFileSync(new URL('../shared/assets-marker.json', import.meta.url), 'utf8'),
    ) as { filename: string }
    assert.equal(ASSETS_MARKER_FILENAME, contract.filename)
  })
})

describe('checkCustomAssetsDir', () => {
  it('fails when the folder is missing', async () => {
    const check = await checkCustomAssetsDir(path.join(root, 'gone'))
    assert.equal(check.ok, false)
  })

  it('fails for a network path without touching it', async () => {
    const check = await checkCustomAssetsDir('\\\\server\\share\\assets')
    assert.equal(check.ok, false)
  })

  it('fails when the marker is missing, and passes once it is written', async () => {
    const before = await checkCustomAssetsDir(to)
    assert.equal(before.ok, false)
    await writeMarker(to)
    assert.deepEqual(await checkCustomAssetsDir(to), { ok: true })
  })
})
