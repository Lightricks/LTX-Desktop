import assert from 'node:assert/strict'
import path from 'node:path'
import { describe, it } from 'node:test'
import { freeDiskBytes } from './free-disk-space.ts'

describe('freeDiskBytes', () => {
  it('returns bavail * bsize for an existing path', async () => {
    const bytes = await freeDiskBytes('/models', {
      statfs: async (p) => {
        assert.equal(p, path.resolve('/models'))
        return { bavail: 100n, bsize: 1024 }
      },
    })
    assert.equal(bytes, 100 * 1024)
  })

  it('walks up to a parent when the target folder does not exist yet', async () => {
    const missing = path.join('/data', 'LTXDesktop', 'models')
    const parent = path.join('/data', 'LTXDesktop')
    const missingErr = Object.assign(new Error('ENOENT'), { code: 'ENOENT' })
    const bytes = await freeDiskBytes(missing, {
      statfs: async (p) => {
        if (p === path.resolve(missing)) throw missingErr
        if (p === path.resolve(parent)) return { bavail: 50, bsize: 4096 }
        throw new Error(`unexpected path ${p}`)
      },
    })
    assert.equal(bytes, 50 * 4096)
  })

  it('does not walk up on permission errors', async () => {
    const target = path.join('/data', 'LTXDesktop', 'models')
    const denied = Object.assign(new Error('EACCES'), { code: 'EACCES' })
    await assert.rejects(
      () =>
        freeDiskBytes(target, {
          statfs: async (p) => {
            if (p === path.resolve(target)) throw denied
            throw new Error(`must not walk up to ${p}`)
          },
        }),
      (err: NodeJS.ErrnoException) => err.code === 'EACCES',
    )
  })

  it('rejects a relative path', async () => {
    await assert.rejects(() => freeDiskBytes('models'), /absolute/)
  })

  it('reads free space from the real filesystem', async () => {
    const bytes = await freeDiskBytes(process.cwd())
    assert.ok(bytes > 0)
  })
})
