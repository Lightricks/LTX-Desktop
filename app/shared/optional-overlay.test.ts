import assert from 'node:assert/strict'
import path from 'node:path'
import { describe, it } from 'node:test'

import {
  findOptionalOverlayDir,
  optionalOverlayAliases,
  shouldBindOptionalOverlay,
} from './optional-overlay.ts'

const ROOT = '/repo'

function overlayIo(present: boolean) {
  const overlay = path.join(ROOT, 'internal', 'app-overlay')
  const files = new Set([
    path.join(ROOT, 'internal'),
    overlay,
    path.join(overlay, 'wrap-app.tsx'),
    path.join(overlay, 'main', 'register-handlers.ts'),
    path.join(overlay, 'main', 'attach-preload.ts'),
  ])
  return {
    existsSync: (p: string) => present && files.has(p),
    readdirSync: (p: string) => {
      if (!present) throw new Error('missing')
      if (p === path.join(ROOT, 'internal')) return ['app-overlay']
      return []
    },
  }
}

describe('findOptionalOverlayDir', () => {
  it('returns null when internal/ is absent', () => {
    assert.equal(findOptionalOverlayDir(ROOT, overlayIo(false)), null)
  })

  it('returns the first overlay folder that has the three entry files', () => {
    assert.equal(
      findOptionalOverlayDir(ROOT, overlayIo(true)),
      path.join(ROOT, 'internal', 'app-overlay'),
    )
  })
})

describe('shouldBindOptionalOverlay', () => {
  const dir = path.join(ROOT, 'internal', 'app-overlay')

  it('binds only during serve when an overlay exists', () => {
    assert.equal(shouldBindOptionalOverlay('serve', dir), true)
    assert.equal(shouldBindOptionalOverlay('build', dir), false)
    assert.equal(shouldBindOptionalOverlay('serve', null), false)
    assert.equal(shouldBindOptionalOverlay('build', null), false)
  })
})

describe('optionalOverlayAliases', () => {
  it('points at stubs on build even if the overlay dir exists', () => {
    const aliases = optionalOverlayAliases({
      root: ROOT,
      command: 'build',
      overlayDir: path.join(ROOT, 'internal', 'app-overlay'),
    })
    const wrap = aliases.find((a) => a.find instanceof RegExp && a.find.test('@optional/app-wrap'))
    assert.ok(wrap)
    assert.equal(wrap.replacement, path.resolve(ROOT, 'frontend/lib/optional-app-wrap.tsx'))
    for (const alias of aliases) {
      assert.equal(
        alias.replacement.includes(`${path.sep}internal${path.sep}`),
        false,
        `build alias leaked internal path: ${alias.replacement}`,
      )
    }
  })

  it('points at overlay files on serve when the dir exists', () => {
    const overlayDir = path.join(ROOT, 'internal', 'app-overlay')
    const aliases = optionalOverlayAliases({
      root: ROOT,
      command: 'serve',
      overlayDir,
    })
    const wrap = aliases.find((a) => a.find instanceof RegExp && a.find.test('@optional/app-wrap'))
    assert.ok(wrap)
    assert.equal(wrap.replacement, path.resolve(overlayDir, 'wrap-app.tsx'))
  })
})
