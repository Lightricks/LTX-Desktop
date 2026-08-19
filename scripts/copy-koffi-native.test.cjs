'use strict'

const assert = require('node:assert/strict')
const path = require('node:path')
const { describe, it } = require('node:test')
const { archName, koffiNativeDest, resourcesDir } = require('./copy-koffi-native.cjs')

describe('archName', () => {
  it('passes string arch through', () => {
    assert.equal(archName('x64'), 'x64')
  })

  it('maps electron-builder Arch enum integers', () => {
    assert.equal(archName(1), 'x64')
    assert.equal(archName(3), 'arm64')
  })
})

describe('koffiNativeDest', () => {
  it('matches Koffi resourcesPath lookup win32_x64/koffi.node', () => {
    assert.equal(
      koffiNativeDest(path.join('app', 'resources'), 'win32', 'x64'),
      path.join('app', 'resources', 'koffi', 'win32_x64', 'koffi.node'),
    )
  })
})

describe('resourcesDir', () => {
  it('uses packager.getResourcesDir when present', () => {
    const dir = resourcesDir({
      appOutDir: '/out',
      packager: { getResourcesDir: (appOutDir) => `${appOutDir}/R` },
    })
    assert.equal(dir, '/out/R')
  })

  it('falls back to Contents/Resources on darwin', () => {
    const dir = resourcesDir({
      appOutDir: '/out',
      electronPlatformName: 'darwin',
      packager: { appInfo: { productFilename: 'LTX Desktop' } },
    })
    assert.equal(dir, path.join('/out', 'LTX Desktop.app', 'Contents', 'Resources'))
  })

  it('falls back to resources/ on win32', () => {
    const dir = resourcesDir({
      appOutDir: '/out',
      electronPlatformName: 'win32',
      packager: { appInfo: { productFilename: 'LTX Desktop' } },
    })
    assert.equal(dir, path.join('/out', 'resources'))
  })
})
