import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  activeDownloadMatchesUpgrade,
  clearUpgradeDeleteOld,
  upgradePromptVisible,
  hasStoredUpgradeDeleteOld,
  readUpgradeDeleteOld,
  writeUpgradeDeleteOld,
} from './upgrade-prompt-download.ts'

function installSessionStorage(): void {
  const store = new Map<string, string>()
  Object.defineProperty(globalThis, 'sessionStorage', {
    configurable: true,
    value: {
      getItem: (key: string) => store.get(key) ?? null,
      setItem: (key: string, value: string) => {
        store.set(key, value)
      },
      removeItem: (key: string) => {
        store.delete(key)
      },
    },
  })
}

describe('upgradePromptVisible', () => {
  it('keeps the offer up after the user starts the download', () => {
    assert.equal(upgradePromptVisible({ frozen: false, background: false }), true)
  })

  it('hides the offer when a refresh finds that download still running', () => {
    assert.equal(upgradePromptVisible({ frozen: false, background: true }), false)
  })

  it('still shows frozen previews', () => {
    assert.equal(upgradePromptVisible({ frozen: true, background: true }), true)
  })
})

describe('activeDownloadMatchesUpgrade', () => {
  it('matches when the running download includes an upgrade checkpoint', () => {
    assert.equal(
      activeDownloadMatchesUpgrade(
        ['ltx-2.5-22b-distilled', 'ltx-2.5-video-vae'],
        ['ltx-2.5-video-vae'],
      ),
      true,
    )
  })

  it('ignores an unrelated download and an idle session', () => {
    assert.equal(
      activeDownloadMatchesUpgrade(['gemma-3-12b-it-qat-q4_0-unquantized'], ['ltx-2.5-22b-distilled']),
      false,
    )
    assert.equal(activeDownloadMatchesUpgrade([], ['ltx-2.5-22b-distilled']), false)
  })
})

describe('upgrade delete-old choice', () => {
  it('restores the choice saved for that model', () => {
    installSessionStorage()
    assert.equal(readUpgradeDeleteOld('ltx-2.5', true), true)
    writeUpgradeDeleteOld('ltx-2.5', false)
    assert.equal(readUpgradeDeleteOld('ltx-2.5', true), false)
    writeUpgradeDeleteOld('ltx-2.3', true)
    assert.equal(readUpgradeDeleteOld('ltx-2.5', true), false)
    assert.equal(hasStoredUpgradeDeleteOld('ltx-2.5'), true)
    clearUpgradeDeleteOld('ltx-2.5')
    assert.equal(hasStoredUpgradeDeleteOld('ltx-2.5'), false)
    assert.equal(readUpgradeDeleteOld('ltx-2.5', true), true)
  })
})
