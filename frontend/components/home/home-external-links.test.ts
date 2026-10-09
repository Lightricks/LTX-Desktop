import assert from 'node:assert/strict'
import { afterEach, describe, it, mock } from 'node:test'
import {
  HOME_EXTERNAL_URLS,
  openExternalBrowserUrl,
  openHomeExternalUrl,
} from './home-external-links.ts'

type OpenExternalUrl = (input: { url: string }) => Promise<boolean>

function installElectronApi(openExternalUrl: OpenExternalUrl) {
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: {
      electronAPI: {
        openExternalUrl,
      },
    },
  })
}

afterEach(() => {
  mock.restoreAll()
})

describe('home external links', () => {
  it('exports the Desktop Home destinations', () => {
    assert.equal(
      HOME_EXTERNAL_URLS.contactSales,
      'https://ltx.io/forms/ltx-contact-sales?placement=desktop_home',
    )
    assert.equal(HOME_EXTERNAL_URLS.support, 'https://support.ltx.studio/')
    assert.equal(
      HOME_EXTERNAL_URLS.reportIssue,
      'https://github.com/Lightricks/LTX-Desktop/issues/new',
    )
  })

  it('opens URLs through the Electron external-url bridge', async () => {
    const calls: Array<{ url: string }> = []
    installElectronApi(async (input) => {
      calls.push(input)
      return true
    })

    openHomeExternalUrl(HOME_EXTERNAL_URLS.contactSales)
    await Promise.resolve()

    assert.deepEqual(calls, [{ url: HOME_EXTERNAL_URLS.contactSales }])
  })

  it('opens arbitrary https URLs through the same Electron bridge', async () => {
    const calls: Array<{ url: string }> = []
    installElectronApi(async (input) => {
      calls.push(input)
      return true
    })

    const licenseUrl =
      'https://github.com/Lightricks/LTX-2/blob/main/LICENSE'
    openExternalBrowserUrl(licenseUrl)
    await Promise.resolve()

    assert.deepEqual(calls, [{ url: licenseUrl }])
  })

  it('opens URLs with window.open when Electron is unavailable', () => {
    const openCalls: unknown[][] = []
    Object.defineProperty(globalThis, 'window', {
      configurable: true,
      value: {
        open: (...args: unknown[]) => {
          openCalls.push(args)
          return null
        },
      },
    })

    openHomeExternalUrl(HOME_EXTERNAL_URLS.support)

    assert.deepEqual(openCalls, [
      [HOME_EXTERNAL_URLS.support, '_blank', 'noopener,noreferrer'],
    ])
  })

  it('logs a warning when the Electron bridge rejects', async () => {
    const warnings: string[] = []
    mock.method(console, 'warn', (...args: unknown[]) => {
      warnings.push(String(args[0]))
    })
    installElectronApi(async () => {
      throw new Error('blocked')
    })

    openHomeExternalUrl(HOME_EXTERNAL_URLS.support)
    await Promise.resolve()
    await Promise.resolve()

    assert.equal(warnings.length, 1)
    assert.match(warnings[0] ?? '', /Failed to open external Home link/)
    assert.match(warnings[0] ?? '', /blocked/)
  })
})
