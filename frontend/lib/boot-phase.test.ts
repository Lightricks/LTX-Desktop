import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  resolveBootPhase,
  resolveBootWindow,
  splashHoldsWindow,
  type BootInputs,
} from './boot-phase.ts'

const ready: BootInputs = {
  pythonReady: true,
  isBackendDead: false,
  backendLoading: false,
  connected: true,
  waitingForRuntimePolicy: false,
  setupState: { needsSetup: false },
  forceApiGenerations: false,
  requiredModelsGate: 'ready',
}

describe('resolveBootPhase', () => {
  it('resolves the steady-state ready phase', () => {
    assert.deepEqual(resolveBootPhase(ready), { kind: 'ready' })
  })

  it('prefers the boot spinner while python readiness is unknown', () => {
    assert.deepEqual(
      resolveBootPhase({ ...ready, pythonReady: null, isBackendDead: true }),
      { kind: 'boot-spinner' },
    )
  })

  it('resolves missing python to the splash with a download payload', () => {
    assert.deepEqual(resolveBootPhase({ ...ready, pythonReady: false }), {
      kind: 'splash',
      pythonRequired: true,
      liveInstall: true,
      windowModeKnown: true,
    })
  })

  it('prefers the splash download payload over a dead backend', () => {
    assert.deepEqual(
      resolveBootPhase({ ...ready, pythonReady: false, isBackendDead: true }),
      {
        kind: 'splash',
        pythonRequired: true,
        liveInstall: true,
        windowModeKnown: true,
      },
    )
  })

  it('prefers backend-dead over setup/missing-models gates', () => {
    assert.deepEqual(
      resolveBootPhase({
        ...ready,
        isBackendDead: true,
        setupState: { needsSetup: true },
        requiredModelsGate: 'missing',
      }),
      { kind: 'backend-dead', liveInstall: true, windowModeKnown: true },
    )
  })

  it('marks the window mode unknown while setup state is still loading', () => {
    assert.deepEqual(
      resolveBootPhase({ ...ready, isBackendDead: true, setupState: 'loading' }),
      { kind: 'backend-dead', liveInstall: false, windowModeKnown: false },
    )
  })

  it('keeps the splash up while the backend is loading', () => {
    const phase = resolveBootPhase({ ...ready, backendLoading: true })
    assert.equal(phase.kind, 'splash')
  })

  it('keeps the splash up while setup state is loading', () => {
    assert.deepEqual(resolveBootPhase({ ...ready, setupState: 'loading' }), {
      kind: 'splash',
      pythonRequired: false,
      liveInstall: false,
      windowModeKnown: false,
    })
  })

  it('keeps the splash up while waiting for the runtime policy', () => {
    const phase = resolveBootPhase({ ...ready, waitingForRuntimePolicy: true })
    assert.equal(phase.kind, 'splash')
  })

  it('keeps the splash up while required models are still checking', () => {
    const phase = resolveBootPhase({ ...ready, requiredModelsGate: 'checking' })
    assert.equal(phase.kind, 'splash')
  })

  it('does not wait for required models while disconnected', () => {
    assert.deepEqual(
      resolveBootPhase({ ...ready, requiredModelsGate: 'checking', connected: false }),
      { kind: 'ready' },
    )
  })

  it('does not wait for required models for forced API generations', () => {
    assert.deepEqual(
      resolveBootPhase({
        ...ready,
        requiredModelsGate: 'checking',
        forceApiGenerations: true,
      }),
      { kind: 'ready' },
    )
  })

  it('prefers setup over missing-models', () => {
    assert.deepEqual(
      resolveBootPhase({
        ...ready,
        setupState: { needsSetup: true },
        requiredModelsGate: 'missing',
      }),
      { kind: 'setup' },
    )
  })

  it('skips setup for forced API generations', () => {
    assert.deepEqual(
      resolveBootPhase({
        ...ready,
        setupState: { needsSetup: true },
        forceApiGenerations: true,
      }),
      { kind: 'ready' },
    )
  })

  it('resolves missing-models once setup clears', () => {
    assert.deepEqual(
      resolveBootPhase({ ...ready, requiredModelsGate: 'missing' }),
      { kind: 'missing-models' },
    )
  })

  it('carries liveInstall on the splash phase for window-mode mapping', () => {
    assert.deepEqual(
      resolveBootPhase({
        ...ready,
        backendLoading: true,
        setupState: { needsSetup: true },
      }),
      { kind: 'splash', pythonRequired: false, liveInstall: true, windowModeKnown: true },
    )
  })
})

describe('splashHoldsWindow', () => {
  it('never holds for the full-screen phases, even before the splash finishes', () => {
    assert.equal(splashHoldsWindow({ kind: 'boot-spinner' }, false), false)
    assert.equal(
      splashHoldsWindow(
        { kind: 'backend-dead', liveInstall: false, windowModeKnown: true },
        false,
      ),
      false,
    )
  })

  it('never holds while the splash carries a python download', () => {
    for (const splashFinished of [false, true]) {
      assert.equal(
        splashHoldsWindow(
          { kind: 'splash', pythonRequired: true, liveInstall: true, windowModeKnown: true },
          splashFinished,
        ),
        false,
      )
    }
  })

  it('holds for every shell phase until the splash finishes', () => {
    assert.equal(
      splashHoldsWindow(
        { kind: 'splash', pythonRequired: false, liveInstall: false, windowModeKnown: true },
        false,
      ),
      true,
    )
    assert.equal(splashHoldsWindow({ kind: 'setup' }, false), true)
    assert.equal(splashHoldsWindow({ kind: 'missing-models' }, false), true)
    assert.equal(splashHoldsWindow({ kind: 'ready' }, false), true)
    assert.equal(splashHoldsWindow({ kind: 'ready' }, true), false)
  })
})

describe('resolveBootWindow', () => {
  it('pins install and skips the IPC while python readiness is unknown', () => {
    assert.deepEqual(resolveBootWindow({ kind: 'boot-spinner' }, false), {
      document: 'install',
      main: null,
    })
  })

  it('pins and notifies install while the splash carries a python download', () => {
    for (const splashFinished of [false, true]) {
      assert.deepEqual(
        resolveBootWindow(
          { kind: 'splash', pythonRequired: true, liveInstall: true, windowModeKnown: true },
          splashFinished,
        ),
        { document: 'install', main: 'install' },
      )
    }
  })

  it('skips the IPC for a dead backend while setup state is loading', () => {
    assert.deepEqual(
      resolveBootWindow(
        { kind: 'backend-dead', liveInstall: false, windowModeKnown: false },
        true,
      ),
      { document: 'install', main: null },
    )
  })

  it('keeps a set-up app in app mode when the backend dies', () => {
    assert.deepEqual(
      resolveBootWindow(
        { kind: 'backend-dead', liveInstall: false, windowModeKnown: true },
        true,
      ),
      { document: 'app', main: 'app' },
    )
  })

  it('suppresses the IPC while the splash holds the window', () => {
    assert.deepEqual(
      resolveBootWindow(
        { kind: 'splash', pythonRequired: false, liveInstall: true, windowModeKnown: true },
        false,
      ),
      { document: 'install', main: null },
    )
    assert.deepEqual(resolveBootWindow({ kind: 'setup' }, false), {
      document: 'install',
      main: null,
    })
    assert.deepEqual(resolveBootWindow({ kind: 'missing-models' }, false), {
      document: 'install',
      main: null,
    })
    assert.deepEqual(resolveBootWindow({ kind: 'ready' }, false), {
      document: 'install',
      main: null,
    })
  })

  it('notifies install for the setup gates once the splash releases', () => {
    assert.deepEqual(resolveBootWindow({ kind: 'setup' }, true), {
      document: 'install',
      main: 'install',
    })
    assert.deepEqual(resolveBootWindow({ kind: 'missing-models' }, true), {
      document: 'install',
      main: 'install',
    })
  })

  it('flips to app mode when the splash releases a ready app', () => {
    assert.deepEqual(resolveBootWindow({ kind: 'ready' }, true), {
      document: 'app',
      main: 'app',
    })
  })
})
