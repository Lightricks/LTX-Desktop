/**
 * Pure boot-phase resolver for App.tsx.
 *
 * Folds the boot booleans (python readiness, backend liveness, setup state,
 * model-gate state, connectivity) into one discriminated union so App renders
 * exactly one view per phase. Precedence is explicit and top-to-bottom:
 *
 *   python readiness → dead backend → splash → setup → missing-models → ready
 *
 * Python readiness folds INTO the splash (no separate screen): missing python
 * resolves to the splash phase with pythonRequired, and App feeds the live
 * download progress into the splash's python prop.
 */

export type BootSetupState = 'loading' | { needsSetup: boolean }
export type BootModelsGate = 'checking' | 'missing' | 'ready'

export interface BootInputs {
  pythonReady: boolean | null
  isBackendDead: boolean
  backendLoading: boolean
  connected: boolean
  waitingForRuntimePolicy: boolean
  setupState: BootSetupState
  forceApiGenerations: boolean
  requiredModelsGate: BootModelsGate
}

export type BootPhase =
  | { kind: 'boot-spinner' }
  | { kind: 'backend-dead'; liveInstall: boolean; windowModeKnown: boolean }
  | { kind: 'splash'; pythonRequired: boolean; liveInstall: boolean; windowModeKnown: boolean }
  | { kind: 'setup' }
  | { kind: 'missing-models' }
  | { kind: 'ready' }

export function resolveBootPhase(inputs: BootInputs): BootPhase {
  const {
    pythonReady,
    isBackendDead,
    backendLoading,
    connected,
    waitingForRuntimePolicy,
    setupState,
    forceApiGenerations,
    requiredModelsGate,
  } = inputs

  // 1. Python readiness gates everything: unknown → spinner, missing → splash
  // with a download payload (the splash embeds the download; no separate screen).
  // Missing python wins over a dead backend: crash UI is meaningless when the
  // runtime itself is absent (unreachable in practice — the backend never
  // starts before python is ready). The install window is definitively known.
  if (pythonReady === null) return { kind: 'boot-spinner' }
  if (pythonReady === false) {
    return { kind: 'splash', pythonRequired: true, liveInstall: true, windowModeKnown: true }
  }

  // pythonReady is true from here, so liveInstall drops the python term.
  // Only the splash/dead phases vary it; every other phase fixes it
  // (setup/missing → true, ready → false), hence the payloads below.
  const liveInstall =
    requiredModelsGate === 'missing' ||
    (setupState !== 'loading' && setupState.needsSetup && !forceApiGenerations)

  // 2. A dead backend preempts all app UI (including the setup gates).
  if (isBackendDead) {
    return { kind: 'backend-dead', liveInstall, windowModeKnown: setupState !== 'loading' }
  }

  // 3. The splash covers the window until every readiness gate clears.
  const waitingForRequiredModels =
    requiredModelsGate === 'checking' &&
    connected &&
    setupState !== 'loading' &&
    !waitingForRuntimePolicy &&
    !forceApiGenerations
  if (
    backendLoading ||
    setupState === 'loading' ||
    waitingForRuntimePolicy ||
    waitingForRequiredModels
  ) {
    return {
      kind: 'splash',
      pythonRequired: false,
      liveInstall,
      windowModeKnown: setupState !== 'loading',
    }
  }

  // setupState is loaded here: the splash gate above returned otherwise.
  if (setupState.needsSetup && !forceApiGenerations) return { kind: 'setup' }
  if (requiredModelsGate === 'missing') return { kind: 'missing-models' }
  return { kind: 'ready' }
}

export type BootWindowMode = 'install' | 'app'

export interface BootWindow {
  /** Always defined: mirrors documentWindowMode ('install' while unknown). */
  document: BootWindowMode
  /** Null while unknown or while the splash holds the window: skip the IPC. */
  main: BootWindowMode | null
}

/**
 * Whether the splash overlay holds the install window: no setMainWindowMode
 * IPC may fire while it does. Only the shell phases
 * (splash/setup/missing-models/ready) can hold — the full-screen
 * phases (spinner/backend-dead) never show the splash. A splash carrying a
 * python download never holds either: the install IPC fires immediately so
 * the download keeps install mode for its whole duration.
 */
export function splashHoldsWindow(phase: BootPhase, splashFinished: boolean): boolean {
  switch (phase.kind) {
    case 'boot-spinner':
    case 'backend-dead':
      return false
    case 'splash':
      if (phase.pythonRequired) return false
      return !splashFinished
    case 'setup':
    case 'missing-models':
    case 'ready':
      return !splashFinished
    default: {
      const _exhaustive: never = phase
      return _exhaustive
    }
  }
}

/**
 * Trivial window-mode mapping off the phase. The splash-hold suppresses the
 * main-window IPC (but pins the document to install); unknown modes (boot
 * spinner, setup state still loading) also skip the IPC.
 */
export function resolveBootWindow(phase: BootPhase, splashFinished: boolean): BootWindow {
  const holdsWindow = splashHoldsWindow(phase, splashFinished)
  switch (phase.kind) {
    case 'boot-spinner':
      return { document: 'install', main: null }
    case 'backend-dead': {
      const mode: BootWindowMode = phase.liveInstall ? 'install' : 'app'
      return {
        document: phase.windowModeKnown ? mode : 'install',
        main: phase.windowModeKnown ? mode : null,
      }
    }
    case 'splash': {
      // Python download keeps install mode (the old python-install mapping):
      // pin the document and fire the IPC immediately instead of waiting for
      // the splash to release.
      if (phase.pythonRequired) return { document: 'install', main: 'install' }
      const mode: BootWindowMode = phase.liveInstall ? 'install' : 'app'
      const known = phase.windowModeKnown && !holdsWindow
      return { document: known ? mode : 'install', main: known ? mode : null }
    }
    case 'setup':
    case 'missing-models':
      return { document: 'install', main: holdsWindow ? null : 'install' }
    case 'ready':
      return { document: holdsWindow ? 'install' : 'app', main: holdsWindow ? null : 'app' }
    default: {
      const _exhaustive: never = phase
      return _exhaustive
    }
  }
}
