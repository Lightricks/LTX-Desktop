export function shouldHoldRemoteSleepBlocker(remoteOn: boolean, onBattery: boolean): boolean {
  // Idle sleep is blocked only on AC. Lid close / Sleep / battery still sleep.
  // TODO: also hold while a local generation is running (see notifyGenerationActive
  // in frontend/lib/generation-active.ts). Same AC-only rule: never on battery.
  return remoteOn && !onBattery
}

export type RemoteKeepAwakePower = {
  isOnBatteryPower: () => boolean
  onAc: (listener: () => void) => void
  onBattery: (listener: () => void) => void
}

export type RemoteKeepAwakeBlocker = {
  start: () => number
  stop: (id: number) => void
  isStarted: (id: number) => boolean
}

export type RemoteKeepAwakeDeps = {
  power: RemoteKeepAwakePower
  blocker: RemoteKeepAwakeBlocker
  log?: (message: string) => void
}

export function createRemoteKeepAwake(deps: RemoteKeepAwakeDeps) {
  let remoteOn = false
  let blockerId: number | null = null
  let listening = false

  function sync(): void {
    const want = shouldHoldRemoteSleepBlocker(remoteOn, deps.power.isOnBatteryPower())
    if (want) {
      if (blockerId !== null && deps.blocker.isStarted(blockerId)) return
      blockerId = deps.blocker.start()
      deps.log?.('Remote keep-awake: prevent-app-suspension started')
      return
    }
    if (blockerId === null) return
    if (deps.blocker.isStarted(blockerId)) {
      deps.blocker.stop(blockerId)
    }
    blockerId = null
    deps.log?.('Remote keep-awake: prevent-app-suspension stopped')
  }

  return {
    setRemoteExposureActive(active: boolean): void {
      remoteOn = active
      sync()
    },
    init(): void {
      if (listening) return
      listening = true
      deps.power.onAc(sync)
      deps.power.onBattery(sync)
      sync()
    },
    stop(): void {
      remoteOn = false
      sync()
    },
  }
}

type RemoteKeepAwakeSession = ReturnType<typeof createRemoteKeepAwake>

let session: RemoteKeepAwakeSession | null = null

export function installRemoteKeepAwake(deps: RemoteKeepAwakeDeps): void {
  if (session) return
  session = createRemoteKeepAwake(deps)
}

export function initRemoteKeepAwake(): void {
  session?.init()
}

export function setRemoteExposureActive(active: boolean): void {
  session?.setRemoteExposureActive(active)
}

export function stopRemoteKeepAwake(): void {
  session?.stop()
}
