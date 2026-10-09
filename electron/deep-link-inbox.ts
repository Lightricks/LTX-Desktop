import {
  findDeepLinkInArgv,
  isDeepLinkIntentFresh,
  parseDeepLink,
  type DeepLinkArrival,
  type DeepLinkIntent,
  type DeepLinkTarget,
} from '../shared/deep-link.ts'

export type DeepLinkInboxDeps = {
  argv: readonly string[]
  now: () => number
  newId: () => string
  onOpenUrl: (listener: (url: string) => void) => void
  onSecondInstance: (listener: (commandLine: readonly string[]) => void) => void
  ensureWindow: () => void
  publish: (intent: DeepLinkIntent) => void
  log?: (message: string) => void
}

export type DeepLinkInbox = {
  take: () => DeepLinkIntent | null
}

export function createDeepLinkInbox(deps: DeepLinkInboxDeps): DeepLinkInbox {
  let pending: DeepLinkIntent | null = null

  function take(): DeepLinkIntent | null {
    if (!pending) return null
    if (!isDeepLinkIntentFresh(pending, deps.now())) {
      deps.log?.('Dropped expired deep link')
      pending = null
      return null
    }
    const intent = pending
    pending = null
    return intent
  }

  function accept(target: DeepLinkTarget | null, arrival: DeepLinkArrival): DeepLinkIntent | null {
    if (!target) {
      if (arrival === 'open-url') {
        deps.log?.(`Ignored deep-link URL (${arrival})`)
      }
      return null
    }
    pending = {
      id: deps.newId(),
      target,
      receivedAt: deps.now(),
      arrival,
    }
    deps.publish(pending)
    deps.ensureWindow()
    deps.log?.(`Queued ${target.kind} ${target.slug} from ${arrival} as ${pending.id}`)
    return pending
  }

  deps.onOpenUrl((url) => {
    accept(parseDeepLink(url), 'open-url')
  })
  deps.onSecondInstance((commandLine) => {
    if (!accept(findDeepLinkInArgv(commandLine), 'second-instance')) {
      deps.ensureWindow()
    }
  })
  accept(findDeepLinkInArgv(deps.argv), 'cold-start')

  return { take }
}

let session: DeepLinkInbox | null = null

export function installDeepLinkInbox(deps: DeepLinkInboxDeps): DeepLinkInbox {
  if (!session) session = createDeepLinkInbox(deps)
  return session
}

export function takePendingDeepLink(): DeepLinkIntent | null {
  return session?.take() ?? null
}
