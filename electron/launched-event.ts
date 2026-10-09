export type LaunchedTrigger = 'startup' | 'installation-started' | 'setup-completed'

// Startup sends only after setup, when the opt-out choice is already stored.
// The Install click sends on a first install, once the renderer has written that choice.
// Setup completion covers an API-only first run that never shows that screen.
// A second trigger in the same process does not send again.
export function shouldSendLaunched(input: {
  alreadySent: boolean
  setupComplete: boolean
  trigger: LaunchedTrigger
}): boolean {
  if (input.alreadySent) return false
  if (input.trigger === 'startup') return input.setupComplete
  return true
}
