// Install awaits flush() so the opt-out write is on disk before `launched` is sent.
export function createAnalyticsOptOutWrites(
  writeEnabled: (enabled: boolean) => Promise<void>,
) {
  let pending: Promise<void> = Promise.resolve()
  let lastWriteFailed = false

  return {
    enqueue(enabled: boolean, onFailure: () => void): void {
      pending = pending.then(() =>
        writeEnabled(enabled).then(
          () => {
            lastWriteFailed = false
          },
          () => {
            lastWriteFailed = true
            onFailure()
          },
        ),
      )
    },
    async flush(): Promise<void> {
      // A toggle during Install appends a newer promise. Keep waiting until
      // the queue is the one we just finished, so launch sees the latest choice.
      let current = pending
      await current
      while (pending !== current) {
        current = pending
        await current
      }
      if (lastWriteFailed) {
        lastWriteFailed = false
        throw new Error('Couldn’t save the analytics choice. Please try again.')
      }
    },
  }
}
