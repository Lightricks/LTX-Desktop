import assert from "node:assert/strict"
import { describe, it } from "node:test"

import {
  REMOTE_DEVICE_IN_USE_WITHIN_MS,
  isRemoteDeviceInUse,
} from "./remote-device-activity.ts"

describe("isRemoteDeviceInUse", () => {
  const now = 1_700_000_000_000

  it("treats a freshly seen device as in use", () => {
    assert.equal(isRemoteDeviceInUse(now, now), true)
    assert.equal(isRemoteDeviceInUse(now - REMOTE_DEVICE_IN_USE_WITHIN_MS, now), true)
  })

  it("treats a stale last-seen as idle", () => {
    assert.equal(
      isRemoteDeviceInUse(now - REMOTE_DEVICE_IN_USE_WITHIN_MS - 1, now),
      false,
    )
  })
})
