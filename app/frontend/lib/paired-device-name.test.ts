import assert from "node:assert/strict"
import { describe, it } from "node:test"

import { pairedDeviceDisplayName } from "./paired-device-name.ts"

describe("pairedDeviceDisplayName", () => {
  it("keeps names that are already human", () => {
    assert.equal(pairedDeviceDisplayName("Test device"), "Test device")
    assert.equal(pairedDeviceDisplayName("Chrome on Mac"), "Chrome on Mac")
    assert.equal(pairedDeviceDisplayName("Phone UA"), "Phone UA")
  })

  it("falls back when the API name is empty", () => {
    assert.equal(pairedDeviceDisplayName("  "), "Paired device")
    assert.equal(pairedDeviceDisplayName(""), "Paired device")
  })

  it("caps overly long labels", () => {
    assert.equal(pairedDeviceDisplayName("x".repeat(100)).length, 80)
  })
})
