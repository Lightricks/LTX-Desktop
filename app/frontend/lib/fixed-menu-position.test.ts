import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { fixedMenuPosition } from './fixed-menu-position.ts'

const trigger = { left: 100, right: 180, top: 500, bottom: 532 }
const viewport = { width: 1000, height: 800 }

describe('fixedMenuPosition', () => {
  it('anchors a menu above the trigger', () => {
    const pos = fixedMenuPosition({
      trigger,
      placement: 'above',
      viewport,
      menuWidth: 160,
      gap: 8,
    })
    assert.equal(pos.left, 100)
    assert.equal(pos.bottom, 308)
    assert.equal(pos.top, undefined)
  })

  it('anchors a menu below the trigger', () => {
    const pos = fixedMenuPosition({
      trigger,
      placement: 'below',
      viewport,
      menuWidth: 160,
      gap: 8,
    })
    assert.equal(pos.left, 100)
    assert.equal(pos.top, 540)
    assert.equal(pos.bottom, undefined)
  })

  it('shifts left so a wide menu stays on-screen', () => {
    const pos = fixedMenuPosition({
      trigger: { left: 900, right: 980, top: 500, bottom: 532 },
      placement: 'above',
      viewport,
      menuWidth: 160,
      gap: 8,
    })
    assert.equal(pos.left, 820)
  })

  it('does not use an assumed width when the menu has not been measured', () => {
    const pos = fixedMenuPosition({
      trigger: { left: 900, right: 980, top: 500, bottom: 532 },
      placement: 'above',
      viewport,
      gap: 8,
    })
    assert.equal(pos.left, 900)
  })

  it('ignores a viewport-filling width from an unpositioned block menu', () => {
    const pos = fixedMenuPosition({
      trigger: { left: 400, right: 480, top: 500, bottom: 532 },
      placement: 'above',
      viewport,
      menuWidth: 1000,
      gap: 8,
    })
    assert.equal(pos.left, 400)
  })
})
