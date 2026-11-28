import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  HOME_SIDEBAR_COLLAPSE_DELAY_MS,
  HOME_SIDEBAR_EXPANDED_WIDTH_PX,
  applyHomeSidebarPinChange,
  createHomeSidebarCollapseDelay,
  HOME_SIDEBAR_COMPACT_MAX_WIDTH_PX,
  homeSidebarContentOffsetPx,
  homeSidebarPinLabel,
  isHomeSidebarOpen,
  matchesHomeSidebarCompact,
} from './home-sidebar-expansion.ts'

const closedIdle = {
  pinned: false,
  hovered: false,
  focused: false,
  menuOpen: false,
} as const

describe('isHomeSidebarOpen', () => {
  it('stays closed without pin, hover, focus, or an open menu', () => {
    assert.equal(isHomeSidebarOpen(closedIdle), false)
  })

  it('stays open while pinned even without hover or focus', () => {
    assert.equal(
      isHomeSidebarOpen({ ...closedIdle, pinned: true }),
      true,
    )
  })

  it('opens temporarily on hover, focus, or an open menu', () => {
    assert.equal(isHomeSidebarOpen({ ...closedIdle, hovered: true }), true)
    assert.equal(isHomeSidebarOpen({ ...closedIdle, focused: true }), true)
    assert.equal(isHomeSidebarOpen({ ...closedIdle, menuOpen: true }), true)
  })
})

describe('homeSidebarPinLabel', () => {
  it('labels the toggle action, not disclosure', () => {
    assert.equal(homeSidebarPinLabel(true), 'Unpin sidebar')
    assert.equal(homeSidebarPinLabel(false), 'Pin sidebar')
  })
})

describe('home sidebar overlay geometry', () => {
  it('keeps content full-width until the sidebar is pinned', () => {
    assert.equal(HOME_SIDEBAR_EXPANDED_WIDTH_PX, 240)
    assert.equal(homeSidebarContentOffsetPx(false), 0)
    assert.equal(homeSidebarContentOffsetPx(true), HOME_SIDEBAR_EXPANDED_WIDTH_PX)
  })
})

describe('matchesHomeSidebarCompact', () => {
  it('reads the compact sidebar breakpoint from matchMedia', () => {
    const queries: string[] = []
    const win = {
      matchMedia(query: string) {
        queries.push(query)
        return {
          matches: true,
          addEventListener: () => {},
          removeEventListener: () => {},
        }
      },
    }
    assert.equal(matchesHomeSidebarCompact(win), true)
    assert.deepEqual(queries, [`(max-width: ${HOME_SIDEBAR_COMPACT_MAX_WIDTH_PX}px)`])
  })

  it('falls back to the rail layout without a window or matchMedia', () => {
    assert.equal(matchesHomeSidebarCompact(undefined), false)
    assert.equal(matchesHomeSidebarCompact({}), false)
  })
})

describe('applyHomeSidebarPinChange', () => {
  it('clears transient hover and focus when unpinning so the overlay can hide immediately', () => {
    assert.deepEqual(
      applyHomeSidebarPinChange(
        {
          pinned: true,
          hovered: true,
          focused: true,
          menuOpen: false,
        },
        false,
      ),
      {
        pinned: false,
        hovered: false,
        focused: false,
        menuOpen: false,
      },
    )
  })

  it('keeps hover, focus, and Support-menu expansion when pinning', () => {
    assert.deepEqual(
      applyHomeSidebarPinChange(
        {
          pinned: false,
          hovered: true,
          focused: true,
          menuOpen: true,
        },
        true,
      ),
      {
        pinned: true,
        hovered: true,
        focused: true,
        menuOpen: true,
      },
    )
  })
})

describe('createHomeSidebarCollapseDelay', () => {
  it('delays collapse and cancels a pending timer on re-enter or cleanup', () => {
    const scheduled: Array<{ id: number; ms: number; fn: () => void }> = []
    let nextId = 1
    const delay = createHomeSidebarCollapseDelay({
      delayMs: HOME_SIDEBAR_COLLAPSE_DELAY_MS,
      setTimeoutFn: (fn, ms) => {
        const id = nextId++
        scheduled.push({ id, ms, fn })
        return id
      },
      clearTimeoutFn: (id) => {
        const index = scheduled.findIndex((entry) => entry.id === id)
        if (index >= 0) scheduled.splice(index, 1)
      },
    })

    let collapsed = false
    delay.schedule(() => {
      collapsed = true
    })
    assert.equal(scheduled.length, 1)
    assert.equal(scheduled[0]?.ms, HOME_SIDEBAR_COLLAPSE_DELAY_MS)
    assert.equal(collapsed, false)

    delay.cancel()
    assert.equal(scheduled.length, 0)

    delay.schedule(() => {
      collapsed = true
    })
    delay.schedule(() => {
      collapsed = true
    })
    assert.equal(scheduled.length, 1)
    scheduled[0]?.fn()
    assert.equal(collapsed, true)

    collapsed = false
    delay.schedule(() => {
      collapsed = true
    })
    delay.cancel()
    assert.equal(collapsed, false)
  })
})
