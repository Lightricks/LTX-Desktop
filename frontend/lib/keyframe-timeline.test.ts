import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  findNearestFreeFrameIndex,
  formatKeyframeSeconds,
  formatKeyframeTimecode,
  frameFromPointer,
  keyframeAtOrBefore,
  lastFrameFromDuration,
  parseKeyframeSeconds,
  parseKeyframeTimecode,
  pickFreeFrameIndex,
  previewKeyframeForPlayhead,
  retimeKeyframes,
  retimeKeyframesForSettings,
  rulerTicks,
  sameDraggedFrame,
  withDraggedFrame,
} from './keyframe-timeline.ts'

type TestKeyframe = { frameIndex: number; id: string; path?: string }

function keyframe(frameIndex: number, id = `file-${frameIndex}`): TestKeyframe {
  return { frameIndex, id }
}

describe('lastFrameFromDuration', () => {
  it('matches backend compute_num_frames - 1 so stills cannot sit past the clip', () => {
    // Cross-language parity (no shared code with frame_math). Re-verify with:
    //   cd backend && uv run python -c "
    //   from frame_math import compute_num_frames
    //   for s, f in [(5,24),(5,25),(5,50),(10,24),(10,25),(10,50),(4,50)]: print(s, f, compute_num_frames(s,f)-1)"
    assert.equal(lastFrameFromDuration(5, 24), 120)
    assert.equal(lastFrameFromDuration(5, 25), 120)
    assert.equal(lastFrameFromDuration(5, 50), 248)
    assert.equal(lastFrameFromDuration(10, 24), 240)
    assert.equal(lastFrameFromDuration(10, 25), 248)
    assert.equal(lastFrameFromDuration(10, 50), 496)
    assert.equal(lastFrameFromDuration(4, 50), 200)
  })
})

describe('frameFromPointer', () => {
  const trackRect = { left: 100, width: 200 }
  const lastFrame = 100

  it('maps the left edge to frame 0', () => {
    assert.equal(frameFromPointer(100, trackRect, lastFrame), 0)
  })

  it('maps the right edge to lastFrame', () => {
    assert.equal(frameFromPointer(300, trackRect, lastFrame), lastFrame)
  })

  it('maps the midpoint to the middle frame', () => {
    assert.equal(frameFromPointer(200, trackRect, lastFrame), 50)
  })
})

describe('formatKeyframeTimecode', () => {
  const cases = [
    { fps: 24, frameIndex: 0, expected: '00:00.00' },
    { fps: 24, frameIndex: 23, expected: '00:00.23' },
    { fps: 24, frameIndex: 24, expected: '00:01.00' },
    { fps: 25, frameIndex: 24, expected: '00:00.24' },
    { fps: 25, frameIndex: 25, expected: '00:01.00' },
    { fps: 48, frameIndex: 49, expected: '00:01.01' },
    { fps: 50, frameIndex: 3005, expected: '01:00.05' },
  ] as const

  for (const { fps, frameIndex, expected } of cases) {
    it(`renders frame ${frameIndex} at ${fps} fps as ${expected}`, () => {
      assert.equal(formatKeyframeTimecode(frameIndex, fps), expected)
    })
  }
})

describe('parseKeyframeTimecode', () => {
  it('accepts a frames field only below the frame rate', () => {
    assert.equal(parseKeyframeTimecode('00:01.24', 25), 49)
    assert.equal(parseKeyframeTimecode('00:01.25', 25), null)
    assert.equal(parseKeyframeTimecode('00:01.25', 48), 73)
  })

  it('defaults a missing frames field to the first frame of that second', () => {
    assert.equal(parseKeyframeTimecode('01:02', 24), 62 * 24)
  })

  for (const value of ['', '1', '00:60.00', '00:01.', 'a:bb.cc', '00:01:02']) {
    it(`rejects ${JSON.stringify(value)}`, () => {
      assert.equal(parseKeyframeTimecode(value, 25), null)
    })
  }

  for (const fps of [24, 25, 48, 50]) {
    it(`round-trips every frame of two seconds at ${fps} fps`, () => {
      for (let frameIndex = 0; frameIndex < fps * 2; frameIndex++) {
        assert.equal(
          parseKeyframeTimecode(formatKeyframeTimecode(frameIndex, fps), fps),
          frameIndex,
        )
      }
    })
  }
})

describe('formatKeyframeSeconds', () => {
  it('shows seconds with two decimals and never minutes', () => {
    assert.equal(formatKeyframeSeconds(0, 24), '0.00s')
    assert.equal(formatKeyframeSeconds(36, 24), '1.50s')
    assert.equal(formatKeyframeSeconds(144, 24), '6.00s')
    assert.equal(formatKeyframeSeconds(1, 48), '0.02s')
    assert.equal(formatKeyframeSeconds(3005, 50), '60.10s')
  })
})

describe('parseKeyframeSeconds', () => {
  it('takes the nearest frame for seconds with or without the unit', () => {
    assert.equal(parseKeyframeSeconds('1.5', 24), 36)
    assert.equal(parseKeyframeSeconds('1.5s', 24), 36)
    assert.equal(parseKeyframeSeconds(' 2 s ', 24), 48)
    assert.equal(parseKeyframeSeconds('.5', 24), 12)
    assert.equal(parseKeyframeSeconds('1.', 24), 24)
    assert.equal(parseKeyframeSeconds('0.02', 48), 1)
  })

  for (const value of ['', 's', '-1', '1:30', '1,5', 'a', '1.5.2', '1 2']) {
    it(`rejects ${JSON.stringify(value)}`, () => {
      assert.equal(parseKeyframeSeconds(value, 24), null)
    })
  }

  for (const fps of [24, 25, 48, 50]) {
    it(`round-trips every frame of a 20 second clip at ${fps} fps, so a typed time can reach every frame`, () => {
      for (let frameIndex = 0; frameIndex <= fps * 20; frameIndex++) {
        assert.equal(
          parseKeyframeSeconds(formatKeyframeSeconds(frameIndex, fps), fps),
          frameIndex,
        )
      }
    })
  }
})

describe('rulerTicks', () => {
  it('gives no ticks, and ends, for a frame rate or a last frame that is not a number', () => {
    assert.deepEqual(rulerTicks(144, Number.NaN), [])
    assert.deepEqual(rulerTicks(Number.NaN, 24), [])
    assert.deepEqual(rulerTicks(144, Number.POSITIVE_INFINITY), [])
  })

  it('puts a labelled tick on each second and a small tick halfway, for a 6 s clip at 24 fps', () => {
    const ticks = rulerTicks(144, 24)

    assert.deepEqual(
      ticks.slice(0, 4),
      [
        { frame: 0, label: '0s' },
        { frame: 12, label: null },
        { frame: 24, label: '1s' },
        { frame: 36, label: null },
      ],
    )
    assert.deepEqual(ticks.at(-1), { frame: 144, label: '6s' })
    assert.equal(ticks.filter((tick) => tick.label !== null).length, 7)
  })

  it('labels every second second on a 20 s clip, so the labels stay apart', () => {
    const labels = rulerTicks(480, 24)
      .map((tick) => tick.label)
      .filter((label) => label !== null)

    assert.deepEqual(labels, ['0s', '2s', '4s', '6s', '8s', '10s', '12s', '14s', '16s', '18s', '20s'])
  })

  it('keeps every tick inside the clip', () => {
    for (const [lastFrame, fps] of [[120, 24], [120, 25], [96, 48], [200, 50]] as const) {
      for (const tick of rulerTicks(lastFrame, fps)) {
        assert.ok(tick.frame >= 0 && tick.frame <= lastFrame)
      }
    }
  })

  it('has no ticks without a clip', () => {
    assert.deepEqual(rulerTicks(0, 24), [])
    assert.deepEqual(rulerTicks(144, 0), [])
  })
})

describe('pickFreeFrameIndex', () => {
  it('uses the preferred frame when it is free', () => {
    assert.equal(pickFreeFrameIndex([keyframe(0), keyframe(100)], 200, 40), 40)
  })

  it('clamps a preferred frame past the end onto the last frame', () => {
    assert.equal(pickFreeFrameIndex([], 200, 500), 200)
  })

  it('falls back to the middle of the widest free stretch', () => {
    assert.equal(pickFreeFrameIndex([keyframe(0), keyframe(9)], 10, 0), 4)
  })

  it('spreads a burst of adds instead of clumping them at the playhead', () => {
    const taken: TestKeyframe[] = []
    for (let i = 0; i < 4; i++) {
      const frameIndex = pickFreeFrameIndex(taken, 100, 0)
      assert.notEqual(frameIndex, null)
      taken.push(keyframe(frameIndex!))
    }
    assert.equal(new Set(taken.map(({ frameIndex }) => frameIndex)).size, 4)
    assert.deepEqual(taken.map(({ frameIndex }) => frameIndex), [0, 50, 75, 25])
  })

  it('returns null when every frame is taken', () => {
    assert.equal(pickFreeFrameIndex([keyframe(0), keyframe(1), keyframe(2)], 2, 1), null)
  })

  it('ignores taken indices outside the legal range when checking fullness', () => {
    assert.equal(pickFreeFrameIndex([keyframe(-1), keyframe(100)], 1, 0), 0)
  })
})

describe('findNearestFreeFrameIndex', () => {
  it('never returns an occupied index', () => {
    const taken = [keyframe(10), keyframe(11), keyframe(12)]
    const occupied = new Set(taken.map(({ frameIndex }) => frameIndex))
    for (let target = 0; target <= 20; target++) {
      const result = findNearestFreeFrameIndex(taken, target, 20)
      if (result !== null) {
        assert.equal(occupied.has(result), false)
      }
    }
  })

  it('lands on an adjacent free frame when dropped on an occupied one', () => {
    assert.equal(findNearestFreeFrameIndex([keyframe(10)], 10, 100), 11)
    assert.equal(findNearestFreeFrameIndex([keyframe(10), keyframe(11)], 10, 100), 9)
  })

  it('does not jump to a far gap when a nearer frame is free', () => {
    assert.equal(findNearestFreeFrameIndex([keyframe(50)], 50, 100), 51)
    assert.notEqual(findNearestFreeFrameIndex([keyframe(50)], 50, 100), 0)
  })

  it('ignores taken indices outside the legal range', () => {
    assert.equal(findNearestFreeFrameIndex([keyframe(-1), keyframe(100)], 0, 1), 0)
  })
})

describe('retimeKeyframes', () => {
  it('keeps keyframes at the same relative place when the frame rate changes', () => {
    const retimed = retimeKeyframes([keyframe(0), keyframe(50)], 249, 499)
    assert.deepEqual(
      retimed.map((k) => k.frameIndex),
      [0, 100],
    )
  })

  it('scales keyframes across a shorter clip instead of packing the tail', () => {
    const retimed = retimeKeyframes(
      [keyframe(0), keyframe(100), keyframe(200)],
      200,
      100,
    )
    assert.deepEqual(
      retimed.map((k) => k.frameIndex),
      [0, 50, 100],
    )
  })

  it('scales keyframes across a longer clip instead of leaving them bunched', () => {
    const retimed = retimeKeyframes(
      [keyframe(0), keyframe(50), keyframe(100)],
      100,
      200,
    )
    assert.deepEqual(
      retimed.map((k) => k.frameIndex),
      [0, 100, 200],
    )
  })

  it('never collapses two keyframes onto one frame', () => {
    const retimed = retimeKeyframes([keyframe(10), keyframe(11)], 100, 100)
    assert.equal(new Set(retimed.map((k) => k.frameIndex)).size, 2)
    assert.deepEqual(
      retimed.map((k) => k.id),
      ['file-10', 'file-11'],
    )
  })

  it('drops excess keyframes when there are more keys than slots', () => {
    const retimed = retimeKeyframes(
      [keyframe(0, 'a'), keyframe(50, 'b'), keyframe(100, 'c')],
      100,
      1,
    )
    assert.equal(retimed.length, 2)
    assert.equal(new Set(retimed.map((k) => k.frameIndex)).size, 2)
    assert.deepEqual(
      retimed.map((k) => k.frameIndex),
      [0, 1],
    )
    assert.deepEqual(retimed.map((k) => k.id), ['a', 'c'])
  })
})

describe('retimeKeyframesForSettings', () => {
  it('retimes retained keyframes for settings changes made outside multi-keyframe mode', () => {
    const keyframes = [
      { id: 'opening', path: '/opening.png', frameIndex: 25 },
      { id: 'closing', path: '/closing.png', frameIndex: 200 },
    ]

    assert.deepEqual(
      retimeKeyframesForSettings(
        keyframes,
        { duration: 10, fps: 25 },
        { duration: 10, fps: 50 },
      ),
      [
        { id: 'opening', path: '/opening.png', frameIndex: 50 },
        { id: 'closing', path: '/closing.png', frameIndex: 400 },
      ],
    )
  })

  it('scales a 10s clip down to 5s at 24 fps without packing the tail', () => {
    const keyframes = [
      { id: 'opening', path: '/opening.png', frameIndex: 0 },
      { id: 'middle', path: '/middle.png', frameIndex: 120 },
      { id: 'closing', path: '/closing.png', frameIndex: 240 },
    ]

    const retimed = retimeKeyframesForSettings(
      keyframes,
      { duration: 10, fps: 24 },
      { duration: 5, fps: 24 },
    )

    assert.deepEqual(retimed.map(({ frameIndex }) => frameIndex), [0, 60, 120])
    assert.equal(new Set(retimed.map(({ frameIndex }) => frameIndex)).size, retimed.length)
    assert.deepEqual(
      retimed.map(({ id, path }) => ({ id, path })),
      keyframes.map(({ id, path }) => ({ id, path })),
    )
  })

  it('scales a 5s clip up to 10s at 24 fps instead of leaving keys bunched', () => {
    const retimed = retimeKeyframesForSettings(
      [
        { id: 'opening', path: '/opening.png', frameIndex: 0 },
        { id: 'middle', path: '/middle.png', frameIndex: 60 },
        { id: 'closing', path: '/closing.png', frameIndex: 120 },
      ],
      { duration: 5, fps: 24 },
      { duration: 10, fps: 24 },
    )

    assert.deepEqual(retimed.map(({ frameIndex }) => frameIndex), [0, 120, 240])
  })

  it('must not treat 10s-authored stills as a scale-up from the 5s remount default', () => {
    const authoredOn10s = [
      { id: 'opening', path: '/opening.png', frameIndex: 0 },
      { id: 'middle', path: '/middle.png', frameIndex: 120 },
      { id: 'closing', path: '/closing.png', frameIndex: 240 },
    ]
    const remountDefault = { duration: 5, fps: 24 }
    const recoveredClip = { duration: 10, fps: 24 }

    // Reload mid-generation restores stills that already live on the recovered
    // clip. If the duration/fps effect still thinks previous=5s, it scales
    // frame 240 as if it were the 5s tail and bunches keys at the new end.
    assert.deepEqual(
      retimeKeyframesForSettings(authoredOn10s, remountDefault, recoveredClip).map(
        ({ frameIndex }) => frameIndex,
      ),
      [0, 239, 240],
    )
    assert.deepEqual(
      retimeKeyframesForSettings(authoredOn10s, recoveredClip, recoveredClip).map(
        ({ frameIndex }) => frameIndex,
      ),
      [0, 120, 240],
    )
  })

  it('retimes and clamps when Auto changes to a numeric duration', () => {
    const retimed = retimeKeyframesForSettings(
      [keyframe(75), keyframe(100)],
      { duration: null, fps: 25 },
      { duration: 4, fps: 50 },
    )

    assert.deepEqual(retimed.map(({ frameIndex }) => frameIndex), [156, 200])
  })
})

describe('keyframeAtOrBefore', () => {
  const opening = keyframe(0, 'opening')
  const middle = keyframe(40, 'middle')
  const closing = keyframe(80, 'closing')

  it('returns the keyframe on the playhead', () => {
    assert.equal(keyframeAtOrBefore([opening, middle, closing], 40), middle)
  })

  it('holds the previous keyframe in a gap', () => {
    assert.equal(keyframeAtOrBefore([opening, middle, closing], 55), middle)
  })

  it('falls back to the first keyframe before any marker', () => {
    assert.equal(keyframeAtOrBefore([middle, closing], 10), middle)
  })

  it('returns undefined when there are no keyframes', () => {
    assert.equal(keyframeAtOrBefore([], 12), undefined)
  })
})

describe('withDraggedFrame', () => {
  const opening = keyframe(0, 'opening')
  const closing = keyframe(80, 'closing')

  it('leaves committed positions alone when nothing is dragging', () => {
    const keyframes = [opening, closing]
    assert.equal(withDraggedFrame(keyframes, null), keyframes)
  })

  it('overlays the in-flight frame on the dragged keyframe only', () => {
    const displayed = withDraggedFrame([opening, closing], { id: 'opening', frameIndex: 40 })
    assert.deepEqual(displayed.map(({ id, frameIndex }) => ({ id, frameIndex })), [
      { id: 'opening', frameIndex: 40 },
      { id: 'closing', frameIndex: 80 },
    ])
    assert.equal(displayed[1], closing)
  })
})

describe('previewKeyframeForPlayhead', () => {
  const opening = keyframe(0, 'opening')
  const closing = keyframe(80, 'closing')

  it('keeps showing the dragged still after it passes a later committed marker', () => {
    const preview = previewKeyframeForPlayhead(
      [opening, closing],
      90,
      { id: 'opening', frameIndex: 90 },
    )
    assert.equal(preview?.id, 'opening')
  })

  it('uses committed positions when the pointer is not dragging', () => {
    assert.equal(previewKeyframeForPlayhead([opening, closing], 90, null)?.id, 'closing')
  })
})

describe('sameDraggedFrame', () => {
  it('treats identical id+frame overlays as unchanged so setState can bail out', () => {
    assert.equal(
      sameDraggedFrame({ id: 'opening', frameIndex: 12 }, { id: 'opening', frameIndex: 12 }),
      true,
    )
    assert.equal(
      sameDraggedFrame({ id: 'opening', frameIndex: 12 }, { id: 'opening', frameIndex: 13 }),
      false,
    )
    assert.equal(sameDraggedFrame(null, null), true)
  })
})
