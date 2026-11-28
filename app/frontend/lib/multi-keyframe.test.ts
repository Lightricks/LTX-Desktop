import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { persistedKeyframeSchema } from '../types/project-model.ts'
import { pickFreeFrameIndex } from './keyframe-timeline.ts'
import {
  appendKeyframePaths,
  applyKeyframeImagePaths,
  DEFAULT_KEYFRAME_STRENGTH,
  MISSING_KEYFRAME_STRENGTH,
  fromPersistedKeyframes,
  toPersistedKeyframes,
  videoGenerationModeFromInputs,
  enhanceKeyframesPayload,
  type KeyframeItem,
} from './multi-keyframe.ts'

function item(
  id: string,
  path: string,
  frameIndex: number,
  strength = DEFAULT_KEYFRAME_STRENGTH,
): KeyframeItem {
  return { id, path, frameIndex, strength }
}

describe('rapid keyframe placement', () => {
  it('never assigns the same frame index twice', () => {
    const placed: { frameIndex: number }[] = []

    for (let count = 0; count < 5; count++) {
      const frameIndex = pickFreeFrameIndex(placed, 120, 0)
      assert.notEqual(frameIndex, null)
      placed.push({ frameIndex: frameIndex! })
    }

    assert.equal(new Set(placed.map(({ frameIndex }) => frameIndex)).size, placed.length)
  })
})

describe('appendKeyframePaths', () => {
  it('spreads a burst of images across free frames', () => {
    let nextId = 0
    const added = appendKeyframePaths(
      [],
      ['/a.png', '/b.png', '/c.png'],
      120,
      0,
      5,
      () => `id-${nextId++}`,
    )

    assert.deepEqual(
      added.map(({ path, frameIndex, strength }) => ({ path, frameIndex, strength })),
      [
        { path: '/a.png', frameIndex: 0, strength: DEFAULT_KEYFRAME_STRENGTH },
        { path: '/b.png', frameIndex: 60, strength: DEFAULT_KEYFRAME_STRENGTH },
        { path: '/c.png', frameIndex: 90, strength: DEFAULT_KEYFRAME_STRENGTH },
      ],
    )
    assert.equal(new Set(added.map(({ frameIndex }) => frameIndex)).size, 3)
  })

  it('stops at the remaining cap', () => {
    const existing: KeyframeItem[] = [
      item('opening', '/opening.png', 0),
      item('closing', '/closing.png', 10),
    ]

    const added = appendKeyframePaths(
      existing,
      ['/a.png', '/b.png', '/c.png'],
      20,
      0,
      3,
      () => 'extra',
    )

    assert.equal(added.length, 3)
    assert.equal(added.at(-1)?.path, '/a.png')
  })
})

describe('applyKeyframeImagePaths', () => {
  const existing: KeyframeItem[] = [
    item('opening', '/opening.png', 0),
  ]

  it('replaces one marker and appends leftover files', () => {
    let nextId = 0
    const next = applyKeyframeImagePaths({
      keyframes: existing,
      paths: ['/new-opening.png', '/middle.png'],
      replaceId: 'opening',
      lastFrame: 80,
      preferredFrame: 0,
      maxCount: 5,
      createId: () => `id-${nextId++}`,
    })

    assert.equal(next[0]?.path, '/new-opening.png')
    assert.equal(next[0]?.frameIndex, 0)
    assert.equal(next[1]?.path, '/middle.png')
    assert.notEqual(next[1]?.frameIndex, 0)
  })

  it('replaces without appending when duration is unknown', () => {
    const next = applyKeyframeImagePaths({
      keyframes: existing,
      paths: ['/new-opening.png', '/ignored.png'],
      replaceId: 'opening',
      lastFrame: null,
      preferredFrame: 0,
      maxCount: 5,
    })

    assert.deepEqual(next, [item('opening', '/new-opening.png', 0)])
  })

  it('keeps the still\'s owned strength when replacing its image', () => {
    const next = applyKeyframeImagePaths({
      keyframes: [item('opening', '/opening.png', 0, 0.7)],
      paths: ['/new-opening.png'],
      replaceId: 'opening',
      lastFrame: 80,
      preferredFrame: 0,
      maxCount: 5,
    })

    assert.deepEqual(next, [item('opening', '/new-opening.png', 0, 0.7)])
  })
})

describe('persisted keyframes', () => {
  it('drops UI ids when snapshotting for generationParams', () => {
    assert.deepEqual(
      toPersistedKeyframes([
        item('opening', '/opening.png', 0),
        item('closing', '/closing.png', 80, 0.7),
      ]),
      [
        { path: '/opening.png', frameIndex: 0, strength: DEFAULT_KEYFRAME_STRENGTH },
        { path: '/closing.png', frameIndex: 80, strength: 0.7 },
      ],
    )
  })

  it('restores items with fresh ids and owned strength', () => {
    let nextId = 0
    const restored = fromPersistedKeyframes(
      [{ path: '/opening.png', frameIndex: 0, strength: 0.7 }],
      () => `id-${nextId++}`,
    )
    assert.deepEqual(restored, [item('id-0', '/opening.png', 0, 0.7)])
  })

  it('fills missing persisted strength with a full lock, not the new-still default', () => {
    let nextId = 0
    const restored = fromPersistedKeyframes(
      [{ path: '/opening.png', frameIndex: 0 }],
      () => `id-${nextId++}`,
    )
    assert.deepEqual(restored, [item('id-0', '/opening.png', 0, MISSING_KEYFRAME_STRENGTH)])
    assert.notEqual(MISSING_KEYFRAME_STRENGTH, DEFAULT_KEYFRAME_STRENGTH)
  })

  it('persists and restores a zero lock instead of treating it as missing', () => {
    assert.deepEqual(
      toPersistedKeyframes([item('opening', '/opening.png', 0, 0)]),
      [{ path: '/opening.png', frameIndex: 0, strength: 0 }],
    )
    let nextId = 0
    const restored = fromPersistedKeyframes(
      [{ path: '/opening.png', frameIndex: 0, strength: 0 }],
      () => `id-${nextId++}`,
    )
    assert.deepEqual(restored, [item('id-0', '/opening.png', 0, 0)])
  })

  it('clamps restored strength to the 0-1 lock range', () => {
    let nextId = 0
    const restored = fromPersistedKeyframes(
      [
        { path: '/hi.png', frameIndex: 0, strength: 1.5 },
        { path: '/lo.png', frameIndex: 40, strength: -0.2 },
      ],
      () => `id-${nextId++}`,
    )
    assert.deepEqual(restored, [
      item('id-0', '/hi.png', 0, 1),
      item('id-1', '/lo.png', 40, 0),
    ])
  })
})

describe('videoGenerationModeFromInputs', () => {
  it('prefers multi-keyframe over image and audio inputs', () => {
    assert.equal(
      videoGenerationModeFromInputs({
        keyframes: [{ path: '/opening.png', frameIndex: 0 }],
        audioUrl: '/clip.mp3',
        imageUrl: '/still.png',
      }),
      'multi-keyframe',
    )
  })

  it('falls back to text-to-video when nothing is attached', () => {
    assert.equal(videoGenerationModeFromInputs({}), 'text-to-video')
  })

  it('treats a last frame alone as image-to-video', () => {
    assert.equal(
      videoGenerationModeFromInputs({ lastImageUrl: '/end.png' }),
      'image-to-video',
    )
  })

  it('prefers audio over a still', () => {
    assert.equal(
      videoGenerationModeFromInputs({
        audioUrl: '/clip.mp3',
        imageUrl: '/still.png',
        lastImageUrl: '/end.png',
      }),
      'audio-to-video',
    )
  })
})

describe('enhanceKeyframesPayload', () => {
  it('returns undefined when there are no stills', () => {
    assert.equal(enhanceKeyframesPayload([]), undefined)
  })

  it('maps a single still onto the enhance keyframe list', () => {
    assert.deepEqual(
      enhanceKeyframesPayload([item('opening', '/opening.png', 40)]),
      [{ imagePath: '/opening.png', frameIndex: 40, strength: DEFAULT_KEYFRAME_STRENGTH }],
    )
  })

  it('sends every still, including middle markers, in frame order', () => {
    assert.deepEqual(
      enhanceKeyframesPayload([
        item('closing', '/closing.png', 80),
        item('opening', '/opening.png', 0),
        item('middle', '/middle.png', 40),
      ]),
      [
        { imagePath: '/opening.png', frameIndex: 0, strength: DEFAULT_KEYFRAME_STRENGTH },
        { imagePath: '/middle.png', frameIndex: 40, strength: DEFAULT_KEYFRAME_STRENGTH },
        { imagePath: '/closing.png', frameIndex: 80, strength: DEFAULT_KEYFRAME_STRENGTH },
      ],
    )
  })

  it('forwards each still\'s owned strength instead of hardcoding the default', () => {
    assert.deepEqual(
      enhanceKeyframesPayload([item('opening', '/opening.png', 0, 0.7)]),
      [{ imagePath: '/opening.png', frameIndex: 0, strength: 0.7 }],
    )
  })

  it('forwards a zero lock instead of treating it as missing', () => {
    assert.deepEqual(
      enhanceKeyframesPayload([item('opening', '/opening.png', 0, 0)]),
      [{ imagePath: '/opening.png', frameIndex: 0, strength: 0 }],
    )
  })

  it('clamps out-of-range strength on the enhance payload', () => {
    assert.deepEqual(
      enhanceKeyframesPayload([
        item('hi', '/hi.png', 0, 1.5),
        item('lo', '/lo.png', 40, -0.2),
      ]),
      [
        { imagePath: '/hi.png', frameIndex: 0, strength: 1 },
        { imagePath: '/lo.png', frameIndex: 40, strength: 0 },
      ],
    )
  })
})

describe('persistedKeyframeSchema', () => {
  it('defaults missing strength to a full lock', () => {
    assert.deepEqual(
      persistedKeyframeSchema.parse({ path: '/opening.png', frameIndex: 0 }),
      { path: '/opening.png', frameIndex: 0, strength: MISSING_KEYFRAME_STRENGTH },
    )
  })

  it('keeps a zero lock and clamps out of range so project parse cannot fail', () => {
    assert.equal(
      persistedKeyframeSchema.parse({ path: '/opening.png', frameIndex: 0, strength: 0 }).strength,
      0,
    )
    assert.equal(
      persistedKeyframeSchema.parse({ path: '/opening.png', frameIndex: 0, strength: 1.5 }).strength,
      1,
    )
    assert.equal(
      persistedKeyframeSchema.parse({ path: '/opening.png', frameIndex: 0, strength: -0.2 }).strength,
      0,
    )
  })
})
