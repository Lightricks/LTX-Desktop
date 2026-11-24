import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  autoDurationOptionVisible,
  canUseMultiKeyframeMode,
  fallbackGenSpaceMode,
  genSpaceUsesAudioInput,
  isGenSpaceLibraryMode,
  isEnhanceAvailableForMode,
  modeAfterCompletedGeneration,
  modeOptionValues,
} from './genspace-multi-keyframe.ts'

const allModesAvailable = {
  canUseMultiKeyframe: true,
  canUseRetake: true,
  canUseExtend: true,
  canUseIcLora: true,
}

describe('autoDurationOptionVisible', () => {
  it('shows Auto only in video mode when the model supports it', () => {
    assert.equal(autoDurationOptionVisible('multi-keyframe', true), false)
    assert.equal(autoDurationOptionVisible('video', false), false)
    assert.equal(autoDurationOptionVisible('video', true), true)
  })
})

describe('canUseMultiKeyframeMode', () => {
  it('is true only in local mode when the capability is enabled', () => {
    assert.equal(
      canUseMultiKeyframeMode({
        isLocalMode: false,
        localCaps: { multi_keyframe: true },
      }),
      false,
    )
    assert.equal(
      canUseMultiKeyframeMode({
        isLocalMode: true,
        localCaps: {},
      }),
      false,
    )
    assert.equal(
      canUseMultiKeyframeMode({
        isLocalMode: true,
        localCaps: { multi_keyframe: false },
      }),
      false,
    )
    assert.equal(
      canUseMultiKeyframeMode({
        isLocalMode: true,
        localCaps: { multi_keyframe: true },
      }),
      true,
    )
  })
})

describe('fallbackGenSpaceMode', () => {
  it('falls back to video when a tool mode is unavailable and keeps it when available', () => {
    assert.equal(
      fallbackGenSpaceMode('multi-keyframe', {
        ...allModesAvailable,
        canUseMultiKeyframe: false,
      }),
      'video',
    )

    for (const [mode, flag] of [
      ['retake', 'canUseRetake'],
      ['extend', 'canUseExtend'],
      ['ic-lora', 'canUseIcLora'],
    ] as const) {
      assert.equal(
        fallbackGenSpaceMode(mode, {
          ...allModesAvailable,
          [flag]: false,
        }),
        'video',
      )
      assert.equal(fallbackGenSpaceMode(mode, allModesAvailable), mode)
    }
  })
})

describe('modeOptionValues', () => {
  it('excludes multi-keyframe when unavailable', () => {
    const values = modeOptionValues({
      ...allModesAvailable,
      canUseMultiKeyframe: false,
    })

    assert.equal(values.includes('multi-keyframe'), false)
  })

  it('includes multi-keyframe immediately after video when available', () => {
    const values = modeOptionValues(allModesAvailable)

    assert.deepEqual(values.slice(0, 3), ['image', 'video', 'multi-keyframe'])
  })
})

describe('isGenSpaceLibraryMode', () => {
  it('shows the asset library for image and video and hides it in tool modes', () => {
    assert.equal(isGenSpaceLibraryMode('image'), true)
    assert.equal(isGenSpaceLibraryMode('video'), true)
    assert.equal(isGenSpaceLibraryMode('multi-keyframe'), false)
    assert.equal(isGenSpaceLibraryMode('retake'), false)
    assert.equal(isGenSpaceLibraryMode('extend'), false)
    assert.equal(isGenSpaceLibraryMode('ic-lora'), false)
  })
})

describe('modeAfterCompletedGeneration', () => {
  it('returns to video gen space after a multi-keyframe job', () => {
    assert.equal(modeAfterCompletedGeneration('multi-keyframe'), 'video')
  })

  it('does not force a mode change for ordinary video jobs', () => {
    assert.equal(modeAfterCompletedGeneration('text-to-video'), null)
    assert.equal(modeAfterCompletedGeneration('image-to-video'), null)
    assert.equal(modeAfterCompletedGeneration('audio-to-video'), null)
  })
})

describe('isEnhanceAvailableForMode', () => {
  it('includes video, image, multi-keyframe, and ic-lora, and hides retake and extend', () => {
    assert.equal(isEnhanceAvailableForMode('multi-keyframe'), true)
    assert.equal(isEnhanceAvailableForMode('video'), true)
    assert.equal(isEnhanceAvailableForMode('image'), true)
    assert.equal(isEnhanceAvailableForMode('ic-lora'), true)
    assert.equal(isEnhanceAvailableForMode('retake'), false)
    assert.equal(isEnhanceAvailableForMode('extend'), false)
  })
})

describe('genSpaceUsesAudioInput', () => {
  it('is only video mode, so leftover A2V audio cannot ride along with keyframes', () => {
    assert.equal(genSpaceUsesAudioInput('video'), true)
    assert.equal(genSpaceUsesAudioInput('multi-keyframe'), false)
    assert.equal(genSpaceUsesAudioInput('image'), false)
    assert.equal(genSpaceUsesAudioInput('retake'), false)
    assert.equal(genSpaceUsesAudioInput('extend'), false)
    assert.equal(genSpaceUsesAudioInput('ic-lora'), false)
  })
})
