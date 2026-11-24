import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { DEFAULT_KEYFRAME_STRENGTH, type KeyframeItem } from './multi-keyframe.ts'
import {
  buildGenerateVideoBody,
  buildGenerateVideoImageInputs,
  type GenerateVideoBodySettings,
} from './build-generate-video-body.ts'

const keyframes: KeyframeItem[] = [
  { id: 'opening', path: '/frames/opening.png', frameIndex: 0, strength: DEFAULT_KEYFRAME_STRENGTH },
  { id: 'ending', path: '/frames/ending.png', frameIndex: 121, strength: DEFAULT_KEYFRAME_STRENGTH },
]

const settings: GenerateVideoBodySettings = {
  model: 'fast',
  duration: 5,
  videoResolution: '540p',
  fps: 24,
  audio: true,
  cameraMotion: 'static',
}

describe('buildGenerateVideoImageInputs', () => {
  it('maps multi-keyframes to the backend request shape', () => {
    assert.deepEqual(
      buildGenerateVideoImageInputs({
        mode: 'multi-keyframe',
        imagePath: null,
        lastImagePath: null,
        keyframes,
      }),
      {
        keyframes: [
          { imagePath: '/frames/opening.png', frameIndex: 0, strength: DEFAULT_KEYFRAME_STRENGTH },
          { imagePath: '/frames/ending.png', frameIndex: 121, strength: DEFAULT_KEYFRAME_STRENGTH },
        ],
      },
    )
  })

  it('omits image inputs when multi-keyframes are present', () => {
    const result = buildGenerateVideoImageInputs({
      mode: 'multi-keyframe',
      imagePath: '/frames/first.png',
      lastImagePath: '/frames/last.png',
      keyframes,
    })

    assert.equal('imagePath' in result, false)
    assert.equal('lastImagePath' in result, false)
    assert.equal('keyframes' in result, true)
  })

  it('forwards each still\'s owned strength on generate', () => {
    assert.deepEqual(
      buildGenerateVideoImageInputs({
        mode: 'multi-keyframe',
        imagePath: null,
        lastImagePath: null,
        keyframes: [
          { id: 'opening', path: '/frames/opening.png', frameIndex: 0, strength: 0.7 },
        ],
      }),
      {
        keyframes: [{ imagePath: '/frames/opening.png', frameIndex: 0, strength: 0.7 }],
      },
    )
  })

  it('preserves image inputs outside multi-keyframe mode', () => {
    assert.deepEqual(
      buildGenerateVideoImageInputs({
        mode: 'video',
        imagePath: '/frames/first.png',
        lastImagePath: '/frames/last.png',
        keyframes,
      }),
      {
        imagePath: '/frames/first.png',
        lastImagePath: '/frames/last.png',
      },
    )
  })
})

describe('buildGenerateVideoBody', () => {
  it('submits a typed prompt with its provenance so the backend enhances it', () => {
    const body = buildGenerateVideoBody({
      prompt: 'a cat',
      promptProvenance: 'typed',
      settings,
      imagePath: null,
      lastImagePath: null,
    })

    assert.equal(body.prompt, 'a cat')
    assert.equal(body.promptProvenance, 'typed')
  })

  it('forwards enhanced provenance so Generate does not rewrite a rewrite', () => {
    const body = buildGenerateVideoBody({
      prompt: 'a long descriptive caption',
      promptProvenance: 'enhanced',
      settings,
      imagePath: null,
      lastImagePath: null,
    })

    assert.equal(body.promptProvenance, 'enhanced')
  })

  it('maps the generation settings onto the backend field names', () => {
    const body = buildGenerateVideoBody({
      prompt: 'a cat',
      promptProvenance: 'typed',
      settings: { ...settings, aspectRatio: '9:16', negativePrompt: 'blurry' },
      imagePath: '/frames/first.png',
      lastImagePath: '/frames/last.png',
    })

    assert.deepEqual(body, {
      prompt: 'a cat',
      promptProvenance: 'typed',
      model: 'fast',
      duration: 5,
      resolution: '540p',
      fps: 24,
      audio: true,
      cameraMotion: 'static',
      negativePrompt: 'blurry',
      aspectRatio: '9:16',
      imagePath: '/frames/first.png',
      lastImagePath: '/frames/last.png',
    })
  })

  it('defaults the negative prompt and aspect ratio the backend requires', () => {
    const body = buildGenerateVideoBody({
      prompt: 'a cat',
      promptProvenance: 'typed',
      settings: { ...settings, aspectRatio: '' },
      imagePath: null,
      lastImagePath: null,
    })

    assert.equal(body.negativePrompt, '')
    assert.equal(body.aspectRatio, '16:9')
  })

  it('routes multi-keyframe inputs instead of the single-image fields', () => {
    const body = buildGenerateVideoBody({
      prompt: 'a cat',
      promptProvenance: 'typed',
      settings,
      imagePath: '/frames/first.png',
      lastImagePath: null,
      imageInputs: { mode: 'multi-keyframe', keyframes },
    })

    assert.equal('imagePath' in body, false)
    assert.deepEqual(body.keyframes, [
      { imagePath: '/frames/opening.png', frameIndex: 0, strength: DEFAULT_KEYFRAME_STRENGTH },
      { imagePath: '/frames/ending.png', frameIndex: 121, strength: DEFAULT_KEYFRAME_STRENGTH },
    ])
  })

  it('omits the optional audio and lora fields when there is nothing to send', () => {
    const body = buildGenerateVideoBody({
      prompt: 'a cat',
      promptProvenance: 'typed',
      settings: { ...settings, loras: [] },
      imagePath: null,
      lastImagePath: null,
      audioPath: null,
    })

    assert.equal('audioPath' in body, false)
    assert.equal('loras' in body, false)
  })

  it('drops the display-only name of each selected lora', () => {
    const body = buildGenerateVideoBody({
      prompt: 'a cat',
      promptProvenance: 'typed',
      settings: {
        ...settings,
        loras: [{ ref: 'style.safetensors', scale: 0.8, name: 'Style' }],
      },
      imagePath: null,
      lastImagePath: null,
      audioPath: '/audio/track.wav',
    })

    assert.equal(body.audioPath, '/audio/track.wav')
    assert.deepEqual(body.loras, [{ ref: 'style.safetensors', scale: 0.8 }])
  })

  it('forwards a catalog lora id so the backend can resolve its trigger', () => {
    const body = buildGenerateVideoBody({
      prompt: 'a fox',
      promptProvenance: 'typed',
      settings: {
        ...settings,
        loras: [
          { ref: 'cozy-felt.safetensors', scale: 0.8, name: 'Cozy Felt', catalogId: 'cozy-felt' },
        ],
      },
      imagePath: null,
      lastImagePath: null,
    })

    assert.deepEqual(body.loras, [
      { ref: 'cozy-felt.safetensors', scale: 0.8, catalogId: 'cozy-felt' },
    ])
  })

  it('omits the catalog id for a custom lora that has none', () => {
    const body = buildGenerateVideoBody({
      prompt: 'a fox',
      promptProvenance: 'typed',
      settings: {
        ...settings,
        loras: [{ ref: 'mine.safetensors', scale: 1, name: 'mine.safetensors' }],
      },
      imagePath: null,
      lastImagePath: null,
    })

    assert.deepEqual(body.loras, [{ ref: 'mine.safetensors', scale: 1 }])
    assert.equal('catalogId' in (body.loras as object[])[0], false)
  })
})
