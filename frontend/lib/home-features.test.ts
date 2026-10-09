import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  FIRST_CLASS_HOME_FEATURES,
  getHomeFeature,
  HOME_FEATURE_SECTIONS,
  HOME_FEATURES,
  isHomeFeatureEnabled,
  isHomeFeatureId,
  matchHomeFeature,
} from './home-features.ts'
import { IC_LORA_RECIPES } from './ic-lora-recipes.ts'
import { LORA_RECIPE_HOME_FEATURES } from './lora-recipes.ts'
import { paths } from '../paths.ts'

describe('home feature registry', () => {
  it('defines the initial Home feature contract', () => {
    assert.equal(isHomeFeatureId('text-to-video'), true)
    assert.equal(isHomeFeatureId('image-to-video'), true)
    assert.equal(isHomeFeatureId('audio-to-video'), true)
    assert.equal(isHomeFeatureId('retake'), true)
    assert.equal(isHomeFeatureId('extend'), true)
    assert.equal(isHomeFeatureId('day-to-night'), true)
    assert.equal(isHomeFeatureId('projects'), false)
    assert.equal(isHomeFeatureId('assets'), false)
    assert.equal(getHomeFeature('text-to-video').path, paths.textToVideo)
    assert.equal(getHomeFeature('image-to-video').path, paths.imageToVideo)
    assert.equal(getHomeFeature('audio-to-video').path, paths.audioToVideo)
    assert.equal(getHomeFeature('retake').path, paths.retake)
    assert.equal(getHomeFeature('extend').path, paths.extend)
    assert.equal(getHomeFeature('day-to-night').path, paths.dayToNight)
    assert.deepEqual(getHomeFeature('image-to-video'), {
      id: 'image-to-video',
      path: paths.imageToVideo,
      title: 'Image to Video',
      description: 'Animate a still image into a short clip.',
      fetcherTool: 'i2v',
      listing: {
        blurb:
          'Bring a still image to life by describing how it should move. Provide a single frame or both a start and end frame, and receive smooth, natural animation that stays true to your original composition.',
        categories: 'Motion',
        typeLabel: 'Image to Video',
      },
    })
    assert.deepEqual(getHomeFeature('text-to-video'), {
      id: 'text-to-video',
      path: paths.textToVideo,
      title: 'Text to Video',
      description: 'Generate a video from a text prompt with matching audio.',
      fetcherTool: 't2v',
      listing: {
        blurb:
          'Generate a video from a text prompt. Receive a high-quality video with synchronized audio, refined motion, and rich detail, optimized for portrait or landscape formats.',
        categories: 'Workflow',
        typeLabel: 'Text to Video',
      },
    })
    assert.deepEqual(getHomeFeature('audio-to-video'), {
      id: 'audio-to-video',
      path: paths.audioToVideo,
      title: 'Audio to Video',
      description: 'Generate a video synchronized to an audio track.',
      fetcherTool: 'a2v',
      listing: {
        blurb:
          'Turn any audio track into video. Provide music, dialogue, or sound effects, and the visuals you describe will be generated in perfect sync with the rhythm, timing, and energy of your audio.',
        categories: 'Audio, Animation',
        typeLabel: 'Audio to Video',
      },
    })
    assert.deepEqual(getHomeFeature('retake'), {
      id: 'retake',
      path: paths.retake,
      title: 'Retake',
      description: 'Replace a region of an existing clip.',
      fetcherTool: 'retake',
      listing: {
        blurb:
          'Refine a video without starting over. Select the segment that needs work, describe the change, and receive a regenerated version that blends seamlessly into the surrounding footage.',
        categories: 'Post production',
        typeLabel: 'Retake video',
      },
    })
    assert.deepEqual(getHomeFeature('extend'), {
      id: 'extend',
      path: paths.extend,
      title: 'Extend',
      description: 'Grow an existing clip forward or backward.',
      fetcherTool: 'extend',
      listing: {
        blurb:
          'Continue a video beyond its original length. Feed in a short clip and receive a natural extension that preserves motion and scene continuity, chainable up to 60 seconds.',
        categories: 'Post production',
        typeLabel: 'Extend Video',
      },
    })
    assert.deepEqual(getHomeFeature('day-to-night'), {
      id: 'day-to-night',
      path: paths.dayToNight,
      title: 'Day to Night',
      description: 'Relight a clip from day into night.',
      fetcherTool: 'day-to-night',
      listing: {
        blurb:
          'Turn a daytime clip into night. Keep the motion and the framing, and describe the light you want.',
        categories: 'Post production',
        typeLabel: 'Day to Night',
        huggingfaceUrl:
          'https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Day-To-Night',
      },
    })
    const recipeIds = LORA_RECIPE_HOME_FEATURES.map((recipe) => recipe.id)
    const icLoraIds = IC_LORA_RECIPES.map((recipe) => recipe.id)
    assert.deepEqual(
      HOME_FEATURES.map((feature) => feature.id),
      ['text-to-video', 'image-to-video', 'audio-to-video', 'retake', 'extend', ...icLoraIds, ...recipeIds],
    )
    assert.ok(isHomeFeatureId('cozy-felt'))
    assert.ok(isHomeFeatureId('dolly-in'))
    const fetcherTools = HOME_FEATURES.flatMap((feature) =>
      'fetcherTool' in feature && feature.fetcherTool ? [feature.fetcherTool] : [],
    )
    assert.deepEqual(fetcherTools, ['t2v', 'i2v', 'a2v', 'retake', 'extend', ...icLoraIds, ...recipeIds])
    assert.equal(new Set(fetcherTools).size, fetcherTools.length)
    assert.equal(matchHomeFeature(paths.textToVideo)?.id, 'text-to-video')
    assert.equal(matchHomeFeature(paths.home), undefined)
    for (const feature of HOME_FEATURES) {
      assert.equal(matchHomeFeature(feature.path)?.id, feature.id)
    }
  })
})

describe('home feature sections', () => {
  it('puts every enabled first-class feature in exactly one section', () => {
    const sectioned = HOME_FEATURE_SECTIONS.flatMap((section) =>
      section.features.map((feature) => feature.id),
    )
    assert.equal(new Set(sectioned).size, sectioned.length)
    assert.deepEqual(
      [...sectioned].sort(),
      FIRST_CLASS_HOME_FEATURES.map((feature) => feature.id)
        .filter(isHomeFeatureEnabled)
        .sort(),
    )
  })

  it('groups IC-LoRA recipes by their section, with VFX before Post production', () => {
    assert.deepEqual(
      HOME_FEATURE_SECTIONS.map((section) => [
        section.heading,
        section.features.map((feature) => feature.id),
      ]),
      [
        ['Workflows', ['text-to-video', 'image-to-video', 'audio-to-video']],
        ['VFX', ['layout-to-render', 'restore']],
        [
          'Post production',
          [
            'retake',
            'extend',
            'day-to-night',
            'deblur',
            'colorization',
            'clean-plate',
            'decompression',
            'water-simulation',
          ],
        ],
      ],
    )
  })
})
