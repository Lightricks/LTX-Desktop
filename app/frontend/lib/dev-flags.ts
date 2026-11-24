import type { components } from '../generated/backend-openapi.ts'

// Dev / escape-hatch feature flags. The values live in the backend's
// feature_flags.json (shared by Desktop and the phone remote); this file only
// holds the display registry. Adding a flag = one field on the backend
// `FeatureFlags` and `FeatureFlagsPatch` models + one entry in DEV_FLAG_SECTIONS;
// dev-flags.test.ts fails if the registry and the generated API schema disagree.
// Read it from a single gate where the feature is used (avoid sprinkling the flag
// across call sites).
export type FeatureFlags = components['schemas']['FeatureFlags']
export type DevFlagKey = keyof FeatureFlags

interface DevFlagSpec {
  key: DevFlagKey
  label: string
  description: string
}

interface DevFlagSection {
  /** Untitled sections hold the new-surface flags. */
  title?: string
  flags: DevFlagSpec[]
}

export const DEV_FLAG_SECTIONS: DevFlagSection[] = [
  {
    flags: [
      {
        key: 'highResVideoToVideo',
        label: 'High-resolution video to video (1440p and 4K)',
        description: 'Add 1440p and 4K to the IC-LoRA recipes, and allow sources above 1080p in Retake and Extend. Text to Video, Image to Video and Audio to Video offer 1440p and 4K without this flag. Long clips run in overlapping time windows. This is slow and needs a large GPU. Each job still passes the hardware check. Reload the model list after you change it. Off by default.',
      },
      {
        key: 'createWithAi',
        label: 'Create with AI',
        description: 'Show "Create with AI" next to the start frame, end frame and reference image fields. It opens the image generator and puts the image in the field. Off by default.',
      },
      {
        key: 'showGenerationNegativePrompt',
        label: 'Show negative prompt on results',
        description: 'Show a "Negative prompt:" line under each result, in every feature that has a negative prompt. It shows the text the model got, with the NAG scale. It shows "(none)" when no negative prompt was sent, for example when the text is empty or NAG is off. Off by default.',
      },
      {
        key: 'keyframes',
        label: 'Keyframes (Home)',
        description: 'Show the Keyframes feature on Home. You place several images on the timeline of one video and set the strength of each. It needs a downloaded local LTX model. GenSpace multi-keyframe is not affected. Off by default.',
      },
    ],
  },
  {
    title: 'Legacy flags',
    flags: [
      {
        key: 'customIcLora',
        label: 'Custom IC-LoRA',
        description: 'Show the "Custom IC-LoRA" conditioning option (user-supplied weights + control video). Off by default — local results are currently low quality. Built-in Canny/Depth IC-LoRA is unaffected.',
      },
      {
        key: 'advancedIcLoraControls',
        label: 'Advanced IC-LoRA controls',
        description: 'Expose all IC-LoRA settings (skip stage 2, resolution factor, audio, strengths) for catalog and custom modes. Off by default — settings use the catalog IC-LoRA defaults.',
      },
    ],
  },
]

export const ALL_DEV_FLAGS: DevFlagSpec[] = DEV_FLAG_SECTIONS.flatMap(section => section.flags)

/**
 * What consumers see until the backend answers (or if it can't): every flag off.
 * This is "unknown means off", not a copy of the backend defaults, so it is built
 * from the registry; the test keeps the registry complete.
 */
export const UNLOADED_FLAGS = Object.fromEntries(
  ALL_DEV_FLAGS.map(flag => [flag.key, false]),
) as FeatureFlags
