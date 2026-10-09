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
  { flags: [] },
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
