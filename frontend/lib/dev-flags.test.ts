import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { ALL_DEV_FLAGS, UNLOADED_FLAGS } from './dev-flags.ts'

// The generated OpenAPI schema is the frontend's view of the backend models, so it
// is the source the registry is checked against (`openapi:check` keeps it current).
const { components } = JSON.parse(
  readFileSync(new URL('../generated/backend-openapi.json', import.meta.url), 'utf8'),
) as { components: { schemas: Record<string, { properties: Record<string, unknown> }> } }

const schemaKeys = (name: string) => Object.keys(components.schemas[name].properties).sort()

test('every backend flag is listed exactly once in the Dev Panel registry', () => {
  const listed = ALL_DEV_FLAGS.map(flag => flag.key).sort()
  assert.deepEqual(listed, schemaKeys('FeatureFlags'))
})

test('the patch model covers exactly the backend flags', () => {
  assert.deepEqual(schemaKeys('FeatureFlagsPatch'), schemaKeys('FeatureFlags'))
})

test('unloaded flags are all off', () => {
  assert.deepEqual(Object.keys(UNLOADED_FLAGS).sort(), schemaKeys('FeatureFlags'))
  assert.ok(Object.values(UNLOADED_FLAGS).every(value => value === false))
})
