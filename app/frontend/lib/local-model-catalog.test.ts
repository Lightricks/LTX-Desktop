import assert from 'node:assert/strict'
import { test } from 'node:test'
import { specFixture } from '../ltx-io/screens/Feature/definitions/specFixture.ts'
import { localModelChoices, matchesModelSearch } from './local-model-catalog.ts'

test('a native H3 host exposes all three H3 choices without advertising the Windows INT8 or Kandinsky workflows', () => {
  const specs = specFixture()
  const h3 = specs.downloaded_local_models![0]
  specs.h3_fl2va_local_model = h3
  specs.h3_fl2va_distill_local_model = h3
  specs.h3_ref2va_local_model = h3
  const choices = localModelChoices(specs)
  assert.deepEqual(choices, ['ltx-2.5', 'ltx-2.3', 'h3-fl2va', 'h3-fl2va-distill', 'h3-ref2va'])
  assert.deepEqual(choices.filter(id => matchesModelSearch(id, 'MiniMax Turbo')), ['h3-fl2va-distill'])
  assert.deepEqual(choices.filter(id => matchesModelSearch(id, 'H3 reference')), ['h3-ref2va'])
})
