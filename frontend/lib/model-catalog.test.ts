import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import {
  catalogModelDeleteConfirmMessage,
  ENCODING_MODEL_SEARCH_HAYSTACK,
  IMAGE_MODEL_SEARCH_HAYSTACK,
  matchesCatalogSearch,
  modelCatalogDownloadButtonLabel,
} from './model-catalog.ts'

describe('modelCatalogDownloadButtonLabel', () => {
  it('says Download when idle', () => {
    assert.equal(modelCatalogDownloadButtonLabel(false, 0), 'Download')
    assert.equal(modelCatalogDownloadButtonLabel(false, 42), 'Download')
  })

  it('shows only the percent while downloading so the spinner fits the action slot', () => {
    assert.equal(modelCatalogDownloadButtonLabel(true, 0), '0%')
    assert.equal(modelCatalogDownloadButtonLabel(true, 4.2), '4%')
    assert.equal(modelCatalogDownloadButtonLabel(true, 100), '100%')
  })
})

describe('catalogModelDeleteConfirmMessage', () => {
  it('names the model and says weights leave this computer', () => {
    assert.equal(
      catalogModelDeleteConfirmMessage('LTX 2.5'),
      'Delete "LTX 2.5"? This removes the downloaded weights from this computer.',
    )
  })
})

describe('matchesCatalogSearch', () => {
  it('matches encoding catalog copy', () => {
    assert.equal(matchesCatalogSearch(ENCODING_MODEL_SEARCH_HAYSTACK, ''), true)
    assert.equal(matchesCatalogSearch(ENCODING_MODEL_SEARCH_HAYSTACK, 'gemma'), true)
    assert.equal(matchesCatalogSearch(ENCODING_MODEL_SEARCH_HAYSTACK, 'encoder'), true)
    assert.equal(matchesCatalogSearch(ENCODING_MODEL_SEARCH_HAYSTACK, 'enhancer'), true)
    assert.equal(matchesCatalogSearch(ENCODING_MODEL_SEARCH_HAYSTACK, 'z image'), false)
  })

  it('matches the image catalog copy', () => {
    assert.equal(matchesCatalogSearch(IMAGE_MODEL_SEARCH_HAYSTACK, 'z image'), true)
    assert.equal(matchesCatalogSearch(IMAGE_MODEL_SEARCH_HAYSTACK, 'turbo'), true)
    assert.equal(matchesCatalogSearch(IMAGE_MODEL_SEARCH_HAYSTACK, 'gemma'), false)
  })
})
