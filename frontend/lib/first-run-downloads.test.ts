import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  buildFirstRunDownloadSteps,
  createDownloadProgressPollGuard,
  defaultSelectedFirstRunLoraKeys,
  defaultSelectedRecommendedQualityCpIds,
  describeCheckpointsErrorMessage,
  firstRunCatalogCpIds,
  firstRunLoraFromCatalog,
  firstRunLoraKey,
  isFirstRunImageModel,
  isFirstRunModelPackItem,
  isFirstRunTextEncoder,
  isRequiredLtxBundleReady,
  mapLoraProgressToDownloadProgress,
  recommendationsErrorMessage,
  recommendedQualityCpIds,
  resolveDownloadFailure,
} from './first-run-downloads.ts'

const REQUIRED_CPS = ['ltx-2.5-22b-distilled', 'ltx-2.5-spatial-upscaler-x2-1.0'] as const
const OPTIONAL_CPS = ['ltx-2.5-video-vae'] as const
const QUALITY_CPS = ['gemma-4-e2b-it'] as const
const IMAGE_CP = 'z-image-turbo'

const downloadRecommendation = {
  status: 'download' as const,
  cps_to_download: [...REQUIRED_CPS],
  optional_cp_ids: [...OPTIONAL_CPS],
  recommended_quality_cp_ids: [...QUALITY_CPS],
}

describe('first-run download progress polling', () => {
  it('allows one in-flight poll and consumes one terminal transition per session', () => {
    const guard = createDownloadProgressPollGuard()

    assert.equal(guard.begin('session-1'), true)
    assert.equal(guard.begin('session-1'), false)

    guard.markTerminal('session-1')
    guard.end()

    assert.equal(guard.begin('session-1'), false)
    assert.equal(guard.begin('session-2'), true)
  })
})

describe('required first-run bundle for the missing-models gate', () => {
  it('clears when the LTX pack is present, even if Z Image Turbo is still recommended', () => {
    assert.equal(
      isRequiredLtxBundleReady({ status: 'ok' }, { cp_to_download: IMAGE_CP }),
      true,
    )
    assert.equal(
      isRequiredLtxBundleReady({ status: 'upgrade' }, { cp_to_download: IMAGE_CP }),
      true,
    )
  })

  it('still blocks when the LTX pack is missing, even if Z Image Turbo is already on disk', () => {
    assert.equal(
      isRequiredLtxBundleReady({ status: 'download' }, { cp_to_download: null }),
      false,
    )
  })
})

describe('first-run recommended quality downloads', () => {
  it('exposes recommended quality checkpoints separately from required and optional ids', () => {
    assert.deepEqual(recommendedQualityCpIds(downloadRecommendation), [...QUALITY_CPS])
    assert.equal(REQUIRED_CPS.includes('gemma-4-e2b-it' as (typeof REQUIRED_CPS)[number]), false)
    assert.equal(OPTIONAL_CPS.includes('gemma-4-e2b-it' as (typeof OPTIONAL_CPS)[number]), false)
  })

  it('selects recommended quality checkpoints by default', () => {
    assert.deepEqual(
      defaultSelectedRecommendedQualityCpIds(downloadRecommendation),
      [...QUALITY_CPS],
    )
  })

  it('queues selected recommended quality after the blocking bundle as a non-blocking step', () => {
    const steps = buildFirstRunDownloadSteps({
      ltxRecommendation: downloadRecommendation,
      imgGenCpToDownload: IMAGE_CP,
      selectedRecommendedQualityCpIds: [...QUALITY_CPS],
      optedInOptionalCpIds: [],
    })

    assert.deepEqual(steps, [
      {
        type: 'download',
        cpIds: [...REQUIRED_CPS, IMAGE_CP],
        failurePolicy: 'blocking',
      },
      {
        type: 'download',
        cpIds: [...QUALITY_CPS],
        failurePolicy: 'nonblocking',
      },
    ])
  })

  it('omits recommended quality when the user skips it', () => {
    const steps = buildFirstRunDownloadSteps({
      ltxRecommendation: downloadRecommendation,
      imgGenCpToDownload: null,
      selectedRecommendedQualityCpIds: [],
      optedInOptionalCpIds: [...OPTIONAL_CPS],
    })

    assert.deepEqual(steps, [
      {
        type: 'download',
        cpIds: [...REQUIRED_CPS, ...OPTIONAL_CPS],
        failurePolicy: 'blocking',
      },
    ])
  })

  it('continues setup when a recommended quality download fails', () => {
    const failed = {
      type: 'download' as const,
      cpIds: [...QUALITY_CPS],
      failurePolicy: 'nonblocking' as const,
    }
    const remaining = [
      {
        type: 'download' as const,
        cpIds: [...OPTIONAL_CPS],
        failurePolicy: 'blocking' as const,
      },
    ]

    assert.deepEqual(resolveDownloadFailure(failed, remaining), {
      action: 'continue',
      next: remaining[0],
    })
    assert.deepEqual(resolveDownloadFailure(failed, []), { action: 'complete' })
  })

  it('still treats required download failures as blocking install errors', () => {
    const failed = {
      type: 'download' as const,
      cpIds: [...REQUIRED_CPS],
      failurePolicy: 'blocking' as const,
    }

    assert.deepEqual(resolveDownloadFailure(failed, []), { action: 'error' })
  })

  it('drops excluded checkpoints from the blocking bundle', () => {
    const steps = buildFirstRunDownloadSteps({
      ltxRecommendation: {
        ...downloadRecommendation,
        cps_to_download: [...REQUIRED_CPS, 'gemma4-12b-with-proj-ltx-2.5'],
      },
      imgGenCpToDownload: IMAGE_CP,
      selectedRecommendedQualityCpIds: [...QUALITY_CPS],
      optedInOptionalCpIds: [],
      excludedCpIds: ['gemma4-12b-with-proj-ltx-2.5'],
    })

    assert.deepEqual(steps[0], {
      type: 'download',
      cpIds: [...REQUIRED_CPS, IMAGE_CP],
      failurePolicy: 'blocking',
    })
  })
})

describe('first-run checkpoint catalog', () => {
  it('lists the full bundle, including optional and recommended ids the install plan does not queue by default', () => {
    const catalog = firstRunCatalogCpIds({
      ltxRecommendation: downloadRecommendation,
      imgGenCpToDownload: IMAGE_CP,
    })
    const planIds = buildFirstRunDownloadSteps({
      ltxRecommendation: downloadRecommendation,
      imgGenCpToDownload: IMAGE_CP,
      selectedRecommendedQualityCpIds: [],
      optedInOptionalCpIds: [],
    }).flatMap((step) => step.cpIds)

    assert.deepEqual(planIds, [...REQUIRED_CPS, IMAGE_CP])
    assert.deepEqual(catalog, [...REQUIRED_CPS, ...OPTIONAL_CPS, ...QUALITY_CPS, IMAGE_CP])
  })

  it('keeps ids the live plan dropped because they are already installed', () => {
    const installedCp = REQUIRED_CPS[0]
    const livePlan = {
      ...downloadRecommendation,
      cps_to_download: REQUIRED_CPS.filter((cpId) => cpId !== installedCp),
    }
    const planIds = buildFirstRunDownloadSteps({
      ltxRecommendation: livePlan,
      imgGenCpToDownload: null,
      selectedRecommendedQualityCpIds: [],
      optedInOptionalCpIds: [],
    }).flatMap((step) => step.cpIds)
    const catalog = firstRunCatalogCpIds({
      ltxRecommendation: downloadRecommendation,
      imgGenCpToDownload: IMAGE_CP,
    })

    assert.equal(planIds.includes(installedCp), false)
    assert.equal(catalog.includes(installedCp), true)
    for (const cpId of planIds) {
      assert.equal(catalog.includes(cpId), true)
    }
  })
})

describe('first-run setup grouping', () => {
  it('keeps LTX pack roles out of encoder and image rows', () => {
    assert.equal(isFirstRunModelPackItem({ role: 'base' }), true)
    assert.equal(isFirstRunModelPackItem({ role: 'vae' }), true)
    assert.equal(isFirstRunModelPackItem({ role: 'text_encoder' }), false)
    assert.equal(isFirstRunTextEncoder({ role: 'text_encoder' }), true)
    assert.equal(isFirstRunImageModel({ role: 'image' }), true)
  })

  it('preselects LoRAs that are not already on disk', () => {
    const items = [
      firstRunLoraFromCatalog(
        'lora',
        { id: 'fpv-motion', name: 'FPV Motion', download: { variants: [{ size_bytes: 10 }] } },
        false,
      ),
      firstRunLoraFromCatalog(
        'ic-lora',
        { id: 'day-to-night', name: 'Day to Night', download: { variants: [{ size_bytes: 20 }] } },
        true,
      ),
    ]
    assert.deepEqual(defaultSelectedFirstRunLoraKeys(items), ['lora:fpv-motion'])
  })

  it('qualifies LoRA selection keys by kind so catalog and IC LoRAs cannot collide', () => {
    const items = [
      firstRunLoraFromCatalog(
        'lora',
        { id: 'shared-id', name: 'Catalog One', download: { variants: [{ size_bytes: 10 }] } },
        false,
      ),
      firstRunLoraFromCatalog(
        'ic-lora',
        { id: 'shared-id', name: 'IC One', download: { variants: [{ size_bytes: 20 }] } },
        false,
      ),
    ]
    assert.equal(firstRunLoraKey(items[0]!), 'lora:shared-id')
    assert.equal(firstRunLoraKey(items[1]!), 'ic-lora:shared-id')
    assert.deepEqual(defaultSelectedFirstRunLoraKeys(items), ['lora:shared-id', 'ic-lora:shared-id'])
  })
})

describe('first-run recommendations errors', () => {
  it('reports which fetch phase failed', () => {
    assert.equal(recommendationsErrorMessage('boom'), 'Failed to fetch model recommendations: boom')
    assert.equal(describeCheckpointsErrorMessage('boom'), 'Failed to describe checkpoints: boom')
  })
})

describe('first-run LoRA progress mapping', () => {
  const item = { name: 'FPV Motion', sizeBytes: 100 }

  it('maps downloading progress onto the model progress shape', () => {
    assert.deepEqual(
      mapLoraProgressToDownloadProgress(
        {
          status: 'downloading',
          progress: 42,
          expected_bytes: 100,
          speed_bytes_per_sec: 10,
          downloaded_bytes: 42,
          error: null,
        },
        item,
      ),
      {
        status: 'downloading',
        all_files: ['FPV Motion'],
        completed_files: [],
        current_downloading_file: 'FPV Motion',
        current_file_progress: 42,
        expected_total_bytes: 100,
        speed_bytes_per_sec: 10,
        total_downloaded_bytes: 42,
        total_progress: 42,
        error: null,
      },
    )
  })

  it('marks the file complete and falls back to the catalog size when expected bytes are unknown', () => {
    const mapped = mapLoraProgressToDownloadProgress(
      {
        status: 'complete',
        progress: 100,
        expected_bytes: 0,
        speed_bytes_per_sec: 0,
        downloaded_bytes: 100,
      },
      item,
    )
    assert.equal(mapped.status, 'complete')
    assert.deepEqual(mapped.completed_files, ['FPV Motion'])
    assert.equal(mapped.expected_total_bytes, 100)
    assert.equal(mapped.error, null)
  })

  it('carries LoRA errors through and treats unknown states as downloading', () => {
    const failed = mapLoraProgressToDownloadProgress(
      {
        status: 'error',
        progress: 10,
        expected_bytes: 100,
        speed_bytes_per_sec: 0,
        downloaded_bytes: 10,
        error: 'boom',
      },
      item,
    )
    assert.equal(failed.status, 'error')
    assert.equal(failed.error, 'boom')

    const unknown = mapLoraProgressToDownloadProgress(
      {
        status: 'queued',
        progress: 0,
        expected_bytes: 100,
        speed_bytes_per_sec: 0,
        downloaded_bytes: 0,
        error: null,
      },
      item,
    )
    assert.equal(unknown.status, 'downloading')
  })
})
