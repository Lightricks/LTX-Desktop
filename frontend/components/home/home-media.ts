import ltx25AvatarReactionPosterUrl from '@/assets/home/ltx-2.5/ltx-2.5-3d-avatar-reaction-poster.webp'
import ltx25AvatarReactionVideoUrl from '@/assets/home/ltx-2.5/ltx-2.5-3d-avatar-reaction.mp4'
import ltx25MultishotPosterUrl from '@/assets/home/ltx-2.5/ltx-2.5-multishot-red-cube-poster.webp'
import ltx25MultishotVideoUrl from '@/assets/home/ltx-2.5/ltx-2.5-multishot-red-cube.mp4'
import ltx25NorthernGlowPosterUrl from '@/assets/home/ltx-2.5/ltx-2.5-northern-glow-poster.webp'
import ltx25NorthernGlowVideoUrl from '@/assets/home/ltx-2.5/ltx-2.5-northern-glow.mp4'
import ltx25ShowreelPosterUrl from '@/assets/home/ltx-2.5/ltx-2.5-showreel-poster.webp'
import ltx25ShowreelVideoUrl from '@/assets/home/ltx-2.5/ltx-2.5-showreel.mp4'
import audioToVideoDiscoverPosterUrl from '@/assets/home/audio-to-video/discover-poster.webp'
import audioToVideoDiscoverPreviewUrl from '@/assets/home/audio-to-video/discover-preview.mp4'
import claymationPosterUrl from '@/assets/home/claymation/poster.webp'
import claymationPreviewUrl from '@/assets/home/claymation/preview.webm'
import cinemagraphDiscoverPosterUrl from '@/assets/home/cinemagraph/discover-poster.webp'
import cinemagraphDiscoverPreviewUrl from '@/assets/home/cinemagraph/discover-preview.mp4'
import cozyFeltDiscoverPosterUrl from '@/assets/home/cozy-felt/discover-poster.webp'
import cozyFeltDiscoverPreviewUrl from '@/assets/home/cozy-felt/discover-preview.mp4'
import dollyInPosterUrl from '@/assets/home/dolly-in/poster.webp'
import dollyInPreviewUrl from '@/assets/home/dolly-in/preview.webm'
import dollyOutPosterUrl from '@/assets/home/dolly-out/poster.webp'
import dollyOutPreviewUrl from '@/assets/home/dolly-out/preview.webm'
import extendDiscoverPosterUrl from '@/assets/home/extend/discover-poster.webp'
import extendDiscoverPreviewUrl from '@/assets/home/extend/discover-preview.mp4'
import alphaGenDiscoverPosterUrl from '@/assets/home/alpha-gen/discover-poster.webp'
import alphaGenDiscoverPreviewUrl from '@/assets/home/alpha-gen/discover-preview.mp4'
import deblurDiscoverPosterUrl from '@/assets/home/deblur/discover-poster.webp'
import deblurDiscoverPreviewUrl from '@/assets/home/deblur/discover-preview.mp4'
import colorizationDiscoverPosterUrl from '@/assets/home/colorization/discover-poster.webp'
import colorizationDiscoverPreviewUrl from '@/assets/home/colorization/discover-preview.webm'
import cleanPlateDiscoverPosterUrl from '@/assets/home/clean-plate/discover-poster.webp'
import cleanPlateDiscoverPreviewUrl from '@/assets/home/clean-plate/discover-preview.webm'
import decompressionDiscoverPosterUrl from '@/assets/home/decompression/discover-poster.webp'
import decompressionDiscoverPreviewUrl from '@/assets/home/decompression/discover-preview.mp4'
import waterSimulationDiscoverPosterUrl from '@/assets/home/water-simulation/discover-poster.webp'
import waterSimulationDiscoverPreviewUrl from '@/assets/home/water-simulation/discover-preview.mp4'
import restoreDiscoverPosterUrl from '@/assets/home/restore/discover-poster.webp'
import restoreDiscoverPreviewUrl from '@/assets/home/restore/discover-preview.mp4'
import dayToNightDiscoverPosterUrl from '@/assets/home/day-to-night/discover-poster.webp'
import dayToNightDiscoverPreviewUrl from '@/assets/home/day-to-night/discover-preview.mp4'
import layoutToRenderExamplePosterUrl from '@/assets/home/layout-to-render/example-poster.webp'
import layoutToRenderExampleVideoUrl from '@/assets/home/layout-to-render/example.mp4'
import fantasyPainterlyPosterUrl from '@/assets/home/fantasy-painterly/poster.webp'
import fantasyPainterlyPreviewUrl from '@/assets/home/fantasy-painterly/preview.webm'
import fpvMotionDiscoverPosterUrl from '@/assets/home/fpv-motion/discover-poster.webp'
import fpvMotionDiscoverPreviewUrl from '@/assets/home/fpv-motion/discover-preview.mp4'
import imageToVideoDiscoverPosterUrl from '@/assets/home/image-to-video/discover-poster.webp'
import imageToVideoDiscoverPreviewUrl from '@/assets/home/image-to-video/discover-preview.mp4'
import jibDownPosterUrl from '@/assets/home/jib-down/poster.webp'
import jibDownPreviewUrl from '@/assets/home/jib-down/preview.webm'
import jibUpPosterUrl from '@/assets/home/jib-up/poster.webp'
import jibUpPreviewUrl from '@/assets/home/jib-up/preview.webm'
import openwheelTCamPosterUrl from '@/assets/home/openwheel-t-cam/poster.webp'
import openwheelTCamPreviewUrl from '@/assets/home/openwheel-t-cam/preview.webm'
import paperCutOutPosterUrl from '@/assets/home/paper-cut-out-style/poster.webp'
import paperCutOutPreviewUrl from '@/assets/home/paper-cut-out-style/preview.webm'
import retakeDiscoverPosterUrl from '@/assets/home/retake/discover-poster.webp'
import retakeDiscoverPreviewUrl from '@/assets/home/retake/discover-preview.mp4'
import textToVideoDiscoverPosterUrl from '@/assets/home/text-to-video/discover-poster.webp'
import textToVideoDiscoverPreviewUrl from '@/assets/home/text-to-video/discover-preview.webm'
import transitionDiscoverPosterUrl from '@/assets/home/transition/discover-poster.webp'
import transitionDiscoverPreviewUrl from '@/assets/home/transition/discover-preview.webm'
import vbvrDiscoverPosterUrl from '@/assets/home/vbvr/discover-poster.webp'
import vbvrDiscoverPreviewUrl from '@/assets/home/vbvr/discover-preview.webm'
import type { HomeFeatureId } from '../../lib/home-features.ts'
import type { LoraRecipeId } from '../../lib/lora-recipes.ts'
import type { PeakCinematicExample } from './peakCinematic'

export type HomeFeatureMedia = {
  readonly posterUrl: string
  readonly previewUrl?: string
}

/** Local LTX-2.5 hero playlist. Kept out of HOME_FEATURE_MEDIA. */
export const PEAK_CINEMATIC_EXAMPLES: readonly PeakCinematicExample[] = [
  {
    id: 'ltx-2.5-showreel',
    videoUrl: ltx25ShowreelVideoUrl,
    posterUrl: ltx25ShowreelPosterUrl,
  },
  {
    id: 'ltx-2.5-multishot',
    videoUrl: ltx25MultishotVideoUrl,
    posterUrl: ltx25MultishotPosterUrl,
  },
  {
    id: 'ltx-2.5-3d-avatar-reaction',
    videoUrl: ltx25AvatarReactionVideoUrl,
    posterUrl: ltx25AvatarReactionPosterUrl,
  },
  {
    id: 'ltx-2.5-northern-glow',
    videoUrl: ltx25NorthernGlowVideoUrl,
    posterUrl: ltx25NorthernGlowPosterUrl,
  },
]

export const HOME_FEATURE_MEDIA: Record<HomeFeatureId, HomeFeatureMedia> = {
  'text-to-video': {
    previewUrl: textToVideoDiscoverPreviewUrl,
    posterUrl: textToVideoDiscoverPosterUrl,
  },
  'image-to-video': {
    previewUrl: imageToVideoDiscoverPreviewUrl,
    posterUrl: imageToVideoDiscoverPosterUrl,
  },
  'audio-to-video': {
    previewUrl: audioToVideoDiscoverPreviewUrl,
    posterUrl: audioToVideoDiscoverPosterUrl,
  },
  retake: {
    previewUrl: retakeDiscoverPreviewUrl,
    posterUrl: retakeDiscoverPosterUrl,
  },
  extend: {
    previewUrl: extendDiscoverPreviewUrl,
    posterUrl: extendDiscoverPosterUrl,
  },
  'day-to-night': {
    previewUrl: dayToNightDiscoverPreviewUrl,
    posterUrl: dayToNightDiscoverPosterUrl,
  },
  'alpha-gen': {
    previewUrl: alphaGenDiscoverPreviewUrl,
    posterUrl: alphaGenDiscoverPosterUrl,
  },
  'deblur': {
    previewUrl: deblurDiscoverPreviewUrl,
    posterUrl: deblurDiscoverPosterUrl,
  },
  'colorization': {
    previewUrl: colorizationDiscoverPreviewUrl,
    posterUrl: colorizationDiscoverPosterUrl,
  },
  'clean-plate': {
    previewUrl: cleanPlateDiscoverPreviewUrl,
    posterUrl: cleanPlateDiscoverPosterUrl,
  },
  'decompression': {
    previewUrl: decompressionDiscoverPreviewUrl,
    posterUrl: decompressionDiscoverPosterUrl,
  },
  'water-simulation': {
    previewUrl: waterSimulationDiscoverPreviewUrl,
    posterUrl: waterSimulationDiscoverPosterUrl,
  },
  'layout-to-render': {
    previewUrl: layoutToRenderExampleVideoUrl,
    posterUrl: layoutToRenderExamplePosterUrl,
  },
  'restore': {
    previewUrl: restoreDiscoverPreviewUrl,
    posterUrl: restoreDiscoverPosterUrl,
  },
  ...({
    'cozy-felt': {
      previewUrl: cozyFeltDiscoverPreviewUrl,
      posterUrl: cozyFeltDiscoverPosterUrl,
    },
    claymation: {
      previewUrl: claymationPreviewUrl,
      posterUrl: claymationPosterUrl,
    },
    'fantasy-painterly': {
      previewUrl: fantasyPainterlyPreviewUrl,
      posterUrl: fantasyPainterlyPosterUrl,
    },
    'paper-cut-out-style': {
      previewUrl: paperCutOutPreviewUrl,
      posterUrl: paperCutOutPosterUrl,
    },
    cinemagraph: {
      previewUrl: cinemagraphDiscoverPreviewUrl,
      posterUrl: cinemagraphDiscoverPosterUrl,
    },
    'jib-up': {
      previewUrl: jibUpPreviewUrl,
      posterUrl: jibUpPosterUrl,
    },
    'jib-down': {
      previewUrl: jibDownPreviewUrl,
      posterUrl: jibDownPosterUrl,
    },
    'dolly-in': {
      previewUrl: dollyInPreviewUrl,
      posterUrl: dollyInPosterUrl,
    },
    'dolly-out': {
      previewUrl: dollyOutPreviewUrl,
      posterUrl: dollyOutPosterUrl,
    },
    'fpv-motion': {
      previewUrl: fpvMotionDiscoverPreviewUrl,
      posterUrl: fpvMotionDiscoverPosterUrl,
    },
    'openwheel-t-cam': {
      previewUrl: openwheelTCamPreviewUrl,
      posterUrl: openwheelTCamPosterUrl,
    },
    transition: {
      previewUrl: transitionDiscoverPreviewUrl,
      posterUrl: transitionDiscoverPosterUrl,
    },
    vbvr: {
      previewUrl: vbvrDiscoverPreviewUrl,
      posterUrl: vbvrDiscoverPosterUrl,
    },
  } satisfies Record<LoraRecipeId, HomeFeatureMedia>),
}
