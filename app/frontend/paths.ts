export const paths = {
  home: '/',
  textToVideo: '/text-to-video',
  imageToVideo: '/image-to-video',
  keyframes: '/keyframes',
  audioToVideo: '/audio-to-video',
  retake: '/retake',
  extend: '/extend',
  dayToNight: '/day-to-night',
  alphaGen: '/alpha-gen',
  deblur: '/deblur',
  colorization: '/colorization',
  cleanPlate: '/clean-plate',
  decompression: '/decompression',
  waterSimulation: '/water-simulation',
  layoutToRender: '/layout-to-render',
  restore: '/restore',
  ingredients: '/ingredients',
  refineDetails: '/refine-details',
  motionTrack: '/motion-track',
  videoGuide: '/video-guide',
  assets: '/assets',
  dashboard: '/activity-dashboard',
  assistant: '/assistant',
  assistantChat: '/assistant/:chatId',
  fetcherTool: '/fetcher/:tool',
  remote: '/remote',
  settings: '/settings',
  cozyFelt: '/cozy-felt',
  claymation: '/claymation',
  fantasyPainterly: '/fantasy-painterly',
  paperCutOutStyle: '/paper-cut-out-style',
  cinemagraph: '/cinemagraph',
  jibUp: '/jib-up',
  jibDown: '/jib-down',
  dollyIn: '/dolly-in',
  dollyOut: '/dolly-out',
  fpvMotion: '/fpv-motion',
  openwheelTCam: '/openwheel-t-cam',
  transition: '/transition',
  vbvr: '/vbvr',
  customLora: '/custom-lora',
  customLoraRun: '/custom-lora/:customLoraId',
  projects: '/projects',
  project: '/projects/:projectId',
} as const

/** The page of one saved custom LoRA. Its entry in the sidebar goes here. */
export function customLoraRunPath(customLoraId: string): string {
  return `${paths.customLora}/${encodeURIComponent(customLoraId)}`
}

export function assistantChatPath(chatId: string): string {
  return `${paths.assistant}/${encodeURIComponent(chatId)}`
}
