import pkg from '../package.json' with { type: 'json' }

export const forkCredit =
  'Original LTX Desktop by Lightricks. Community fork by Peter Beery / SpaceSpeak.com.'
export const forkGitHubUrl = 'https://github.com/pbeery1live/LTX-Desktop/tree/spacespeak-gpu'
export const forkUpdateGuidance = 'Look for updates to this edition under Releases in the SpaceSpeak GitHub repository.'

export function forkProductName(platform: string | undefined): string {
  return platform === 'darwin' || platform === 'linux' ? 'LTX Desktop - SpaceSpeak' : pkg.productName
}

export function usesCommunityInstaller(platform: string): boolean {
  return platform === 'win32' || platform === 'darwin' || platform === 'linux'
}
