import pkg from '../package.json' with { type: 'json' }

export const forkCredit =
  'Original LTX Desktop by Lightricks. Community fork by Peter Beery / SpaceSpeak.com.'

export function forkProductName(platform: string | undefined): string {
  return platform === 'darwin' || platform === 'linux' ? 'LTX Desktop - SpaceSpeak' : pkg.productName
}

export function usesCommunityInstaller(platform: string): boolean {
  return platform === 'win32' || platform === 'darwin' || platform === 'linux'
}
