import pkg from '../package.json' with { type: 'json' }

export const forkCredit =
  'Peter Beery created this branch of Lightricks’ LTX Desktop on behalf of SpaceSpeak.com.'

export function forkProductName(platform: string | undefined): string {
  return platform === 'darwin' ? 'LTX Desktop - SpaceSpeak' : pkg.productName
}

export function usesCommunityInstaller(platform: string): boolean {
  return platform === 'win32' || platform === 'darwin'
}
