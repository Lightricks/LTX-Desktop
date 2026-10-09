import path from 'path'
import { DEEP_LINK_SCHEME } from '../shared/deep-link.ts'

export function registerDeepLinkProtocolClient(
  setAsDefaultProtocolClient: (scheme: string, execPath?: string, args?: string[]) => boolean,
  options: { defaultApp: boolean; execPath: string; argv: readonly string[] },
): boolean {
  if (options.defaultApp) {
    const appPath = options.argv[1]
    if (appPath) {
      return setAsDefaultProtocolClient(
        DEEP_LINK_SCHEME,
        options.execPath,
        [path.resolve(appPath)],
      )
    }
  }
  return setAsDefaultProtocolClient(DEEP_LINK_SCHEME)
}
