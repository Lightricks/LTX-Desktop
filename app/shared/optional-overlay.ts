import fs from 'node:fs'
import path from 'node:path'

export type ViteCommand = 'serve' | 'build'

export type OverlayIo = {
  existsSync: (p: string) => boolean
  readdirSync: (p: string) => string[]
}

const defaultIo: OverlayIo = {
  existsSync: (p) => fs.existsSync(p),
  readdirSync: (p) => fs.readdirSync(p),
}

/** First `internal/*` folder that implements the three @optional entry files. */
export function findOptionalOverlayDir(root: string, io: OverlayIo = defaultIo): string | null {
  const internal = path.resolve(root, 'internal')
  if (!io.existsSync(internal)) return null
  let names: string[]
  try {
    names = io.readdirSync(internal)
  } catch {
    return null
  }
  for (const name of names) {
    if (name.startsWith('.')) continue
    const dir = path.join(internal, name)
    if (
      io.existsSync(path.join(dir, 'wrap-app.tsx')) &&
      io.existsSync(path.join(dir, 'main', 'register-handlers.ts')) &&
      io.existsSync(path.join(dir, 'main', 'attach-preload.ts'))
    ) {
      return dir
    }
  }
  return null
}

export function shouldBindOptionalOverlay(
  command: ViteCommand,
  overlayDir: string | null,
): overlayDir is string {
  return command === 'serve' && overlayDir !== null
}

export function optionalOverlayAliases(opts: {
  root: string
  command: ViteCommand
  overlayDir?: string | null
}): { find: string | RegExp; replacement: string }[] {
  const overlayDir =
    'overlayDir' in opts ? (opts.overlayDir ?? null) : findOptionalOverlayDir(opts.root)
  const bind = shouldBindOptionalOverlay(opts.command, overlayDir)
  return [
    {
      find: /^@optional\/app-wrap$/,
      replacement: bind
        ? path.resolve(overlayDir, 'wrap-app.tsx')
        : path.resolve(opts.root, 'frontend/lib/optional-app-wrap.tsx'),
    },
    {
      find: /^@optional\/app-handlers$/,
      replacement: bind
        ? path.resolve(overlayDir, 'main', 'register-handlers.ts')
        : path.resolve(opts.root, 'electron/optional-app-handlers.ts'),
    },
    {
      find: /^@optional\/preload$/,
      replacement: bind
        ? path.resolve(overlayDir, 'main', 'attach-preload.ts')
        : path.resolve(opts.root, 'electron/optional-preload.ts'),
    },
  ]
}
