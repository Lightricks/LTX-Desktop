import fs from 'fs/promises'
import path from 'path'

export type StatFsLike = {
  bavail: number | bigint
  bsize: number | bigint
}

export type FreeDiskBytesDeps = {
  statfs: (targetPath: string) => Promise<StatFsLike>
}

const defaultDeps: FreeDiskBytesDeps = {
  statfs: (targetPath) => fs.statfs(targetPath),
}

function isMissingPathError(err: unknown): boolean {
  return (
    typeof err === 'object' &&
    err !== null &&
    'code' in err &&
    (err as NodeJS.ErrnoException).code === 'ENOENT'
  )
}

/** Bytes free to this user on the volume that contains `targetPath`. Walks up if the folder is not created yet. */
export async function freeDiskBytes(
  targetPath: string,
  deps: FreeDiskBytesDeps = defaultDeps,
): Promise<number> {
  if (!path.isAbsolute(targetPath)) {
    throw new Error(`Path must be absolute: ${targetPath}`)
  }

  let dir = path.resolve(targetPath)
  for (;;) {
    try {
      const stats = await deps.statfs(dir)
      return Number(stats.bavail) * Number(stats.bsize)
    } catch (err) {
      const parent = path.dirname(dir)
      if (parent === dir || !isMissingPathError(err)) throw err
      dir = parent
    }
  }
}
