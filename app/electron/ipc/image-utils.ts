import { spawnSync } from 'child_process'
import fs from 'fs'
import path from 'path'
import { getPythonPath } from '../python-backend'

const DEFAULT_THUMBNAIL_MAX_DIMENSION = 400

export function getThumbnailPaths(assetPath: string): { bigThumbnailPath: string; smallThumbnailPath: string } {
  const parsed = path.parse(assetPath)
  return {
    bigThumbnailPath: path.join(parsed.dir, `${parsed.name}_big_thumbnail.png`),
    smallThumbnailPath: path.join(parsed.dir, `${parsed.name}_small_thumbnail.png`),
  }
}

export function createDownsampledThumbnail(
  sourcePath: string,
  outputPath: string,
  maxDimension = DEFAULT_THUMBNAIL_MAX_DIMENSION,
): void {
  const pythonPath = getPythonPath()
  const script = [
    'from PIL import Image, ImageOps',
    'import sys',
    '',
    'src = sys.argv[1]',
    'dst = sys.argv[2]',
    'max_dim = int(sys.argv[3])',
    '',
    'with Image.open(src) as img:',
    '    img = ImageOps.exif_transpose(img)',
    '    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)',
    '    if img.mode not in ("RGB", "RGBA"):',
    '        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")',
    '    img.save(dst, format="PNG")',
  ].join('\n')
  const result = spawnSync(
    pythonPath,
    ['-c', script, sourcePath, outputPath, String(maxDimension)],
    { timeout: 15000 },
  )
  if (result.status !== 0) {
    const stderr = result.stderr?.toString().trim() || ''
    throw new Error(`Pillow resize failed (code ${result.status}): ${stderr}`)
  }
  if (!fs.existsSync(outputPath)) {
    throw new Error(`Failed to create small thumbnail: ${outputPath}`)
  }
}

export function getImageDimensions(sourcePath: string): { width: number; height: number } {
  const pythonPath = getPythonPath()
  const script = [
    'from PIL import Image, ImageOps',
    'import sys',
    '',
    'src = sys.argv[1]',
    '',
    'with Image.open(src) as img:',
    '    img = ImageOps.exif_transpose(img)',
    '    print(f"{img.width},{img.height}")',
  ].join('\n')
  const result = spawnSync(
    pythonPath,
    ['-c', script, sourcePath],
    { encoding: 'utf8', timeout: 10000 },
  )
  if (result.status !== 0) {
    const stderr = result.stderr?.toString().trim() || ''
    throw new Error(`Pillow dimension probe failed (code ${result.status}): ${stderr}`)
  }

  const [widthText, heightText] = (result.stdout || '').trim().split(',')
  const width = Number(widthText)
  const height = Number(heightText)
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error(`Invalid image dimensions for ${sourcePath}: ${(result.stdout || '').trim()}`)
  }

  return { width, height }
}
