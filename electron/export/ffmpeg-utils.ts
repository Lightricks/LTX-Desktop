import { spawn, spawnSync, ChildProcess, execSync } from 'child_process'
import os from 'os'
import path from 'path'
import fs from 'fs'
import { isDev, getCurrentDir } from '../config'
import { logger } from '../logger'
import { getPythonDir } from '../python-setup'

let activeExportProcess: ChildProcess | null = null

export function findFfmpegPath(): string | null {
  let binDir: string | null = null

  if (process.platform === 'win32') {
    const imageioRelPath = path.join('Lib', 'site-packages', 'imageio_ffmpeg', 'binaries')
    binDir = isDev
      ? path.join(getCurrentDir(), 'backend', '.venv', imageioRelPath)
      : path.join(getPythonDir(), imageioRelPath)
  } else {
    // macOS/Linux: find lib/python3.X/site-packages dynamically
    const venvBase = isDev
      ? path.join(getCurrentDir(), 'backend', '.venv')
      : getPythonDir()
    const libDir = path.join(venvBase, 'lib')
    if (fs.existsSync(libDir)) {
      const pythonDir = fs.readdirSync(libDir).find(e => e.startsWith('python3'))
      if (pythonDir) {
        binDir = path.join(libDir, pythonDir, 'site-packages', 'imageio_ffmpeg', 'binaries')
      }
    }
  }

  if (binDir && fs.existsSync(binDir)) {
    const bin = fs.readdirSync(binDir).find(f => f.startsWith('ffmpeg') && (f.endsWith('.exe') || !f.includes('.')))
    if (bin) return path.join(binDir, bin)
  }

  try { execSync('ffmpeg -version', { stdio: 'ignore' }); return 'ffmpeg' } catch { return null }
}

/** Check if a video file contains an audio stream using ffprobe/ffmpeg */
export function fileHasAudio(ffmpegPath: string, filePath: string): boolean {
  try {
    const result = spawnSync(ffmpegPath, ['-i', filePath, '-hide_banner'], {
      encoding: 'utf8',
      timeout: 5000,
    })
    const output = (result.stdout || '') + (result.stderr || '')
    return output.includes('Audio:')
  } catch {
    return false
  }
}


/** Run an ffmpeg command and return a promise. Logs stderr and sets activeExportProcess. */
export function runFfmpeg(ffmpegPath: string, args: string[]): Promise<{ success: boolean; error?: string }> {
  return new Promise((resolve) => {
    logger.info( `[ffmpeg] spawn: ${args.join(' ').slice(0, 400)}`)
    const proc = spawn(ffmpegPath, args, { stdio: ['pipe', 'pipe', 'pipe'] })
    activeExportProcess = proc
    let stderrLog = ''
    proc.stderr?.on('data', (chunk: Buffer) => {
      const text = chunk.toString()
      stderrLog += text
      const lines = text.trim().split('\n')
      for (const line of lines) {
        if (line.includes('frame=') || line.includes('Error') || line.includes('error')) {
          logger.info( `[ffmpeg] ${line.trim().slice(0, 200)}`)
        }
      }
    })
    proc.on('close', (code) => {
      activeExportProcess = null
      if (code === 0) {
        resolve({ success: true })
      } else {
        const errLines = stderrLog.split('\n').filter(l => l.trim()).slice(-5).join('\n')
        logger.error( `[ffmpeg] exited ${code}:\n${errLines}`)
        resolve({ success: false, error: `FFmpeg failed (code ${code}): ${errLines.slice(0, 300)}` })
      }
    })
    proc.on('error', (err) => {
      activeExportProcess = null
      resolve({ success: false, error: `Failed to start ffmpeg: ${err.message}` })
    })
  })
}

function runFfmpegSyncOrThrow(ffmpegPath: string, args: string[], timeoutMs = 30000): void {
  logger.info(`[ffmpeg-sync] spawn: ${args.join(' ').slice(0, 400)}`)
  const result = spawnSync(ffmpegPath, args, { timeout: timeoutMs })
  if (result.status === 0) return
  const stderr = (result.stderr?.toString() || '').split('\n').filter(Boolean).slice(-5).join('\n')
  throw new Error(`FFmpeg failed (code ${result.status}): ${stderr.slice(0, 300)}`)
}

export function extractVideoFrameToFile({
  videoPath,
  seekTime,
  width,
  quality,
  outputPath,
  timeoutMs = 10000,
}: {
  videoPath: string
  seekTime: number
  width?: number
  quality?: number
  outputPath?: string
  timeoutMs?: number
}): string {
  const ffmpegPath = findFfmpegPath()
  if (!ffmpegPath) {
    throw new Error('ffmpeg not found')
  }
  if (!fs.existsSync(videoPath)) {
    throw new Error(`Video file not found: ${videoPath}`)
  }

  const resolvedOutputPath = outputPath
    ?? path.join(
      os.tmpdir(),
      `ltx_frame_${Date.now()}_${Math.random().toString(36).slice(2, 8)}.jpg`,
    )

  const args: string[] = [
    '-ss', String(Math.max(0, seekTime)),
    '-i', videoPath,
    ...(width ? ['-vf', `scale=${width}:-2`] : []),
    '-frames:v', '1',
    ...(quality !== undefined ? ['-q:v', String(quality)] : []),
    '-y',
    resolvedOutputPath,
  ]

  logger.info(`[extract-frame] ${args.join(' ').slice(0, 300)}`)
  runFfmpegSyncOrThrow(ffmpegPath, args, timeoutMs)

  if (!fs.existsSync(resolvedOutputPath)) {
    throw new Error('ffmpeg produced no output file')
  }

  return resolvedOutputPath
}

export function getVideoDimensions(videoPath: string): { width: number; height: number } {
  const ffmpegPath = findFfmpegPath()
  if (!ffmpegPath) {
    throw new Error('ffmpeg not found')
  }
  if (!fs.existsSync(videoPath)) {
    throw new Error(`Video file not found: ${videoPath}`)
  }

  const result = spawnSync(ffmpegPath, ['-hide_banner', '-i', videoPath], {
    encoding: 'utf8',
    timeout: 10000,
  })
  const output = `${result.stdout || ''}\n${result.stderr || ''}`
  const videoStreamLine = output.split('\n').find(line => line.includes('Video:'))
  const match = videoStreamLine?.match(/(\d{2,5})x(\d{2,5})(?:[,\s\[]|$)/)

  if (!match) {
    throw new Error(`Could not determine video dimensions for ${videoPath}`)
  }

  const width = Number(match[1])
  const height = Number(match[2])
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error(`Invalid video dimensions for ${videoPath}: ${match[1]}x${match[2]}`)
  }

  return { width, height }
}

export function stopExportProcess(): void {
  if (activeExportProcess) {
    logger.info( 'Stopping active export process...')
    activeExportProcess.kill()
    activeExportProcess = null
  }
}
