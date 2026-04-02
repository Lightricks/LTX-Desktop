import { extractVideoFrameToFile } from '../export/ffmpeg-utils'
import { handle } from './typed-handle'

export function registerVideoProcessingHandlers(): void {
  handle('extractVideoFrame', async ({ videoPath, seekTime, width, quality }) => {
    return {
      path: extractVideoFrameToFile({
        videoPath,
        seekTime,
        width,
        quality: quality ?? 2,
        timeoutMs: 10000,
      }),
    }
  })
}
