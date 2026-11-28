import type { FlatSegment } from './timeline'

export interface ExportSubtitle {
  text: string; startTime: number; endTime: number;
  style: { fontSize: number; fontFamily: string; fontWeight: string; color: string; backgroundColor: string; position: string; italic: boolean };
}

/**
 * Build the ffmpeg filter_complex script and input arguments for the video-only pass.
 * Pure string building — zero I/O.
 */
export function buildVideoFilterGraph(
  segments: FlatSegment[],
  opts: {
    width: number; height: number; fps: number;
    letterbox?: { ratio: number; color: string; opacity: number };
    subtitles?: ExportSubtitle[];
  },
): { inputs: string[]; filterScript: string } {
  const { width, height, fps, letterbox, subtitles } = opts
  const inputs: string[] = []
  const filterParts: string[] = []
  let idx = 0

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i]

    if (seg.type === 'gap') {
      // Gap: generate black frames at target fps (synthetic input)
      inputs.push('-f', 'lavfi', '-i', `color=c=black:s=${width}x${height}:r=${fps}:d=${seg.duration.toFixed(6)}`)
      filterParts.push(`[${idx}:v]setsar=1[v${i}]`)
      idx++
    } else if (seg.type === 'image') {
      // Image: loop for exact duration, use target fps for frame generation
      inputs.push('-loop', '1', '-framerate', String(fps), '-t', seg.duration.toFixed(6), '-i', seg.filePath)
      let chain = `[${idx}:v]scale=${width}:${height}:force_original_aspect_ratio=decrease,pad=${width}:${height}:-1:-1:color=black,setsar=1`
      if (seg.flipH) chain += ',hflip'
      if (seg.flipV) chain += ',vflip'
      chain += `[v${i}]`
      filterParts.push(chain)
      idx++
    } else {
      // Video: trim -> speed -> scale, NO per-segment fps conversion
      // (fps is applied ONCE after concat to avoid per-segment duration quantization)
      const trimEnd = seg.trimStart + seg.duration * seg.speed
      inputs.push('-i', seg.filePath)
      let chain = `[${idx}:v]trim=start=${seg.trimStart.toFixed(6)}:end=${trimEnd.toFixed(6)},setpts=PTS-STARTPTS`
      if (seg.speed !== 1) chain += `,setpts=PTS/${seg.speed.toFixed(6)}`
      if (seg.reversed) chain += ',reverse'
      chain += `,scale=${width}:${height}:force_original_aspect_ratio=decrease,pad=${width}:${height}:-1:-1:color=black,setsar=1`
      if (seg.flipH) chain += ',hflip'
      if (seg.flipV) chain += ',vflip'
      chain += `[v${i}]`
      filterParts.push(chain)
      idx++
    }
  }

  const concatInputs = segments.map((_, i) => `[v${i}]`).join('')

  // Concat all segments, then apply fps ONCE to the entire output.
  // This is how real NLEs work: frame rate conversion happens globally,
  // not per-clip, so per-segment duration quantization doesn't accumulate.
  let lastLabel = 'fpsout'
  filterParts.push(`${concatInputs}concat=n=${segments.length}:v=1:a=0[concatraw]`)
  filterParts.push(`[concatraw]fps=${fps}[${lastLabel}]`)

  // Letterbox overlay (drawbox)
  if (letterbox) {
    const containerRatio = width / height
    const targetRatio = letterbox.ratio
    const hexColor = letterbox.color.replace('#', '')
    const alphaHex = Math.round(letterbox.opacity * 255).toString(16).padStart(2, '0')
    const colorStr = `0x${hexColor}${alphaHex}`
    const nextLabel = 'lbout'

    if (targetRatio >= containerRatio) {
      // Letterbox: bars on top and bottom
      const visibleH = Math.round(width / targetRatio)
      const barH = Math.round((height - visibleH) / 2)
      if (barH > 0) {
        filterParts.push(`[${lastLabel}]drawbox=x=0:y=0:w=iw:h=${barH}:c=${colorStr}:t=fill,drawbox=x=0:y=ih-${barH}:w=iw:h=${barH}:c=${colorStr}:t=fill[${nextLabel}]`)
        lastLabel = nextLabel
      }
    } else {
      // Pillarbox: bars on left and right
      const visibleW = Math.round(height * targetRatio)
      const barW = Math.round((width - visibleW) / 2)
      if (barW > 0) {
        filterParts.push(`[${lastLabel}]drawbox=x=0:y=0:w=${barW}:h=ih:c=${colorStr}:t=fill,drawbox=x=iw-${barW}:y=0:w=${barW}:h=ih:c=${colorStr}:t=fill[${nextLabel}]`)
        lastLabel = nextLabel
      }
    }
  }

  // Subtitle burn-in (drawtext)
  if (subtitles && subtitles.length > 0) {
    for (let si = 0; si < subtitles.length; si++) {
      const sub = subtitles[si]
      const nextLabel = `sub${si}`
      // Escape text for ffmpeg drawtext: replace special chars
      const escapedText = sub.text
        .replace(/\\/g, '\\\\\\\\')
        .replace(/'/g, "'\\\\\\''")
        .replace(/:/g, '\\:')
        .replace(/%/g, '%%')
        .replace(/\n/g, '\\n')

      const fontSize = Math.round(sub.style.fontSize * (height / 1080)) // scale relative to export res
      const fontColor = sub.style.color.replace('#', '0x')

      // Y position based on style.position
      let yExpr: string
      if (sub.style.position === 'top') {
        yExpr = '20'
      } else if (sub.style.position === 'center') {
        yExpr = '(h-text_h)/2'
      } else {
        yExpr = 'h-text_h-30'
      }

      // Background box
      let boxPart = ''
      if (sub.style.backgroundColor && sub.style.backgroundColor !== 'transparent') {
        const bgHex = sub.style.backgroundColor.replace('#', '')
        // Handle 8-char hex with alpha (e.g., 00000099)
        const bgColor = bgHex.length > 6 ? `0x${bgHex.slice(0, 6)}` : `0x${bgHex}`
        const bgAlpha = bgHex.length > 6 ? (parseInt(bgHex.slice(6), 16) / 255).toFixed(2) : '0.6'
        boxPart = `:box=1:boxcolor=${bgColor}@${bgAlpha}:boxborderw=8`
      }

      const dtFilter = `drawtext=text='${escapedText}':fontsize=${fontSize}:fontcolor=${fontColor}:x=(w-text_w)/2:y=${yExpr}${boxPart}:enable='between(t\\,${sub.startTime.toFixed(3)}\\,${sub.endTime.toFixed(3)})'`

      filterParts.push(`[${lastLabel}]${dtFilter}[${nextLabel}]`)
      lastLabel = nextLabel
    }
  }

  // Rename final label to outv
  if (lastLabel !== 'outv') {
    filterParts.push(`[${lastLabel}]null[outv]`)
  }

  return { inputs, filterScript: filterParts.join(';\n') }
}
