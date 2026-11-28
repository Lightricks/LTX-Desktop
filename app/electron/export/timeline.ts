export interface ExportClip {
  path: string; type: string; startTime: number; duration: number; trimStart: number;
  speed: number; reversed: boolean; flipH: boolean; flipV: boolean; opacity: number; trackIndex: number;
  muted: boolean; volume: number;
}

export interface FlatSegment {
  filePath: string; type: string; startTime: number; duration: number; trimStart: number;
  speed: number; reversed: boolean; flipH: boolean; flipV: boolean; opacity: number;
  muted: boolean; volume: number;
}

/**
 * Flatten a multi-track timeline into a sequence of segments for ffmpeg concat.
 * At each point in time, the highest trackIndex wins for video (NLE convention).
 */
export function flattenTimeline(clips: ExportClip[]): FlatSegment[] {
  // Only consider video/image clips for visual flattening
  const videoClips = clips.filter(c => c.type === 'video' || c.type === 'image')
  if (videoClips.length === 0) return []

  // Collect all time boundaries
  const boundaries = new Set<number>()
  boundaries.add(0)
  for (const c of videoClips) {
    boundaries.add(c.startTime)
    boundaries.add(c.startTime + c.duration)
  }
  const sorted = [...boundaries].sort((a, b) => a - b)

  const segments: FlatSegment[] = []

  for (let i = 0; i < sorted.length - 1; i++) {
    const t0 = sorted[i]
    const t1 = sorted[i + 1]
    const segDur = t1 - t0
    if (segDur < 0.001) continue

    const mid = (t0 + t1) / 2
    // Find highest-track clip at this time
    const active = videoClips
      .filter(c => mid >= c.startTime && mid < c.startTime + c.duration)
      .sort((a, b) => b.trackIndex - a.trackIndex)

    if (active.length > 0) {
      const c = active[0]
      const offsetInClip = t0 - c.startTime
      segments.push({
        filePath: c.path,
        type: c.type,
        startTime: t0,
        duration: segDur,
        trimStart: c.trimStart + offsetInClip * c.speed,
        speed: c.speed,
        reversed: c.reversed,
        flipH: c.flipH,
        flipV: c.flipV,
        opacity: c.opacity,
        muted: c.muted,
        volume: c.volume,
      })
    } else {
      segments.push({
        filePath: '', type: 'gap', startTime: t0, duration: segDur, trimStart: 0,
        speed: 1, reversed: false, flipH: false, flipV: false, opacity: 100,
        muted: true, volume: 0,
      })
    }
  }

  // Merge adjacent segments from the same file with contiguous trim
  const merged: FlatSegment[] = []
  for (const seg of segments) {
    const prev = merged[merged.length - 1]
    if (prev && prev.filePath === seg.filePath && prev.filePath !== '' &&
        prev.speed === seg.speed && prev.reversed === seg.reversed &&
        prev.flipH === seg.flipH && prev.flipV === seg.flipV &&
        prev.opacity === seg.opacity && prev.muted === seg.muted && prev.volume === seg.volume &&
        Math.abs((prev.trimStart + prev.duration * prev.speed) - seg.trimStart) < 0.01) {
      prev.duration += seg.duration
    } else {
      merged.push({ ...seg })
    }
  }

  return merged
}
