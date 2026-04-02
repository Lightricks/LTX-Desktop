/**
 * Timeline Import: Parse FCP7 XML and FCPXML files from Premiere Pro, DaVinci Resolve, Final Cut Pro
 * 
 * Supported formats:
 * - FCP 7 XML (.xml) - Universal interchange format (Premiere, DaVinci, FCP7)
 * - FCPXML (.fcpxml) - Final Cut Pro X / DaVinci Resolve
 * 
 * AAF is a binary format and cannot be parsed in JavaScript.
 * Users should export as XML from their NLE instead.
 */

// Intermediate representation of a parsed timeline
export interface ParsedMediaRef {
  id: string
  name: string
  path: string           // Effective local file path used for import
  bigThumbnailPath?: string
  smallThumbnailPath?: string
  duration: number       // in seconds
  type: 'video' | 'audio' | 'image'
  width?: number
  height?: number
  fps?: number
}

export interface ParsedClip {
  name: string
  mediaRefId: string     // references a ParsedMediaRef
  trackIndex: number
  trackType: 'video' | 'audio'
  startTime: number      // position on timeline in seconds
  duration: number       // duration on timeline in seconds
  sourceIn: number       // source in point in seconds
  sourceOut: number      // source out point in seconds
  speed?: number         // playback speed multiplier (1 = normal, 2 = 2x, 0.5 = half)
  reversed?: boolean     // true if clip is playing in reverse
  volume?: number        // audio volume 0-1
  muted?: boolean        // true if audio is muted / clip is disabled
  flipH?: boolean        // horizontal flip
  flipV?: boolean        // vertical flip
  opacity?: number       // opacity 0-100 (100 = fully visible)
  linkedVideoClipIndex?: number  // For audio clips: index of the matching video clip in the clips array (for establishing links)
}

export interface ParsedTimeline {
  name: string
  fps: number
  duration: number       // total duration in seconds
  width?: number
  height?: number
  videoTrackCount: number
  audioTrackCount: number
  mediaRefs: ParsedMediaRef[]
  clips: ParsedClip[]
  format: 'fcp7xml' | 'fcpxml' | 'unknown'
}

// ─── Helper: decode pathurl to local file path ─────────────────────────────
function decodePathUrl(pathUrl: string): string {
  if (!pathUrl) return ''
  
  let decoded = pathUrl.trim()
  
  // Remove file:// or file:/// or file://localhost/ prefix
  decoded = decoded
    .replace(/^file:\/\/localhost\//i, '/')
    .replace(/^file:\/\/\//i, '/')
    .replace(/^file:\/\//i, '/')
  
  // URL decode
  try {
    decoded = decodeURIComponent(decoded)
  } catch {
    // ignore decode errors
  }
  
  // On Windows, paths start with /C: → remove leading slash
  if (/^\/[A-Za-z]:/.test(decoded)) {
    decoded = decoded.slice(1)
  }
  
  // Normalize slashes to OS-appropriate
  if (typeof navigator !== 'undefined' && navigator.platform?.startsWith('Win')) {
    decoded = decoded.replace(/\//g, '\\')
  }
  
  return decoded
}

// ─── Helper: get text content of a child element ───────────────────────────
function getChildText(parent: Element, tagName: string): string {
  const el = parent.querySelector(`:scope > ${tagName}`)
  return el?.textContent?.trim() || ''
}

function getChildNumber(parent: Element, tagName: string): number {
  const text = getChildText(parent, tagName)
  return text ? parseFloat(text) : 0
}

// ─── Helper: get timebase (fps) from a rate element ────────────────────────
function getTimebase(parent: Element): number {
  const rateEl = parent.querySelector(':scope > rate')
  if (!rateEl) return 24
  const tb = getChildNumber(rateEl, 'timebase')
  const ntsc = getChildText(rateEl, 'ntsc').toLowerCase() === 'true'
  // NTSC drop-frame: 30 → 29.97, 24 → 23.976, 60 → 59.94
  if (ntsc && tb > 0) {
    return tb * (1000 / 1001)
  }
  return tb || 24
}

// ─── Helper: detect media type from file extension ─────────────────────────
function detectMediaType(filename: string): 'video' | 'audio' | 'image' {
  const ext = filename.split('.').pop()?.toLowerCase() || ''
  const videoExts = ['mp4', 'mov', 'avi', 'mkv', 'wmv', 'flv', 'webm', 'mxf', 'r3d', 'braw', 'ari']
  const audioExts = ['mp3', 'wav', 'aac', 'flac', 'ogg', 'm4a', 'wma', 'aiff', 'aif']
  const imageExts = ['jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp', 'gif', 'webp', 'exr', 'dpx', 'psd']
  
  if (audioExts.includes(ext)) return 'audio'
  if (imageExts.includes(ext)) return 'image'
  if (videoExts.includes(ext)) return 'video'
  return 'video' // default
}

// ─── Helper: parse clip effects (FCP7) ───────────────────────────────────
// Extracts speed, reverse, flips, opacity, muted from <filter>/<effect> and
// other clip-level elements in FCP 7 XML.
interface ParsedClipEffects {
  speed: number
  reversed: boolean
  flipH: boolean
  flipV: boolean
  opacity: number  // 0-100
  muted: boolean
}

function parseFcp7ClipEffects(clipEl: Element): ParsedClipEffects {
  const result: ParsedClipEffects = {
    speed: 1,
    reversed: false,
    flipH: false,
    flipV: false,
    opacity: 100,
    muted: false,
  }

  // ── Check <enabled>FALSE</enabled> → clip is disabled/muted ──
  const enabled = getChildText(clipEl, 'enabled')
  if (enabled.toUpperCase() === 'FALSE') {
    result.muted = true
  }

  // ── Collect all effects ──
  const effects = clipEl.querySelectorAll(':scope > filter > effect')
  effects.forEach(effect => {
    const effectId = getChildText(effect, 'effectid').toLowerCase()
    const effectName = getChildText(effect, 'name').toLowerCase()

    // ── Speed / Time Remap ──
    if (effectId.includes('timeremap') || effectId.includes('speed') ||
        effectName.includes('time remap') || effectName.includes('speed')) {
      // Get all parameters
      const params = effect.querySelectorAll(':scope > parameter')
      params.forEach(param => {
        const paramId = getChildText(param, 'parameterid').toLowerCase()
        const paramName = getChildText(param, 'name').toLowerCase()

        if (paramId === 'speed' || paramName === 'speed') {
          const val = getChildText(param, 'value')
          if (val) {
            // Speed is stored as percentage (100 = normal, 200 = 2x, 50 = half)
            const speedPercent = parseFloat(val)
            if (!isNaN(speedPercent) && speedPercent !== 0) {
              // Negative speed = reverse (in some exports)
              if (speedPercent < 0) {
                result.speed = Math.abs(speedPercent) / 100
                result.reversed = true
              } else {
                result.speed = speedPercent / 100
              }
            }
          }
        }

        if (paramId === 'reverse' || paramName === 'reverse') {
          const val = getChildText(param, 'value').toUpperCase()
          if (val === 'TRUE' || val === '1') {
            result.reversed = true
          }
        }
      })
    }

    // ── Horizontal Flip ──
    if (effectId.includes('horizflip') || effectId.includes('hflip') ||
        effectName.includes('horizontal flip') || effectName.includes('flip horizontal')) {
      result.flipH = true
    }

    // ── Vertical Flip ──
    if (effectId.includes('vertflip') || effectId.includes('vflip') ||
        effectName.includes('vertical flip') || effectName.includes('flip vertical')) {
      result.flipV = true
    }

    // ── Basic Motion — check for negative scale (flip via scale) ──
    if (effectId === 'basic' || effectName.includes('basic motion') || effectName.includes('motion')) {
      const params = effect.querySelectorAll(':scope > parameter')
      params.forEach(param => {
        const paramId = getChildText(param, 'parameterid').toLowerCase()

        // Scale X negative = horizontal flip
        if (paramId === 'scalex' || paramId === 'scale_x') {
          const val = parseFloat(getChildText(param, 'value'))
          if (!isNaN(val) && val < 0) result.flipH = true
        }
        // Scale Y negative = vertical flip
        if (paramId === 'scaley' || paramId === 'scale_y') {
          const val = parseFloat(getChildText(param, 'value'))
          if (!isNaN(val) && val < 0) result.flipV = true
        }
      })
    }

    // ── Opacity ──
    if (effectId === 'opacity' || effectId.includes('opacity') ||
        effectName.includes('opacity')) {
      const params = effect.querySelectorAll(':scope > parameter')
      params.forEach(param => {
        const paramId = getChildText(param, 'parameterid').toLowerCase()
        if (paramId === 'opacity' || paramId === 'level') {
          const val = getChildText(param, 'value')
          if (val) {
            const opacityVal = parseFloat(val)
            if (!isNaN(opacityVal)) {
              result.opacity = Math.max(0, Math.min(100, opacityVal))
            }
          }
        }
      })
    }
  })

  // ── Check <compositemode> for opacity (alternative location) ──
  const compositeOpacity = clipEl.querySelector(':scope > compositemode > opacity > value')
  if (compositeOpacity) {
    const val = parseFloat(compositeOpacity.textContent || '')
    if (!isNaN(val)) {
      result.opacity = Math.max(0, Math.min(100, val))
    }
  }

  return result
}

// ═══════════════════════════════════════════════════════════════════════════
// FCP 7 XML Parser
// ═══════════════════════════════════════════════════════════════════════════

function parseFcp7Xml(doc: Document): ParsedTimeline | null {
  // Look for sequence element
  const sequence = doc.querySelector('xmeml > sequence') 
    || doc.querySelector('xmeml > project > children > sequence')
    || doc.querySelector('sequence')
  
  if (!sequence) return null
  
  const name = getChildText(sequence, 'name') || 'Imported Timeline'
  const fps = getTimebase(sequence)
  const totalDuration = getChildNumber(sequence, 'duration') / fps
  
  // Sequence format (resolution)
  const seqFormat = sequence.querySelector(':scope > media > video > format > samplecharacteristics')
  const seqWidth = seqFormat ? getChildNumber(seqFormat, 'width') : undefined
  const seqHeight = seqFormat ? getChildNumber(seqFormat, 'height') : undefined
  
  // Collect all file references (deduplicating by id)
  const mediaRefs = new Map<string, ParsedMediaRef>()
  const fileElements = sequence.querySelectorAll('file')
  
  fileElements.forEach(fileEl => {
    const fileId = fileEl.getAttribute('id') || ''
    if (!fileId || mediaRefs.has(fileId)) return
    
    // Some file elements are just references (no children) - skip those
    const xmlPath = getChildText(fileEl, 'pathurl')
    if (!xmlPath && fileEl.children.length === 0) return
    
    const decodedPath = decodePathUrl(xmlPath)
    const fileName = getChildText(fileEl, 'name') || decodedPath.split(/[/\\]/).pop() || 'Unknown'
    const fileFps = getTimebase(fileEl)
    const fileDuration = getChildNumber(fileEl, 'duration') / (fileFps || fps)
    
    // Get media dimensions
    const mediaVideo = fileEl.querySelector(':scope > media > video > samplecharacteristics')
    const width = mediaVideo ? getChildNumber(mediaVideo, 'width') : undefined
    const height = mediaVideo ? getChildNumber(mediaVideo, 'height') : undefined
    
    const type = detectMediaType(fileName)
    
    mediaRefs.set(fileId, {
      id: fileId,
      name: fileName,
      path: decodedPath,
      duration: fileDuration,
      type,
      width,
      height,
      fps: fileFps,
    })
  })
  
  // Parse clips from video tracks
  const clips: ParsedClip[] = []
  let videoTrackCount = 0
  let audioTrackCount = 0
  
  // Video tracks
  const videoTracks = sequence.querySelectorAll(':scope > media > video > track')
  videoTracks.forEach((trackEl, trackIdx) => {
    videoTrackCount++
    const clipItems = trackEl.querySelectorAll(':scope > clipitem')
    
    clipItems.forEach(clipEl => {
      const clipName = getChildText(clipEl, 'name')
      const clipFps = getTimebase(clipEl) || fps
      
      const startFrame = getChildNumber(clipEl, 'start')
      const endFrame = getChildNumber(clipEl, 'end')
      const inFrame = getChildNumber(clipEl, 'in')
      const outFrame = getChildNumber(clipEl, 'out')
      
      // Find the file reference
      const fileEl = clipEl.querySelector(':scope > file')
      let mediaRefId = fileEl?.getAttribute('id') || ''
      
      // If file element has no children, it's a reference - look it up
      if (fileEl && fileEl.children.length === 0 && mediaRefId) {
        // Just a reference to an existing file
      } else if (fileEl && mediaRefId) {
        // Full file definition inline - make sure it's in our map
        if (!mediaRefs.has(mediaRefId)) {
          const xmlPath = getChildText(fileEl, 'pathurl')
          const decodedPath = decodePathUrl(xmlPath)
          const fileName = getChildText(fileEl, 'name') || decodedPath.split(/[/\\]/).pop() || clipName
          mediaRefs.set(mediaRefId, {
            id: mediaRefId,
            name: fileName,
            path: decodedPath,
            duration: (outFrame - inFrame) / clipFps,
            type: detectMediaType(fileName),
          })
        }
      }
      
      // Parse all clip effects (speed, reverse, flips, opacity, muted)
      const effects = parseFcp7ClipEffects(clipEl)
      
      if (startFrame >= 0 && endFrame > startFrame && mediaRefId) {
        clips.push({
          name: clipName,
          mediaRefId,
          trackIndex: trackIdx,
          trackType: 'video',
          startTime: startFrame / clipFps,
          duration: (endFrame - startFrame) / clipFps,
          sourceIn: inFrame / clipFps,
          sourceOut: outFrame / clipFps,
          speed: effects.speed !== 1 ? effects.speed : undefined,
          reversed: effects.reversed || undefined,
          flipH: effects.flipH || undefined,
          flipV: effects.flipV || undefined,
          opacity: effects.opacity !== 100 ? effects.opacity : undefined,
          muted: effects.muted || undefined,
        })
      }
    })
  })
  
  // Audio tracks
  const audioTracks = sequence.querySelectorAll(':scope > media > audio > track')
  audioTracks.forEach((trackEl, audioIdx) => {
    audioTrackCount++
    const clipItems = trackEl.querySelectorAll(':scope > clipitem')
    
    clipItems.forEach(clipEl => {
      const clipName = getChildText(clipEl, 'name')
      const clipFps = getTimebase(clipEl) || fps
      
      const startFrame = getChildNumber(clipEl, 'start')
      const endFrame = getChildNumber(clipEl, 'end')
      const inFrame = getChildNumber(clipEl, 'in')
      const outFrame = getChildNumber(clipEl, 'out')
      
      const fileEl = clipEl.querySelector(':scope > file')
      const mediaRefId = fileEl?.getAttribute('id') || ''
      
      // Parse effects (muted via <enabled>FALSE</enabled>)
      const effects = parseFcp7ClipEffects(clipEl)
      
      // Check volume from audio level filter
      let volume: number | undefined
      const volFilter = Array.from(clipEl.querySelectorAll('filter > effect')).find(e => 
        getChildText(e, 'effectid')?.toLowerCase().includes('audiolevel') ||
        getChildText(e, 'name')?.toLowerCase().includes('level')
      )
      if (volFilter) {
        const volParam = volFilter.querySelector('parameter > value')
        if (volParam) {
          volume = parseFloat(volParam.textContent || '0')
          // FCP7 volume is in dB, convert rough approximation: 0dB=1, -inf=0
          volume = Math.pow(10, (volume || 0) / 20)
        }
      }
      
      // Find matching video clip (same file, same timeline position) to establish a link
      const linkedVideoIndex = clips.findIndex(c => 
        c.mediaRefId === mediaRefId && 
        c.trackType === 'video' &&
        Math.abs(c.startTime - startFrame / clipFps) < 0.01
      )
      
      if (startFrame >= 0 && endFrame > startFrame && mediaRefId) {
        clips.push({
          name: clipName,
          mediaRefId,
          trackIndex: videoTrackCount + audioIdx, // offset by video tracks
          trackType: 'audio',
          startTime: startFrame / clipFps,
          duration: (endFrame - startFrame) / clipFps,
          sourceIn: inFrame / clipFps,
          sourceOut: outFrame / clipFps,
          volume,
          muted: effects.muted || undefined,
          linkedVideoClipIndex: linkedVideoIndex >= 0 ? linkedVideoIndex : undefined,
        })
      }
    })
  })
  
  // ── Merge stereo pairs: Premiere exports stereo as two mono audio tracks ──
  // If multiple audio clips reference the same source file at the same timeline
  // position, they are channel splits of a single stereo clip. Keep only one
  // clip per unique (mediaRefId, startTime) and collapse the track count.
  const audioClips = clips.filter(c => c.trackType === 'audio')
  const nonAudioClips = clips.filter(c => c.trackType !== 'audio')
  
  // Group audio clips by (mediaRefId + startTime) to detect stereo pairs
  const audioGroups = new Map<string, ParsedClip[]>()
  for (const ac of audioClips) {
    const key = `${ac.mediaRefId}|${ac.startTime.toFixed(4)}`
    if (!audioGroups.has(key)) audioGroups.set(key, [])
    audioGroups.get(key)!.push(ac)
  }
  
  // Build merged audio clips: keep the first clip of each group, discard duplicates
  const mergedAudioClips: ParsedClip[] = []
  let mergedAudioTrackIdx = 0
  // Track which original track indices have been assigned a merged index
  const trackRemap = new Map<number, number>()
  
  for (const group of audioGroups.values()) {
    const representative = group[0] // keep first channel's metadata (volume, link, etc.)
    
    // Determine the merged track index for this clip
    const origTrack = representative.trackIndex
    if (!trackRemap.has(origTrack)) {
      trackRemap.set(origTrack, videoTrackCount + mergedAudioTrackIdx)
      mergedAudioTrackIdx++
    }
    representative.trackIndex = trackRemap.get(origTrack)!
    
    // If any channel in the group has a linked video, preserve it
    if (representative.linkedVideoClipIndex === undefined) {
      const linked = group.find(c => c.linkedVideoClipIndex !== undefined)
      if (linked) representative.linkedVideoClipIndex = linked.linkedVideoClipIndex
    }
    
    mergedAudioClips.push(representative)
  }
  
  // Reassemble clips and update track count
  const finalClips = [...nonAudioClips, ...mergedAudioClips]
  const finalAudioTrackCount = mergedAudioTrackIdx || (audioTrackCount > 0 ? 1 : 0)
  
  return {
    name,
    fps,
    duration: totalDuration || Math.max(...finalClips.map(c => c.startTime + c.duration), 0),
    width: seqWidth,
    height: seqHeight,
    videoTrackCount,
    audioTrackCount: finalAudioTrackCount,
    mediaRefs: Array.from(mediaRefs.values()),
    clips: finalClips,
    format: 'fcp7xml',
  }
}

// ─── Helper: parse clip effects from FCPXML elements ─────────────────────
// FCPXML stores speed in <timeMap> or <conform-rate>, flips via <filter-video> transform,
// opacity as an attribute, etc.
function parseFcpxmlClipEffects(el: Element): Partial<ParsedClipEffects> {
  const result: Partial<ParsedClipEffects> = {}

  // ── Enabled attribute (enabled="0" = muted/disabled) ──
  const enabled = el.getAttribute('enabled')
  if (enabled === '0') {
    result.muted = true
  }

  // ── Speed: check for <timeMap> with remapped speed ──
  const timeMap = el.querySelector(':scope > timeMap')
  if (timeMap) {
    const timepts = timeMap.querySelectorAll(':scope > timept')
    // Simple constant speed: 2 timepts → calculate speed ratio
    if (timepts.length >= 2) {
      const parseFcpxmlTimeLocal = (timeStr: string | null): number => {
        if (!timeStr) return 0
        const rational = timeStr.trim().match(/^(\d+)\/(\d+)s$/)
        if (rational) return parseInt(rational[1]) / parseInt(rational[2])
        const simple = timeStr.trim().match(/^([\d.]+)s$/)
        if (simple) return parseFloat(simple[1])
        return 0
      }
      const first = timepts[0]
      const last = timepts[timepts.length - 1]
      const srcDur = parseFcpxmlTimeLocal(last.getAttribute('time')) - parseFcpxmlTimeLocal(first.getAttribute('time'))
      const dstDur = parseFcpxmlTimeLocal(last.getAttribute('value')) - parseFcpxmlTimeLocal(first.getAttribute('value'))
      if (srcDur > 0 && dstDur > 0) {
        result.speed = dstDur / srcDur
      }
    }
  }

  // ── Speed: check for <conform-rate> ──
  const conformRate = el.querySelector(':scope > conform-rate')
  if (conformRate) {
    const scaleEnabled = conformRate.getAttribute('scaleEnabled')
    if (scaleEnabled === '1') {
      const srcFrameDur = conformRate.getAttribute('srcFrameDuration')
      const frameDur = conformRate.getAttribute('frameDuration')
      if (srcFrameDur && frameDur) {
        const parseFcpxmlTimeLocal = (timeStr: string): number => {
          const rational = timeStr.trim().match(/^(\d+)\/(\d+)s$/)
          if (rational) return parseInt(rational[1]) / parseInt(rational[2])
          const simple = timeStr.trim().match(/^([\d.]+)s$/)
          if (simple) return parseFloat(simple[1])
          return 0
        }
        const src = parseFcpxmlTimeLocal(srcFrameDur)
        const dst = parseFcpxmlTimeLocal(frameDur)
        if (src > 0 && dst > 0) {
          result.speed = src / dst
        }
      }
    }
  }

  // ── Flips and opacity: check <adjust-transform> ──
  const adjustTransform = el.querySelector(':scope > adjust-transform')
  if (adjustTransform) {
    // Flips: scaleX="-100" or scaleY="-100"
    const scaleX = parseFloat(adjustTransform.getAttribute('scaleX') || '100')
    const scaleY = parseFloat(adjustTransform.getAttribute('scaleY') || '100')
    if (scaleX < 0) result.flipH = true
    if (scaleY < 0) result.flipV = true
  }

  // ── Flips: check <filter-video> with "Flipped" or transform ──
  const filterVideos = el.querySelectorAll(':scope > filter-video')
  filterVideos.forEach(fv => {
    const filterName = (fv.getAttribute('name') || '').toLowerCase()
    const filterRef = (fv.getAttribute('ref') || '').toLowerCase()
    if (filterName.includes('flipped') || filterName.includes('flip') ||
        filterRef.includes('flip')) {
      // Check params for which axis
      const params = fv.querySelectorAll(':scope > param')
      let hasSpecificAxis = false
      params.forEach(p => {
        const pName = (p.getAttribute('name') || '').toLowerCase()
        if (pName.includes('horizontal') || pName.includes('horiz')) {
          result.flipH = true
          hasSpecificAxis = true
        }
        if (pName.includes('vertical') || pName.includes('vert')) {
          result.flipV = true
          hasSpecificAxis = true
        }
      })
      // If no specific axis param found, assume horizontal flip
      if (!hasSpecificAxis) result.flipH = true
    }
  })

  // ── Opacity: check <adjust-blend> ──
  const adjustBlend = el.querySelector(':scope > adjust-blend')
  if (adjustBlend) {
    const amount = adjustBlend.getAttribute('amount')
    if (amount) {
      const opacityVal = parseFloat(amount) * 100 // FCPXML uses 0-1 range
      if (!isNaN(opacityVal)) {
        result.opacity = Math.max(0, Math.min(100, opacityVal))
      }
    }
  }

  return result
}

// ═══════════════════════════════════════════════════════════════════════════
// FCPXML Parser (Final Cut Pro X / DaVinci Resolve)
// ═══════════════════════════════════════════════════════════════════════════

function parseFcpXml(doc: Document): ParsedTimeline | null {
  const fcpxml = doc.querySelector('fcpxml')
  if (!fcpxml) return null
  
  // Find the first project/event/sequence
  const project = fcpxml.querySelector('project') || fcpxml.querySelector('event > project')
  const sequence = project?.querySelector('sequence') || fcpxml.querySelector('sequence')
  
  if (!sequence) return null
  
  const name = sequence.getAttribute('name') || project?.getAttribute('name') || 'Imported Timeline'
  const format = sequence.getAttribute('format') || ''
  
  // Parse duration (FCPXML uses rational time: "86400/2400s" or "10s")
  const parseFcpxmlTime = (timeStr: string | null): number => {
    if (!timeStr) return 0
    timeStr = timeStr.trim()
    // Format: "86400/2400s" (rational)
    const rational = timeStr.match(/^(\d+)\/(\d+)s$/)
    if (rational) return parseInt(rational[1]) / parseInt(rational[2])
    // Format: "10s" (simple seconds)
    const simple = timeStr.match(/^([\d.]+)s$/)
    if (simple) return parseFloat(simple[1])
    return 0
  }
  
  const totalDuration = parseFcpxmlTime(sequence.getAttribute('duration'))
  
  // Get format info (resolution, fps)
  let fps = 24
  let seqWidth: number | undefined
  let seqHeight: number | undefined
  
  if (format) {
    // Format is defined as a resource reference
    const formatEl = fcpxml.querySelector(`resources > format[id="${format}"]`)
    if (formatEl) {
      seqWidth = parseInt(formatEl.getAttribute('width') || '0') || undefined
      seqHeight = parseInt(formatEl.getAttribute('height') || '0') || undefined
      const frameDur = formatEl.getAttribute('frameDuration')
      if (frameDur) {
        const dur = parseFcpxmlTime(frameDur)
        if (dur > 0) fps = 1 / dur
      }
    }
  }
  
  // Collect media references from resources
  const mediaRefs = new Map<string, ParsedMediaRef>()
  
  const assets = fcpxml.querySelectorAll('resources > asset')
  assets.forEach(assetEl => {
    const assetId = assetEl.getAttribute('id') || ''
    if (!assetId) return
    
    const assetName = assetEl.getAttribute('name') || ''
    const src = assetEl.getAttribute('src') || ''
    const dur = parseFcpxmlTime(assetEl.getAttribute('duration'))
    const hasVideo = assetEl.getAttribute('hasVideo') !== '0'
    const hasAudio = assetEl.getAttribute('hasAudio') !== '0'
    
    // Get format from the asset's format ref
    const assetFormat = assetEl.getAttribute('format')
    let width: number | undefined
    let height: number | undefined
    if (assetFormat) {
      const fmtEl = fcpxml.querySelector(`resources > format[id="${assetFormat}"]`)
      if (fmtEl) {
        width = parseInt(fmtEl.getAttribute('width') || '0') || undefined
        height = parseInt(fmtEl.getAttribute('height') || '0') || undefined
      }
    }
    
    const decodedPath = decodePathUrl(src)
    const type = !hasVideo && hasAudio ? 'audio' : detectMediaType(assetName || src)
    
    mediaRefs.set(assetId, {
      id: assetId,
      name: assetName || src.split('/').pop() || 'Unknown',
      path: decodedPath,
      duration: dur,
      type,
      width,
      height,
      fps,
    })
  })
  
  // Parse spine (main timeline) and lanes
  const clips: ParsedClip[] = []
  let videoTrackCount = 0
  let audioTrackCount = 0
  
  // FCPXML has a "spine" which is the main track, with clips that can have "lane" attributes
  const spine = sequence.querySelector(':scope > spine')
  if (spine) {
    let currentOffset = 0
    const spineChildren = spine.children
    
    for (let i = 0; i < spineChildren.length; i++) {
      const el = spineChildren[i]
      const tagName = el.tagName.toLowerCase()
      
      if (tagName === 'asset-clip' || tagName === 'clip' || tagName === 'video' || tagName === 'audio') {
        const ref = el.getAttribute('ref') || ''
        const clipName = el.getAttribute('name') || ''
        const clipDuration = parseFcpxmlTime(el.getAttribute('duration'))
        const clipOffset = parseFcpxmlTime(el.getAttribute('offset')) || currentOffset
        const clipStart = parseFcpxmlTime(el.getAttribute('start'))
        const lane = parseInt(el.getAttribute('lane') || '0')
        
        const trackIdx = Math.max(0, lane)
        videoTrackCount = Math.max(videoTrackCount, trackIdx + 1)
        
        if (ref && mediaRefs.has(ref)) {
          const fx = parseFcpxmlClipEffects(el)
          clips.push({
            name: clipName,
            mediaRefId: ref,
            trackIndex: trackIdx,
            trackType: tagName === 'audio' ? 'audio' : 'video',
            startTime: clipOffset,
            duration: clipDuration,
            sourceIn: clipStart,
            sourceOut: clipStart + clipDuration,
            speed: fx.speed && fx.speed !== 1 ? fx.speed : undefined,
            reversed: fx.speed !== undefined && fx.speed < 0 ? true : undefined,
            flipH: fx.flipH || undefined,
            flipV: fx.flipV || undefined,
            opacity: fx.opacity !== undefined && fx.opacity !== 100 ? fx.opacity : undefined,
            muted: fx.muted || undefined,
          })
        }
        
        // If no lane attribute (lane 0 / spine), advance the offset
        if (lane === 0 || !el.hasAttribute('lane')) {
          currentOffset += clipDuration
        }
        
        // Check for attached clips (connected clips / B-roll)
        const attached = el.querySelectorAll(':scope > asset-clip, :scope > clip, :scope > audio, :scope > video')
        attached.forEach(att => {
          const attRef = att.getAttribute('ref') || ''
          const attName = att.getAttribute('name') || ''
          const attDur = parseFcpxmlTime(att.getAttribute('duration'))
          const attOffset = parseFcpxmlTime(att.getAttribute('offset'))
          const attStart = parseFcpxmlTime(att.getAttribute('start'))
          const attLane = parseInt(att.getAttribute('lane') || '1')
          
          const attTrackIdx = Math.max(0, attLane)
          videoTrackCount = Math.max(videoTrackCount, attTrackIdx + 1)
          
          if (attRef && mediaRefs.has(attRef)) {
            const isAudio = att.tagName.toLowerCase() === 'audio'
            const attFx = parseFcpxmlClipEffects(att)
            clips.push({
              name: attName,
              mediaRefId: attRef,
              trackIndex: attTrackIdx,
              trackType: isAudio ? 'audio' : 'video',
              startTime: clipOffset + attOffset,
              duration: attDur,
              sourceIn: attStart,
              sourceOut: attStart + attDur,
              speed: attFx.speed && attFx.speed !== 1 ? attFx.speed : undefined,
              reversed: attFx.speed !== undefined && attFx.speed < 0 ? true : undefined,
              flipH: attFx.flipH || undefined,
              flipV: attFx.flipV || undefined,
              opacity: attFx.opacity !== undefined && attFx.opacity !== 100 ? attFx.opacity : undefined,
              muted: attFx.muted || undefined,
            })
          }
        })
      } else if (tagName === 'gap') {
        // Empty space
        const gapDuration = parseFcpxmlTime(el.getAttribute('duration'))
        currentOffset += gapDuration
        
        // Check for clips attached to the gap
        const attached = el.querySelectorAll(':scope > asset-clip, :scope > clip, :scope > audio, :scope > video')
        attached.forEach(att => {
          const attRef = att.getAttribute('ref') || ''
          const attName = att.getAttribute('name') || ''
          const attDur = parseFcpxmlTime(att.getAttribute('duration'))
          const attOffset = parseFcpxmlTime(att.getAttribute('offset'))
          const attStart = parseFcpxmlTime(att.getAttribute('start'))
          const attLane = parseInt(att.getAttribute('lane') || '1')
          
          const attTrackIdx = Math.max(0, attLane)
          videoTrackCount = Math.max(videoTrackCount, attTrackIdx + 1)
          
          if (attRef && mediaRefs.has(attRef)) {
            const attFx = parseFcpxmlClipEffects(att)
            clips.push({
              name: attName,
              mediaRefId: attRef,
              trackIndex: attTrackIdx,
              trackType: att.tagName.toLowerCase() === 'audio' ? 'audio' : 'video',
              startTime: (currentOffset - gapDuration) + attOffset,
              duration: attDur,
              sourceIn: attStart,
              sourceOut: attStart + attDur,
              speed: attFx.speed && attFx.speed !== 1 ? attFx.speed : undefined,
              reversed: attFx.speed !== undefined && attFx.speed < 0 ? true : undefined,
              flipH: attFx.flipH || undefined,
              flipV: attFx.flipV || undefined,
              opacity: attFx.opacity !== undefined && attFx.opacity !== 100 ? attFx.opacity : undefined,
              muted: attFx.muted || undefined,
            })
          }
        })
      }
    }
  }
  
  if (videoTrackCount === 0) videoTrackCount = 1
  
  // ── Merge stereo pairs (same logic as FCP7 XML) ──
  // FCPXML can also represent stereo as two separate audio clips from the same
  // source at the same position. Collapse them into a single stereo clip.
  const fcpxAudioClips = clips.filter(c => c.trackType === 'audio')
  const fcpxNonAudioClips = clips.filter(c => c.trackType !== 'audio')
  
  const fcpxAudioGroups = new Map<string, ParsedClip[]>()
  for (const ac of fcpxAudioClips) {
    const key = `${ac.mediaRefId}|${ac.startTime.toFixed(4)}`
    if (!fcpxAudioGroups.has(key)) fcpxAudioGroups.set(key, [])
    fcpxAudioGroups.get(key)!.push(ac)
  }
  
  const fcpxMergedAudio: ParsedClip[] = []
  let fcpxMergedTrackIdx = 0
  const fcpxTrackRemap = new Map<number, number>()
  
  for (const group of fcpxAudioGroups.values()) {
    const representative = group[0]
    const origTrack = representative.trackIndex
    if (!fcpxTrackRemap.has(origTrack)) {
      fcpxTrackRemap.set(origTrack, videoTrackCount + fcpxMergedTrackIdx)
      fcpxMergedTrackIdx++
    }
    representative.trackIndex = fcpxTrackRemap.get(origTrack)!
    fcpxMergedAudio.push(representative)
  }
  
  const fcpxFinalClips = [...fcpxNonAudioClips, ...fcpxMergedAudio]
  const fcpxFinalAudioCount = fcpxMergedTrackIdx || (audioTrackCount > 0 ? 1 : 0)
  
  return {
    name,
    fps,
    duration: totalDuration || Math.max(...fcpxFinalClips.map(c => c.startTime + c.duration), 0),
    width: seqWidth,
    height: seqHeight,
    videoTrackCount,
    audioTrackCount: fcpxFinalAudioCount,
    mediaRefs: Array.from(mediaRefs.values()),
    clips: fcpxFinalClips,
    format: 'fcpxml',
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

export type ImportFormat = 'fcp7xml' | 'fcpxml' | 'aaf' | 'unknown'

export function detectFormat(content: string, filename: string): ImportFormat {
  const ext = filename.split('.').pop()?.toLowerCase() || ''
  
  if (ext === 'aaf') return 'aaf'
  
  // Check XML content
  if (content.includes('<xmeml') || content.includes('<!DOCTYPE xmeml')) return 'fcp7xml'
  if (content.includes('<fcpxml')) return 'fcpxml'
  
  if (ext === 'fcpxml') return 'fcpxml'
  if (ext === 'xml') return 'fcp7xml' // default XML to FCP7
  
  return 'unknown'
}

export function parseTimelineXml(content: string, filename: string): ParsedTimeline | null {
  const format = detectFormat(content, filename)
  
  if (format === 'aaf') {
    throw new Error(
      'AAF files cannot be imported directly. Please export your timeline as FCP 7 XML (.xml) from your editing software:\n\n' +
      '- Premiere Pro: File → Export → Final Cut Pro XML\n' +
      '- DaVinci Resolve: File → Export Timeline → FCP 7 XML\n' +
      '- Avid Media Composer: File → Export → FCP 7 XML'
    )
  }
  
  // Parse XML
  const parser = new DOMParser()
  const doc = parser.parseFromString(content, 'text/xml')
  
  // Check for parse errors
  const parseError = doc.querySelector('parsererror')
  if (parseError) {
    throw new Error('Invalid XML file: ' + (parseError.textContent?.slice(0, 200) || 'Parse error'))
  }
  
  if (format === 'fcpxml') {
    const result = parseFcpXml(doc)
    if (result) return result
  }
  
  if (format === 'fcp7xml' || format === 'unknown') {
    const result = parseFcp7Xml(doc)
    if (result) return result
  }
  
  // Try both parsers
  const fcpxml = parseFcpXml(doc)
  if (fcpxml) return fcpxml
  
  const fcp7 = parseFcp7Xml(doc)
  if (fcp7) return fcp7
  
  throw new Error('Could not parse timeline. The file format is not recognized as FCP 7 XML or FCPXML.')
}

/**
 * Export the current timeline as FCP 7 XML
 */
export function exportFcp7Xml(options: {
  name: string
  fps: number
  width: number
  height: number
  clips: Array<{
    name: string
    filePath: string
    trackIndex: number
    type: 'video' | 'audio' | 'image'
    startTime: number
    duration: number
    trimStart: number
    sourceDuration: number
    width?: number
    height?: number
  }>
}): string {
  const { name, fps, width, height, clips } = options
  
  // Collect unique files
  const files = new Map<string, { id: string; path: string; name: string; duration: number; type: string; width?: number; height?: number }>()
  clips.forEach((clip, i) => {
    if (!files.has(clip.filePath)) {
      files.set(clip.filePath, {
        id: `file-${i + 1}`,
        path: clip.filePath,
        name: clip.name,
        duration: clip.sourceDuration,
        type: clip.type,
        width: clip.width,
        height: clip.height,
      })
    }
  })
  
  const toFrames = (seconds: number) => Math.round(seconds * fps)
  
  // Escape XML special characters
  const escXml = (s: string) => s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;')
  
  // Build file path URL
  const toPathUrl = (p: string) => {
    let normalized = p.replace(/\\/g, '/')
    if (!normalized.startsWith('/')) normalized = '/' + normalized
    return `file://localhost${encodeURI(normalized)}`
  }
  
  // Group clips by video/audio tracks
  const videoClips = clips.filter(c => c.type !== 'audio')
  const audioClips = clips.filter(c => c.type === 'audio')
  
  // Group by track index
  const videoTrackMap = new Map<number, typeof videoClips>()
  videoClips.forEach(c => {
    const arr = videoTrackMap.get(c.trackIndex) || []
    arr.push(c)
    videoTrackMap.set(c.trackIndex, arr)
  })
  
  const audioTrackMap = new Map<number, typeof audioClips>()
  audioClips.forEach(c => {
    const arr = audioTrackMap.get(c.trackIndex) || []
    arr.push(c)
    audioTrackMap.set(c.trackIndex, arr)
  })
  
  const totalDuration = Math.max(...clips.map(c => c.startTime + c.duration), 1)
  
  let xml = `<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="4">\n  <sequence id="sequence-1">\n    <name>${escXml(name)}</name>\n    <duration>${toFrames(totalDuration)}</duration>\n    <rate>\n      <timebase>${Math.round(fps)}</timebase>\n      <ntsc>${Math.abs(fps - Math.round(fps)) > 0.01 ? 'TRUE' : 'FALSE'}</ntsc>\n    </rate>\n    <media>\n      <video>\n        <format>\n          <samplecharacteristics>\n            <width>${width}</width>\n            <height>${height}</height>\n          </samplecharacteristics>\n        </format>\n`
  
  // Video tracks
  for (const [, trackClips] of videoTrackMap) {
    xml += `        <track>\n`
    for (const clip of trackClips) {
      const file = files.get(clip.filePath)!
      const inFrame = toFrames(clip.trimStart)
      const outFrame = toFrames(clip.trimStart + clip.duration)
      xml += `          <clipitem id="clipitem-${Math.random().toString(36).slice(2, 8)}">\n`
      xml += `            <name>${escXml(clip.name)}</name>\n`
      xml += `            <duration>${toFrames(clip.sourceDuration)}</duration>\n`
      xml += `            <rate><timebase>${Math.round(fps)}</timebase></rate>\n`
      xml += `            <start>${toFrames(clip.startTime)}</start>\n`
      xml += `            <end>${toFrames(clip.startTime + clip.duration)}</end>\n`
      xml += `            <in>${inFrame}</in>\n`
      xml += `            <out>${outFrame}</out>\n`
      xml += `            <file id="${file.id}">\n`
      xml += `              <name>${escXml(file.name)}</name>\n`
      xml += `              <pathurl>${escXml(toPathUrl(file.path))}</pathurl>\n`
      xml += `              <duration>${toFrames(file.duration)}</duration>\n`
      xml += `              <rate><timebase>${Math.round(fps)}</timebase></rate>\n`
      if (file.width && file.height) {
        xml += `              <media><video><samplecharacteristics><width>${file.width}</width><height>${file.height}</height></samplecharacteristics></video></media>\n`
      }
      xml += `            </file>\n`
      xml += `          </clipitem>\n`
    }
    xml += `        </track>\n`
  }
  
  xml += `      </video>\n      <audio>\n`
  
  // Audio tracks
  for (const [, trackClips] of audioTrackMap) {
    xml += `        <track>\n`
    for (const clip of trackClips) {
      const file = files.get(clip.filePath)!
      const inFrame = toFrames(clip.trimStart)
      const outFrame = toFrames(clip.trimStart + clip.duration)
      xml += `          <clipitem id="clipitem-${Math.random().toString(36).slice(2, 8)}">\n`
      xml += `            <name>${escXml(clip.name)}</name>\n`
      xml += `            <duration>${toFrames(clip.sourceDuration)}</duration>\n`
      xml += `            <rate><timebase>${Math.round(fps)}</timebase></rate>\n`
      xml += `            <start>${toFrames(clip.startTime)}</start>\n`
      xml += `            <end>${toFrames(clip.startTime + clip.duration)}</end>\n`
      xml += `            <in>${inFrame}</in>\n`
      xml += `            <out>${outFrame}</out>\n`
      xml += `            <file id="${file.id}">\n`
      xml += `              <name>${escXml(file.name)}</name>\n`
      xml += `              <pathurl>${escXml(toPathUrl(file.path))}</pathurl>\n`
      xml += `              <duration>${toFrames(file.duration)}</duration>\n`
      xml += `              <rate><timebase>${Math.round(fps)}</timebase></rate>\n`
      xml += `            </file>\n`
      xml += `          </clipitem>\n`
    }
    xml += `        </track>\n`
  }
  
  xml += `      </audio>\n    </media>\n  </sequence>\n</xmeml>\n`
  
  return xml
}
