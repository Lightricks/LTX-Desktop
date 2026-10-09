import type { UpdateInfo } from 'electron-updater'

import { releaseNotesText } from '../shared/release-notes-text'

function rawNotes(info: UpdateInfo): string | undefined {
  const notes = info.releaseNotes
  if (typeof notes === 'string' && notes.length > 0) return notes
  if (Array.isArray(notes)) {
    const parts = notes
      .map((entry) => (typeof entry.note === 'string' ? entry.note : ''))
      .filter((note) => note.length > 0)
    return parts.length > 0 ? parts.join('\n\n') : undefined
  }
  return undefined
}

export function releaseNotesFromFeed(info: UpdateInfo): string | undefined {
  const raw = rawNotes(info)
  if (!raw) return undefined
  return releaseNotesText(raw)
}
