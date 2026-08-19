import { convert } from 'html-to-text'
import type { UpdateInfo } from 'electron-updater'

const MAX_RELEASE_NOTES_CHARS = 16_384

const htmlToText = (html: string): string => convert(html, {
  wordwrap: false,
  selectors: [
    { selector: 'a', options: { ignoreHref: true } },
  ],
})

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

function looksLikeHtml(value: string): boolean {
  return /<\/?[a-z][\s\S]*>/i.test(value)
}

function cap(value: string): string {
  if (value.length <= MAX_RELEASE_NOTES_CHARS) return value
  return `${value.slice(0, MAX_RELEASE_NOTES_CHARS)}\n…`
}

/** GitHub's feed is HTML. The modal renders notes as text, so convert first. */
export function releaseNotesFromFeed(info: UpdateInfo): string | undefined {
  const raw = rawNotes(info)
  if (!raw) return undefined
  const bounded = cap(raw)
  const text = (looksLikeHtml(bounded) ? htmlToText(bounded) : bounded).trim()
  if (!text) return undefined
  return cap(text)
}
