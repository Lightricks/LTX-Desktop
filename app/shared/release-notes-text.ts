import { convert } from 'html-to-text'

const MAX_RELEASE_NOTES_CHARS = 16_384

const htmlToText = (html: string): string => convert(html, {
  wordwrap: false,
  selectors: [
    { selector: 'a', options: { ignoreHref: true } },
  ],
})

function looksLikeHtml(value: string): boolean {
  return /<\/?[a-z][\s\S]*>/i.test(value)
}

function cap(value: string): string {
  if (value.length <= MAX_RELEASE_NOTES_CHARS) return value
  return `${value.slice(0, MAX_RELEASE_NOTES_CHARS)}\n…`
}

/** GitHub's feed is HTML. The modal renders notes as text, so convert first. */
export function releaseNotesText(raw: string): string | undefined {
  const bounded = cap(raw)
  const text = (looksLikeHtml(bounded) ? htmlToText(bounded) : bounded).trim()
  if (!text) return undefined
  return cap(text)
}
