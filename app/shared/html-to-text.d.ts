declare module 'html-to-text' {
  export function convert(
    html: string,
    options?: {
      wordwrap?: number | false
      selectors?: Array<{
        selector: string
        options?: { ignoreHref?: boolean }
      }>
    },
  ): string
}
