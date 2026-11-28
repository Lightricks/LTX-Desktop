import assert from 'node:assert/strict'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { describe, it } from 'node:test'
import { fileURLToPath } from 'node:url'

const frontendRoot = join(dirname(fileURLToPath(import.meta.url)), '..')

const ZINC_SCAN_ROOTS = [
  'views/Project.tsx',
  'views/Projects.tsx',
  'views/GenSpace.tsx',
  'views/VideoEditor.tsx',
  'views/genspace',
  'views/editor',
  'components/SettingsDropdown.tsx',
  'components/IcLoraSettingsControls.tsx',
  'components/IcLoraAdvancedPanel.tsx',
  'components/ICLoraPanel.tsx',
  'components/VideoPreviewPanel.tsx',
  'components/MultiKeyframePanel.tsx',
  'components/KeyframePreview.tsx',
  'components/KeyframeTimeline.tsx',
  'components/KeyframeStrengthRail.tsx',
  'components/SelectedLoraInfo.tsx',
  'components/RetakePanel.tsx',
  'components/ExtendPanel.tsx',
  'components/OutpaintCanvasEditor.tsx',
  'components/SettingsPanel.tsx',
  'components/SettingsModal.tsx',
  'components/KeyboardShortcutsModal.tsx',
  'components/settings',
  'components/LtxApiKeyInput.tsx',
  'components/RemoteKeepAwakeNote.tsx',
  'components/MenuBar.tsx',
  'components/LoraLibraryModal.tsx',
  'components/LoraInfoPopover.tsx',
  'components/library',
]

const UTILITY = '(?:bg|text|border|divide|ring|from|to|via|outline|fill|stroke|placeholder|accent|shadow)'

const FORBIDDEN = new RegExp(
  [
    `${UTILITY}-zinc-`,
    'text-white\\b',
    'ring-offset-zinc-',
    // Opacity suffixes on a raw var() silently do nothing — go through the
    // Tailwind aliases, which emit color-mix.
    '\\[var\\(--semantic-[a-z-]+\\)\\]/\\d',
    // Truncated opacity utilities, e.g. `bg-brand/` with no digits.
    `${UTILITY}-[A-Za-z0-9-]+\\/(?![\\d])`,
  ].join('|'),
)

// Status and accent hues are theme-blind: a 400-level tint that reads on a
// dark panel washes out on white. Blue is exempt — the Tailwind config remaps
// that whole scale onto the brand colour.
const FORBIDDEN_HUES = new RegExp(
  `${UTILITY}-(?:red|rose|amber|yellow|orange|green|emerald|cyan|teal|violet|purple|indigo|pink|lime|fuchsia|sky)-\\d`,
)

// Clip colour labels are content, not theme: the user picks "Violet" by name.
const HUE_EXEMPT = ['views/editor/video-editor-utils.ts']

function walk(rel: string): string[] {
  const abs = join(frontendRoot, rel)
  const stat = statSync(abs)
  if (stat.isDirectory()) {
    return readdirSync(abs).flatMap((name) => walk(join(rel, name)))
  }
  return abs.endsWith('.tsx') || abs.endsWith('.ts') ? [rel] : []
}

function source(rel: string): string {
  return readFileSync(join(frontendRoot, rel), 'utf8')
}

describe('legacy project design-system conversion', () => {
  it('uses design-system type on the project name, Gen Space empty state, and project field', () => {
    const project = source('views/Project.tsx')
    assert.match(project, /variant="label"/)
    assert.match(project, /size="xl"/)
    assert.match(project, /<Text[\s\S]*activeProject\.name/)

    const genspace = source('views/GenSpace.tsx')
    assert.match(genspace, /Start Creating/)
    assert.match(genspace, /variant="heading"/)
    assert.match(genspace, /size="lg"/)
    assert.match(genspace, /variant="body"/)
    assert.match(genspace, /size="xl"/)
    assert.equal(genspace.includes('text-xl font-semibold'), false)
    assert.match(genspace, /bg-brand text-fg-white hover:bg-brand-hover/)
    assert.doesNotMatch(genspace, /bg-white text-black/)

    const projects = source('views/Projects.tsx')
    assert.match(projects, /<TextField/)
    assert.doesNotMatch(projects, /<input[\s\S]*placeholder="Project name"/)
  })

  it('uses one app theme instead of forcing Home light or Project dark', () => {
    const app = source('App.tsx')
    const rootTheme = app.match(/<ThemeProvider defaultPreference="system"[^>]*>/)
    assert.ok(rootTheme, 'root ThemeProvider should default to system')
    assert.equal(rootTheme[0].includes('scheme='), false)

    const layout = source('components/home/DesktopHomeLayout.tsx')
    assert.equal(layout.includes('scheme="light"'), false)
    assert.equal(layout.includes('ThemeProvider'), false)
    assert.equal(layout.includes('homeFonts'), false)

    const router = source('router.tsx')
    assert.equal(router.includes('ThemeProvider'), false)
    assert.equal(router.includes('scheme="dark"'), false)

    const setup = source('components/FirstRunSetup.tsx')
    assert.equal(setup.includes('data-theme="light"'), false)

    const layoutStyles = source('components/home/DesktopHomeLayout.module.scss')
    assert.doesNotMatch(layoutStyles, /--app-font-family:\s*"Inter"/)

    const dsFonts = source('ds/styles/fonts.scss')
    assert.match(dsFonts, /--app-font-family:\s*"Aeonik"/)
  })

  it('keeps theme-blind colour and truncated opacity utilities out of Gen Space and the project chrome', () => {
    const hits: string[] = []
    for (const rel of ZINC_SCAN_ROOTS.flatMap(walk)) {
      const text = source(rel)
      if (FORBIDDEN.test(text)) hits.push(rel)
      if (!HUE_EXEMPT.includes(rel) && FORBIDDEN_HUES.test(text)) hits.push(rel)
    }
    assert.deepEqual(hits, [])
  })

  it('keeps project chrome at 16px and timeline ticks off the 40px Text root', () => {
    const projectChrome = source('views/Project.module.scss')
    const dsRoot = source('ds/styles/index.css')
    assert.match(projectChrome, /font-size:\s*16px/)
    assert.match(dsRoot, /\.ltx-io[\s\S]*font-size:\s*40px/)

    const timeline = source('views/editor/VideoEditorTimelineEditingPanel.tsx')
    assert.match(timeline, /semantic-bg-separator-secondary/)
    assert.doesNotMatch(timeline, /<Text[^>]*formatTime/)

    const dropdown = source('components/SettingsDropdown.tsx')
    assert.match(dropdown, /useThemedPortalContainer/)
    assert.match(dropdown, /portalRoot \?\? document\.body/)
  })
})
