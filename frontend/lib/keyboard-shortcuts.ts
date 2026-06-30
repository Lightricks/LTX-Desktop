// Keyboard Shortcuts system — action registry, keybinding types, preset layouts

// ── Action IDs ──
// Every bindable action in the editor
export type ActionId =
  // Tools
  | 'tool.select'
  | 'tool.blade'
  | 'tool.ripple'
  | 'tool.roll'
  | 'tool.slide'
  | 'tool.slip'
  | 'tool.trackForward'
  // Transport
  | 'transport.playPause'
  | 'transport.stop'
  | 'transport.shuttleReverse'   // J
  | 'transport.shuttleStop'      // K
  | 'transport.shuttleForward'   // L
  | 'transport.stepBackward'     // Left arrow
  | 'transport.stepForward'      // Right arrow
  | 'transport.jumpBackward'     // Shift+Left
  | 'transport.jumpForward'      // Shift+Right
  | 'transport.goToStart'
  | 'transport.goToEnd'
  | 'transport.goToIn'
  | 'transport.goToOut'
  // Editing
  | 'edit.undo'
  | 'edit.redo'
  | 'edit.cut'
  | 'edit.copy'
  | 'edit.paste'
  | 'edit.delete'
  | 'edit.selectAll'
  | 'edit.deselect'
  // Marking
  | 'mark.setIn'
  | 'mark.setOut'
  | 'mark.clearIn'
  | 'mark.clearOut'
  | 'mark.clearInOut'
  // 3-Point Editing
  | 'edit.insertEdit'
  | 'edit.overwriteEdit'
  | 'edit.matchFrame'
  // Timeline
  | 'timeline.zoomIn'
  | 'timeline.zoomOut'
  | 'timeline.fitToView'
  | 'timeline.toggleSnap'
  // Navigation
  | 'nav.prevEdit'
  | 'nav.nextEdit'
  // View
  | 'view.fullscreen'

// ── Key Combo ──
export interface KeyCombo {
  key: string        // e.g. 'b', ' ', 'arrowleft', 'delete', ','
  ctrl?: boolean
  shift?: boolean
  alt?: boolean
  meta?: boolean     // Cmd on Mac
}

// ── Action Definition ──
export interface ActionDefinition {
  id: ActionId
  label: string
  category: 'Tools' | 'Transport' | 'Editing' | 'Marking' | 'Timeline'
  description?: string
}

// ── Full Registry ──
export const ACTION_REGISTRY: ActionDefinition[] = [
  // Tools
  { id: 'tool.select',       label: 'Selection Tool',        category: 'Tools' },
  { id: 'tool.blade',        label: 'Blade / Razor Tool',    category: 'Tools' },
  { id: 'tool.ripple',       label: 'Ripple Edit Tool',      category: 'Tools' },
  { id: 'tool.roll',         label: 'Roll Edit Tool',        category: 'Tools' },
  { id: 'tool.slide',        label: 'Slide Tool',            category: 'Tools' },
  { id: 'tool.slip',         label: 'Slip Tool',             category: 'Tools' },
  { id: 'tool.trackForward', label: 'Track Select Forward',  category: 'Tools' },
  // Transport
  { id: 'transport.playPause',       label: 'Play / Pause',          category: 'Transport' },
  { id: 'transport.stop',            label: 'Stop',                  category: 'Transport' },
  { id: 'transport.shuttleReverse',  label: 'Shuttle Reverse (J)',   category: 'Transport' },
  { id: 'transport.shuttleStop',     label: 'Shuttle Stop (K)',      category: 'Transport' },
  { id: 'transport.shuttleForward',  label: 'Shuttle Forward (L)',   category: 'Transport' },
  { id: 'transport.stepBackward',    label: 'Step Backward (1 frame)', category: 'Transport' },
  { id: 'transport.stepForward',     label: 'Step Forward (1 frame)',  category: 'Transport' },
  { id: 'transport.jumpBackward',    label: 'Jump Backward (1 sec)',   category: 'Transport' },
  { id: 'transport.jumpForward',     label: 'Jump Forward (1 sec)',    category: 'Transport' },
  { id: 'transport.goToStart',       label: 'Go to Start',          category: 'Transport' },
  { id: 'transport.goToEnd',         label: 'Go to End',            category: 'Transport' },
  { id: 'transport.goToIn',          label: 'Go to In Point',       category: 'Transport' },
  { id: 'transport.goToOut',         label: 'Go to Out Point',      category: 'Transport' },
  // Editing
  { id: 'edit.undo',          label: 'Undo',               category: 'Editing' },
  { id: 'edit.redo',          label: 'Redo',               category: 'Editing' },
  { id: 'edit.cut',           label: 'Cut',                category: 'Editing' },
  { id: 'edit.copy',          label: 'Copy',               category: 'Editing' },
  { id: 'edit.paste',         label: 'Paste',              category: 'Editing' },
  { id: 'edit.delete',        label: 'Delete',             category: 'Editing' },
  { id: 'edit.selectAll',     label: 'Select All',         category: 'Editing' },
  { id: 'edit.deselect',      label: 'Deselect All',       category: 'Editing' },
  { id: 'edit.insertEdit',    label: 'Insert Edit',        category: 'Editing' },
  { id: 'edit.overwriteEdit', label: 'Overwrite Edit',     category: 'Editing' },
  { id: 'edit.matchFrame',    label: 'Match Frame',        category: 'Editing', description: 'Load the clip under the playhead into the source monitor at the matching frame' },
  // Marking
  { id: 'mark.setIn',       label: 'Set In Point',         category: 'Marking' },
  { id: 'mark.setOut',      label: 'Set Out Point',        category: 'Marking' },
  { id: 'mark.clearIn',     label: 'Clear In Point',       category: 'Marking' },
  { id: 'mark.clearOut',    label: 'Clear Out Point',      category: 'Marking' },
  { id: 'mark.clearInOut',  label: 'Clear In / Out',       category: 'Marking' },
  // Timeline
  { id: 'timeline.zoomIn',    label: 'Zoom In',            category: 'Timeline' },
  { id: 'timeline.zoomOut',   label: 'Zoom Out',           category: 'Timeline' },
  { id: 'timeline.fitToView', label: 'Fit Timeline to View', category: 'Timeline' },
  { id: 'timeline.toggleSnap', label: 'Toggle Snap',       category: 'Timeline' },
  // Navigation
  { id: 'nav.prevEdit',        label: 'Go to Previous Edit Point', category: 'Transport', description: 'Jump playhead to previous cut on timeline' },
  { id: 'nav.nextEdit',        label: 'Go to Next Edit Point',     category: 'Transport', description: 'Jump playhead to next cut on timeline' },
  { id: 'view.fullscreen',     label: 'Fullscreen Preview', category: 'Timeline' },
]

// ── Keyboard Layout (mapping from ActionId to one or more key combos) ──
export type KeyboardLayout = Partial<Record<ActionId, KeyCombo[]>>

// ── Named Preset ──
export interface KeyboardPreset {
  id: string
  name: string
  description: string
  layout: KeyboardLayout
  builtIn: boolean  // true for factory presets, false for user-created
}

// Helper to create key combos concisely
function k(key: string, mods?: { ctrl?: boolean; shift?: boolean; alt?: boolean; meta?: boolean }): KeyCombo {
  return { key, ...mods }
}

// ═══════════════════════════════════════════
// ── PRESET: LTX Default ──
// ═══════════════════════════════════════════
export const LTX_DEFAULT_LAYOUT: KeyboardLayout = {
  // Tools
  'tool.select':       [k('v')],
  'tool.blade':        [k('b')],
  'tool.ripple':       [k('r')],
  'tool.roll':         [k('n')],
  'tool.slide':        [k('u')],
  'tool.slip':         [k('y')],
  'tool.trackForward': [k('a')],
  // Transport
  'transport.playPause':      [k(' ')],
  'transport.shuttleReverse': [k('j')],
  'transport.shuttleStop':    [k('k')],
  'transport.shuttleForward': [k('l')],
  'transport.stepBackward':   [k('arrowleft')],
  'transport.stepForward':    [k('arrowright')],
  'transport.jumpBackward':   [k('arrowleft', { shift: true })],
  'transport.jumpForward':    [k('arrowright', { shift: true })],
  'transport.goToStart':      [k('home')],
  'transport.goToEnd':        [k('end')],
  'transport.goToIn':         [k('i', { shift: true })],
  'transport.goToOut':        [k('o', { shift: true })],
  // Editing
  'edit.undo':          [k('z', { ctrl: true })],
  'edit.redo':          [k('z', { ctrl: true, shift: true }), k('y', { ctrl: true })],
  'edit.cut':           [k('x', { ctrl: true })],
  'edit.copy':          [k('c', { ctrl: true })],
  'edit.paste':         [k('v', { ctrl: true })],
  'edit.delete':        [k('delete'), k('backspace')],
  'edit.selectAll':     [k('a', { ctrl: true })],
  'edit.deselect':      [k('escape')],
  'edit.insertEdit':    [k(',')],
  'edit.overwriteEdit': [k('.')],
  'edit.matchFrame':    [k('f')],
  // Marking
  'mark.setIn':       [k('i')],
  'mark.setOut':       [k('o')],
  'mark.clearIn':      [k('i', { alt: true })],
  'mark.clearOut':     [k('o', { alt: true })],
  'mark.clearInOut':   [k('x', { alt: true })],
  // Timeline
  'timeline.zoomIn':    [k('='), k('+')],
  'timeline.zoomOut':   [k('-')],
  'timeline.fitToView': [k('0', { ctrl: true })],
  'timeline.toggleSnap': [k('s')],
  // Navigation
  'nav.prevEdit': [k('arrowup')],
  'nav.nextEdit': [k('arrowdown')],
  'view.fullscreen': [k('`'), k('f11')],
}

// ═══════════════════════════════════════════
// ── PRESET: Adobe Premiere Pro ──
// ═══════════════════════════════════════════
export const PREMIERE_LAYOUT: KeyboardLayout = {
  // Tools (Premiere defaults)
  'tool.select':       [k('v')],
  'tool.blade':        [k('c')],         // Premiere uses C for razor
  'tool.ripple':       [k('b')],         // Premiere: B = ripple edit
  'tool.roll':         [k('n')],
  'tool.slide':        [k('u')],
  'tool.slip':         [k('y')],
  'tool.trackForward': [k('a')],
  // Transport (same JKL)
  'transport.playPause':      [k(' ')],
  'transport.shuttleReverse': [k('j')],
  'transport.shuttleStop':    [k('k')],
  'transport.shuttleForward': [k('l')],
  'transport.stepBackward':   [k('arrowleft')],
  'transport.stepForward':    [k('arrowright')],
  'transport.jumpBackward':   [k('arrowleft', { shift: true })],
  'transport.jumpForward':    [k('arrowright', { shift: true })],
  'transport.goToStart':      [k('home')],
  'transport.goToEnd':        [k('end')],
  'transport.goToIn':         [k('i', { shift: true })],
  'transport.goToOut':        [k('o', { shift: true })],
  // Editing
  'edit.undo':          [k('z', { ctrl: true })],
  'edit.redo':          [k('z', { ctrl: true, shift: true })],
  'edit.cut':           [k('x', { ctrl: true })],
  'edit.copy':          [k('c', { ctrl: true })],
  'edit.paste':         [k('v', { ctrl: true })],
  'edit.delete':        [k('delete'), k('backspace')],
  'edit.selectAll':     [k('a', { ctrl: true })],
  'edit.deselect':      [k('escape')],
  'edit.insertEdit':    [k(',')],         // Premiere: , = insert
  'edit.overwriteEdit': [k('.')],         // Premiere: . = overwrite
  'edit.matchFrame':    [k('f')],         // Premiere: F = match frame
  // Marking (same as Premiere)
  'mark.setIn':       [k('i')],
  'mark.setOut':       [k('o')],
  'mark.clearIn':      [k('i', { alt: true })],   // Premiere: Alt+I / Option+I
  'mark.clearOut':     [k('o', { alt: true })],   // Premiere: Alt+O / Option+O
  'mark.clearInOut':   [k('x', { alt: true })],   // Premiere: Alt+X or Option+X
  // Timeline
  'timeline.zoomIn':    [k('=')],
  'timeline.zoomOut':   [k('-')],
  'timeline.fitToView': [k('\\')],       // Premiere: backslash fits timeline
  'timeline.toggleSnap': [k('s')],       // Premiere: S = snap
  // Navigation (Premiere: Up/Down = previous/next edit)
  'nav.prevEdit': [k('arrowup')],
  'nav.nextEdit': [k('arrowdown')],
  'view.fullscreen': [k('`')],           // Premiere: ` = fullscreen
}

// ═══════════════════════════════════════════
// ── PRESET: DaVinci Resolve ──
// ═══════════════════════════════════════════
export const DAVINCI_LAYOUT: KeyboardLayout = {
  // Tools
  'tool.select':       [k('a')],         // DaVinci: A = selection
  'tool.blade':        [k('b')],         // DaVinci: B = blade
  'tool.ripple':       [k('t')],         // DaVinci: T = trim (contextual ripple/roll)
  'tool.roll':         [k('g')],         // DaVinci has no separate roll key; G assigned to avoid conflict
  'tool.slide':        [k('u')],
  'tool.slip':         [k('s')],         // DaVinci: S = slip (not snap)
  'tool.trackForward': [k('y')],
  // Transport
  'transport.playPause':      [k(' ')],
  'transport.shuttleReverse': [k('j')],
  'transport.shuttleStop':    [k('k')],
  'transport.shuttleForward': [k('l')],
  'transport.stepBackward':   [k('arrowleft')],
  'transport.stepForward':    [k('arrowright')],
  'transport.jumpBackward':   [k('arrowleft', { shift: true })],
  'transport.jumpForward':    [k('arrowright', { shift: true })],
  'transport.goToStart':      [k('home')],
  'transport.goToEnd':        [k('end')],
  'transport.goToIn':         [k('i', { shift: true })],
  'transport.goToOut':        [k('o', { shift: true })],
  // Editing
  'edit.undo':          [k('z', { ctrl: true })],
  'edit.redo':          [k('z', { ctrl: true, shift: true })],
  'edit.cut':           [k('x', { ctrl: true })],
  'edit.copy':          [k('c', { ctrl: true })],
  'edit.paste':         [k('v', { ctrl: true })],
  'edit.delete':        [k('delete'), k('backspace')],
  'edit.selectAll':     [k('a', { ctrl: true })],
  'edit.deselect':      [k('escape')],
  'edit.insertEdit':    [k('f9')],       // DaVinci: F9 = insert
  'edit.overwriteEdit': [k('f10')],      // DaVinci: F10 = overwrite
  'edit.matchFrame':    [k('f')],        // DaVinci: F = match frame
  // Marking
  'mark.setIn':       [k('i')],
  'mark.setOut':       [k('o')],
  'mark.clearIn':      [k('i', { alt: true })],
  'mark.clearOut':     [k('o', { alt: true })],
  'mark.clearInOut':   [k('x', { alt: true })],
  // Timeline
  'timeline.zoomIn':    [k('=', { ctrl: true })],
  'timeline.zoomOut':   [k('-', { ctrl: true })],
  'timeline.fitToView': [k('z', { shift: true })],  // DaVinci: Shift+Z = fit
  'timeline.toggleSnap': [k('n')],                   // DaVinci: N = snap
  // Navigation (DaVinci: Up/Down = previous/next edit)
  'nav.prevEdit': [k('arrowup')],
  'nav.nextEdit': [k('arrowdown')],
  'view.fullscreen': [k('p', { ctrl: true, shift: true })], // DaVinci: Ctrl+Shift+P
}

// ═══════════════════════════════════════════
// ── PRESET: Avid Media Composer ──
// ═══════════════════════════════════════════
export const AVID_LAYOUT: KeyboardLayout = {
  // Tools
  'tool.select':       [k('v')],
  'tool.blade':        [k('/')],         // Avid uses different paradigm but closest
  'tool.ripple':       [k('r')],
  'tool.roll':         [k('n')],
  'tool.slide':        [k('u')],
  'tool.slip':         [k('y')],
  'tool.trackForward': [k('a')],
  // Transport — Avid uses different keys
  'transport.playPause':      [k(' '), k('5')],  // Avid: 5 or Space
  'transport.shuttleReverse': [k('j')],
  'transport.shuttleStop':    [k('k')],
  'transport.shuttleForward': [k('l')],
  'transport.stepBackward':   [k('3')],           // Avid: 3 = step back
  'transport.stepForward':    [k('4')],           // Avid: 4 = step forward
  'transport.jumpBackward':   [k('1')],           // Avid: 1 = fast reverse 
  'transport.jumpForward':    [k('2')],           // Avid: 2 = fast forward
  'transport.goToStart':      [k('home')],
  'transport.goToEnd':        [k('end')],
  'transport.goToIn':         [k('i', { shift: true })],
  'transport.goToOut':        [k('o', { shift: true })],
  // Editing
  'edit.undo':          [k('z', { ctrl: true })],
  'edit.redo':          [k('z', { ctrl: true, shift: true })],
  'edit.cut':           [k('x', { ctrl: true })],
  'edit.copy':          [k('c', { ctrl: true })],
  'edit.paste':         [k('v', { ctrl: true })],
  'edit.delete':        [k('delete'), k('backspace')],
  'edit.selectAll':     [k('a', { ctrl: true })],
  'edit.deselect':      [k('escape')],
  'edit.insertEdit':    [k('v', { shift: true })],  // Avid: Shift+V to avoid conflict with tool.select
  'edit.overwriteEdit': [k('b')],         // Avid: B = overwrite
  'edit.matchFrame':    [k('f')],         // Avid: match frame
  // Marking — Avid classic: I/O or E/R
  'mark.setIn':       [k('i'), k('e')],   // Avid: E = mark in
  'mark.setOut':       [k('o')],           // Avid: O = mark out (R removed to avoid conflict with tool.ripple)
  'mark.clearIn':      [k('i', { alt: true })],
  'mark.clearOut':     [k('o', { alt: true })],
  'mark.clearInOut':   [k('g')],           // Avid: G = clear both marks
  // Timeline
  'timeline.zoomIn':    [k('=', { ctrl: true })],
  'timeline.zoomOut':   [k('-', { ctrl: true })],
  'timeline.fitToView': [k('0', { ctrl: true })],
  'timeline.toggleSnap': [k('s')],
  // Navigation (Avid: similar to A/S or arrow keys)
  'nav.prevEdit': [k('arrowup')],
  'nav.nextEdit': [k('arrowdown')],
  'view.fullscreen': [k('`'), k('f11')],
}

// ── All Built-in Presets ──
export const BUILT_IN_PRESETS: KeyboardPreset[] = [
  {
    id: 'ltx-default',
    name: 'LTX Default',
    description: 'Default keyboard layout for LTX Desktop',
    layout: LTX_DEFAULT_LAYOUT,
    builtIn: true,
  },
  {
    id: 'premiere',
    name: 'Adobe Premiere Pro',
    description: 'Keyboard layout matching Premiere Pro defaults',
    layout: PREMIERE_LAYOUT,
    builtIn: true,
  },
  {
    id: 'davinci',
    name: 'DaVinci Resolve',
    description: 'Keyboard layout matching DaVinci Resolve defaults',
    layout: DAVINCI_LAYOUT,
    builtIn: true,
  },
  {
    id: 'avid',
    name: 'Avid Media Composer',
    description: 'Keyboard layout matching Avid Media Composer defaults',
    layout: AVID_LAYOUT,
    builtIn: true,
  },
]

// ── Utilities ──

/** Format a KeyCombo into a human-readable string like "Ctrl+Shift+B" */
export function formatKeyCombo(combo: KeyCombo): string {
  const parts: string[] = []
  if (combo.ctrl || combo.meta) parts.push('Ctrl')
  if (combo.shift) parts.push('Shift')
  if (combo.alt) parts.push('Alt')

  // Pretty-print special keys
  const keyMap: Record<string, string> = {
    ' ': 'Space',
    'arrowleft': '\u2190',
    'arrowright': '\u2192',
    'arrowup': '\u2191',
    'arrowdown': '\u2193',
    'delete': 'Del',
    'backspace': 'Bksp',
    'escape': 'Esc',
    'enter': 'Enter',
    ',': ',',
    '.': '.',
    '/': '/',
    '\\': '\\',
    '=': '=',
    '+': '+',
    '-': '-',
    'home': 'Home',
    'end': 'End',
    'f9': 'F9',
    'f10': 'F10',
  }
  parts.push(keyMap[combo.key] || combo.key.toUpperCase())
  return parts.join('+')
}

/** Map a physical KeyboardEvent.code to a logical key string matching KeyCombo.key */
function codeToKey(code: string): string {
  if (code.startsWith('Key')) return code.slice(3).toLowerCase()
  if (code.startsWith('Digit')) return code.slice(5)
  return code.toLowerCase()
}

/** Check if a keyboard event matches a key combo */
export function eventMatchesCombo(e: KeyboardEvent, combo: KeyCombo): boolean {
  // On Mac, Option (alt) remaps e.key to special characters (e.g. Option+I → 'Dead', Option+O → 'ø').
  // Use the physical key code instead when alt is involved so shortcuts still resolve correctly.
  const key = (combo.alt && e.altKey) ? codeToKey(e.code) : e.key.toLowerCase()
  if (key !== combo.key) return false
  if (!!combo.ctrl !== (e.ctrlKey || e.metaKey)) return false
  if (!!combo.shift !== e.shiftKey) return false
  if (!!combo.alt !== e.altKey) return false
  return true
}

/** Given a layout and an event, return the matching ActionId (or null) */
export function resolveAction(layout: KeyboardLayout, e: KeyboardEvent): ActionId | null {
  for (const [actionId, combos] of Object.entries(layout)) {
    if (!combos) continue
    for (const combo of combos) {
      if (eventMatchesCombo(e, combo)) {
        return actionId as ActionId
      }
    }
  }
  return null
}

/** Find conflicts: actions that share the same key combo */
export function findConflicts(layout: KeyboardLayout): Map<string, ActionId[]> {
  const comboMap = new Map<string, ActionId[]>()
  for (const [actionId, combos] of Object.entries(layout)) {
    if (!combos) continue
    for (const combo of combos) {
      const key = formatKeyCombo(combo)
      const existing = comboMap.get(key) || []
      existing.push(actionId as ActionId)
      comboMap.set(key, existing)
    }
  }
  // Only return entries with conflicts (more than 1 action)
  const conflicts = new Map<string, ActionId[]>()
  for (const [key, actions] of comboMap) {
    if (actions.length > 1) conflicts.set(key, actions)
  }
  return conflicts
}

/** Deep clone a layout for mutation */
export function cloneLayout(layout: KeyboardLayout): KeyboardLayout {
  const result: KeyboardLayout = {}
  for (const [key, combos] of Object.entries(layout)) {
    if (combos) {
      result[key as ActionId] = combos.map(c => ({ ...c }))
    }
  }
  return result
}

// ── I18n ──
// Maps each ACTION_REGISTRY English label to its i18n key.
// Components that render action labels should use `translateActionLabel()` below.

export const ACTION_I18N_MAP: Record<string, string> = {
  'Selection Tool': 'editor.selectionTool',
  'Blade / Razor Tool': 'editor.bladeTool',
  'Ripple Edit Tool': 'editor.rippleTool',
  'Roll Edit Tool': 'editor.rollTool',
  'Slide Tool': 'editor.slideTool',
  'Slip Tool': 'editor.slipTool',
  'Track Select Forward': 'editor.trackSelect',
  'Play / Pause': 'editor.playPause',
  'Stop': 'editor.stop',
  'Shuttle Reverse (J)': 'editor.shuttleReverse',
  'Shuttle Stop (K)': 'editor.shuttleStop',
  'Shuttle Forward (L)': 'editor.shuttleForward',
  'Step Backward (1 frame)': 'editor.stepBackward',
  'Step Forward (1 frame)': 'editor.stepForward',
  'Jump Backward (1 sec)': 'editor.jumpBackward',
  'Jump Forward (1 sec)': 'editor.jumpForward',
  'Go to Start': 'editor.goToStart',
  'Go to End': 'editor.goToEnd',
  'Go to In Point': 'editor.goToIn',
  'Go to Out Point': 'editor.goToOut',
  'Undo': 'editor.undo',
  'Redo': 'editor.redo',
  'Cut': 'editor.cut',
  'Copy': 'editor.copy',
  'Paste': 'editor.paste',
  'Delete': 'editor.delete',
  'Select All': 'editor.selectAll',
  'Deselect All': 'editor.deselectAll',
  'Insert Edit': 'editor.insertEdit',
  'Overwrite Edit': 'editor.overwriteEdit',
  'Match Frame': 'editor.matchFrame',
  'Set In Point': 'editor.setIn',
  'Set Out Point': 'editor.setOut',
  'Clear In Point': 'editor.clearIn',
  'Clear Out Point': 'editor.clearOut',
  'Clear In / Out': 'editor.clearInOut',
  'Zoom In': 'editor.zoomIn',
  'Zoom Out': 'editor.zoomOut',
  'Fit Timeline to View': 'editor.fitTimeline',
  'Toggle Snap': 'editor.toggleSnap',
  'Go to Previous Edit Point': 'editor.prevEdit',
  'Go to Next Edit Point': 'editor.nextEdit',
  'Fullscreen Preview': 'editor.fullscreen',
}

export const ACTION_DESC_I18N_MAP: Record<string, string> = {
  'Load the clip under the playhead into the source monitor at the matching frame': 'editor.matchFrameDesc',
  'Jump playhead to previous cut on timeline': 'editor.prevEditDesc',
  'Jump playhead to next cut on timeline': 'editor.nextEditDesc',
}

/** Translate an action label from English to the current locale using the given `t` function.
 *  Falls back to the English label if no i18n key is mapped. */
export function translateActionLabel(label: string, t: (key: string) => string): string {
  const i18nKey = ACTION_I18N_MAP[label]
  if (i18nKey) {
    const translated = t(i18nKey)
    // Only return translation if it actually differs (i.e. the key was found);
    // otherwise stick with the English default.
    if (translated && translated !== i18nKey) return translated
  }
  return label
}

/** Translate an action description from English to the current locale using the given `t` function.
 *  Falls back to the English description if no i18n key is mapped. */
export function translateActionDescription(description: string, t: (key: string) => string): string {
  const i18nKey = ACTION_DESC_I18N_MAP[description]
  if (i18nKey) {
    const translated = t(i18nKey)
    if (translated && translated !== i18nKey) return translated
  }
  return description
}
