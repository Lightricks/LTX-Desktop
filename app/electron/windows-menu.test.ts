import assert from 'node:assert/strict'
import { test } from 'node:test'
import { OPEN_ABOUT_CHANNEL, listenForAboutMenu, windowsMenuTemplate } from './windows-menu.ts'

test('Help → About survives an early click and continues to work when the settings shell remounts', () => {
  const handlers = new Map<string, () => void>()
  const register = listenForAboutMenu((channel, receive) => handlers.set(channel, receive))
  const menu = windowsMenuTemplate(() => handlers.get(OPEN_ABOUT_CHANNEL)!())
  assert.deepEqual(menu.map(entry => entry.role), ['fileMenu', 'editMenu', 'viewMenu', 'windowMenu', 'help'])
  const help = menu.find(entry => entry.role === 'help')!
  assert.ok(Array.isArray(help.submenu))
  const about = help.submenu.find(entry => entry.label === 'About')
  assert.ok(about?.click)
  // Electron supplies extra positional arguments; this menu action uses none.
  const click = about.click as () => void
  click()
  click()
  const opened: string[] = []
  const stop = register(() => opened.push('first'))
  assert.deepEqual(opened, ['first'])
  click()
  stop()
  click()
  assert.deepEqual(opened, ['first', 'first'])
  register(() => opened.push('remounted'))
  assert.deepEqual(opened, ['first', 'first', 'remounted'])
})
