import type { MenuItemConstructorOptions } from 'electron'

export const OPEN_ABOUT_CHANNEL = 'menu-open-about'

export function windowsMenuTemplate(showAbout: () => void): MenuItemConstructorOptions[] {
  return [
    { role: 'fileMenu' },
    { role: 'editMenu' },
    { role: 'viewMenu' },
    { role: 'windowMenu' },
    { role: 'help', submenu: [{ label: 'About', click: showAbout }] },
  ]
}

/** Listen from preload: the Windows menu can be clicked before React mounts. */
export function listenForAboutMenu(on: (channel: string, receive: () => void) => void) {
  let pending = false
  let listener: (() => void) | undefined
  on(OPEN_ABOUT_CHANNEL, () => {
    if (listener) listener()
    else pending = true
  })
  return (showAbout: () => void) => {
    listener = showAbout
    if (pending) {
      pending = false
      showAbout()
    }
    return () => { if (listener === showAbout) listener = undefined }
  }
}
