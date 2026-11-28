import { dialog } from 'electron'
import path from 'path'
import { getMainWindow } from './window'

/** Native message box. It is modal to the main window when one exists. Returns the button index. */
export async function showMessage(options: Electron.MessageBoxOptions): Promise<number> {
  const win = getMainWindow()
  const result = win ? await dialog.showMessageBox(win, options) : await dialog.showMessageBox(options)
  return result.response
}

/** Native folder picker. Returns the resolved path, or `null` when the user cancels. */
export async function pickFolder(title: string, { canCreate = false } = {}): Promise<string | null> {
  const win = getMainWindow()
  const options: Electron.OpenDialogOptions = {
    title,
    properties: canCreate ? ['openDirectory', 'createDirectory'] : ['openDirectory'],
  }
  const result = win ? await dialog.showOpenDialog(win, options) : await dialog.showOpenDialog(options)
  return result.canceled || !result.filePaths[0] ? null : path.resolve(result.filePaths[0])
}
