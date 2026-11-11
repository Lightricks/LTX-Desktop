import { app } from 'electron'
import { getAssetsDir, getDefaultAssetsDir, isCustomAssetsDir, setAssetsDir } from './app-state'
import { checkCustomAssetsDir } from './assets-relocation'
import { logger } from './logger'
import { pickFolder, showMessage } from './native-dialogs'

const RETRY = 0
const CHOOSE = 1
const USE_DEFAULT = 2
const QUIT = 3

async function askWhatToDo(dir: string, reason: string): Promise<number> {
  return showMessage({
    type: 'warning',
    title: 'Assets folder not available',
    message: 'The assets folder is not available.',
    detail:
      `${dir}\n\n${reason}\n\n` +
      'Your assets are not deleted. Connect the drive or restore the folder, then choose Retry. ' +
      'Choose "Choose folder" if you moved the folder. ' +
      'Choose "Use default folder" to start with the default folder. Your assets in the old folder stay there.',
    buttons: ['Retry', 'Choose folder…', 'Use default folder', 'Quit'],
    defaultId: RETRY,
    cancelId: QUIT,
    noLink: true,
  })
}

async function confirmUseDefault(): Promise<boolean> {
  const response = await showMessage({
    type: 'warning',
    message: 'Use the default folder?',
    detail:
      'Your assets in the other folder will not show in the app, and the app marks them as deleted. ' +
      'The files stay in the other folder, but the app cannot show them again. ' +
      'Choose "Back" if you can still connect or find the folder.',
    buttons: ['Use default folder', 'Back'],
    defaultId: 1,
    cancelId: 1,
    noLink: true,
  })
  return response === 0
}

/** A relink does not copy anything: the chosen folder must already hold the library. */
async function chooseMovedFolder(): Promise<void> {
  const picked = await pickFolder('Choose the assets folder')
  if (!picked) return
  const check = await checkCustomAssetsDir(picked)
  if (!check.ok) {
    await showMessage({
      type: 'warning',
      message: `This folder cannot be used. ${check.reason}`,
      buttons: ['OK'],
      noLink: true,
    })
    return
  }
  setAssetsDir(picked)
}

/**
 * Runs before every backend start. A saved custom folder that is missing or has no marker
 * must never fall back to the default folder without the user's choice: the backend would
 * start with an empty library. The only caller runs inside `startPromise`, so two checks never overlap.
 */
export async function ensureAssetsDirReady({ askUser = true }: { askUser?: boolean } = {}): Promise<void> {
  // The default folder needs no check: the backend creates it.
  while (isCustomAssetsDir()) {
    const dir = getAssetsDir()
    const check = await checkCustomAssetsDir(dir)
    if (check.ok) return
    logger.warn(`Assets folder unavailable: ${dir} (${check.reason})`)
    if (!askUser) throw new Error(`The assets folder is not available. ${check.reason}`)

    const choice = await askWhatToDo(dir, check.reason)
    if (choice === QUIT) {
      app.quit()
      throw new Error('The assets folder is not available.')
    }
    if (choice === USE_DEFAULT) {
      // The default folder exists, so the backend marks every asset that is not in it as deleted.
      if (!(await confirmUseDefault())) continue
      setAssetsDir(undefined)
      logger.info(`Assets folder reset to default ${getDefaultAssetsDir()}`)
      return
    }
    if (choice === CHOOSE) await chooseMovedFolder()
    // RETRY, or a folder that failed the check: the loop checks again.
  }
}
