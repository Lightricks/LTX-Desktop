import { getAssetsDir } from '../app-state'
import { cancelAssetsMove, changeAssetsDir, getAssetsMoveState } from '../assets-move'
import { handle } from './typed-handle'

export function registerAssetsHandlers(): void {
  handle('getAssetsDir', () => getAssetsDir())
  handle('changeAssetsDir', () => changeAssetsDir('pick'))
  handle('resetAssetsDir', () => changeAssetsDir('default'))
  handle('cancelAssetsMove', () => cancelAssetsMove())
  handle('getAssetsMoveState', () => getAssetsMoveState())
}
