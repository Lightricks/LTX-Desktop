import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { imagePathsFromDataTransfer, imagePathsFromFiles } from './keyframe-drop.ts'

describe('imagePathsFromFiles', () => {
  it('keeps image files in selection order and skips non-images', () => {
    const files = [
      new File(['a'], 'a.png', { type: 'image/png' }),
      new File(['b'], 'notes.txt', { type: 'text/plain' }),
      new File(['c'], 'c.jpg', { type: 'image/jpeg' }),
    ]

    assert.deepEqual(
      imagePathsFromFiles(files, (file) => `/picked/${file.name}`),
      ['/picked/a.png', '/picked/c.jpg'],
    )
  })
})

describe('imagePathsFromDataTransfer', () => {
  it('reads a gallery image asset', () => {
    const dataTransfer = {
      getData: (type: string) => (
        type === 'asset'
          ? JSON.stringify({ type: 'image', path: '/gallery/still.png' })
          : ''
      ),
      files: [] as unknown as FileList,
    } as DataTransfer

    assert.deepEqual(imagePathsFromDataTransfer(dataTransfer), ['/gallery/still.png'])
  })

  it('ignores non-image gallery assets', () => {
    const dataTransfer = {
      getData: (type: string) => (
        type === 'asset'
          ? JSON.stringify({ type: 'video', path: '/gallery/clip.mp4' })
          : ''
      ),
      files: [] as unknown as FileList,
    } as DataTransfer

    assert.deepEqual(imagePathsFromDataTransfer(dataTransfer), [])
  })

  it('collects dropped OS image files when no gallery asset is present', () => {
    const files = [
      new File(['a'], 'one.png', { type: 'image/png' }),
      new File(['b'], 'two.png', { type: 'image/png' }),
    ]
    const dataTransfer = {
      getData: () => '',
      files,
    } as unknown as DataTransfer

    assert.deepEqual(
      imagePathsFromDataTransfer(dataTransfer, (file) => `/os/${file.name}`),
      ['/os/one.png', '/os/two.png'],
    )
  })
})
