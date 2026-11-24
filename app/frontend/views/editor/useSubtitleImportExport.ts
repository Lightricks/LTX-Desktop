import { useCallback, useRef } from 'react'
import { selectActiveTimeline, selectSubtitles } from './editor-selectors'
import { parseSrt, exportSrt } from '../../lib/srt'
import { useEditorActions, useEditorStore } from './editor-store'

export function useSubtitleImportExport() {
  const { importSrtCues } = useEditorActions()
  const subtitles = useEditorStore(selectSubtitles)
  const activeTimelineName = useEditorStore(state => selectActiveTimeline(state)?.name || 'timeline')
  const subtitleFileInputRef = useRef<HTMLInputElement>(null)

  const handleImportSrt = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = event => {
      const content = event.target?.result as string
      if (!content) return
      const cues = parseSrt(content)
      importSrtCues(cues)
    }
    reader.readAsText(file)

    e.target.value = ''
    if (subtitleFileInputRef.current && subtitleFileInputRef.current !== e.target) {
      subtitleFileInputRef.current.value = ''
    }
  }, [importSrtCues])

  const handleExportSrt = useCallback(() => {
    const cues = subtitles
      .filter(subtitle => subtitle.text.trim())
      .sort((a, b) => a.startTime - b.startTime)

    if (cues.length === 0) {
      alert('No subtitles to export')
      return
    }

    const srtContent = exportSrt(cues)
    const timelineName = activeTimelineName

    if (window.electronAPI?.showSaveDialog) {
      window.electronAPI.showSaveDialog({
        title: 'Export Subtitles',
        defaultPath: `subtitles_${timelineName}.srt`,
        filters: [{ name: 'SRT Files', extensions: ['srt'] }],
      }).then(filePath => {
        if (filePath) {
          window.electronAPI!.saveFile({ filePath, data: srtContent })
        }
      })
      return
    }

    const blob = new Blob([srtContent], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `subtitles_${timelineName}.srt`
    anchor.click()
    URL.revokeObjectURL(url)
  }, [activeTimelineName, subtitles])

  return {
    subtitleFileInputRef,
    handleImportSrt,
    handleExportSrt,
  }
}
