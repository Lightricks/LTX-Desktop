import type { Project, Timeline } from '../../types/project-model'
import type { EditorModel } from './editor-state'
import { DEFAULT_TRACKS } from '../../types/project-model'
import { migrateClip, migrateTracks } from './video-editor-utils'

function normalizeTimeline(timeline: Timeline): Timeline {
  const sourceTracks = timeline.tracks.length > 0
    ? timeline.tracks
    : DEFAULT_TRACKS.map(track => ({ ...track }))

  return {
    ...timeline,
    tracks: migrateTracks(sourceTracks),
    clips: timeline.clips.map(migrateClip),
    subtitles: timeline.subtitles || [],
  }
}

export function getEditorModel(project: Project): EditorModel {
  const timelines = project.timelines.map(normalizeTimeline)
  return {
    assets: project.assets,
    bins: project.bins,
    timelines,
    activeTimelineId: project.activeTimelineId ?? timelines[0]?.id ?? null,
  }
}

export function updatedProject(fromProject: Project, editorModel: EditorModel): Project {
  return {
    ...fromProject,
    assets: editorModel.assets,
    bins: editorModel.bins,
    timelines: editorModel.timelines,
    activeTimelineId: editorModel.activeTimelineId ?? editorModel.timelines[0]?.id,
    updatedAt: Date.now(),
  }
}
