import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { generatePath, matchPath } from 'react-router'
import { paths } from './paths.ts'

describe('paths', () => {
  it('defines the Home and project routes', () => {
    assert.equal(paths.home, '/')
    assert.equal(paths.textToVideo, '/text-to-video')
    assert.equal(paths.imageToVideo, '/image-to-video')
    assert.equal(paths.audioToVideo, '/audio-to-video')
    assert.equal(paths.retake, '/retake')
    assert.equal(paths.extend, '/extend')
    assert.equal(paths.dayToNight, '/day-to-night')
    assert.equal(paths.alphaGen, '/alpha-gen')
    assert.equal(paths.assets, '/assets')
    assert.equal(paths.dashboard, '/activity-dashboard')
    assert.equal(paths.fetcherTool, '/fetcher/:tool')
    assert.equal(paths.remote, '/remote')
    assert.equal(paths.projects, '/projects')
    assert.equal(paths.project, '/projects/:projectId')
    assert.equal(
      generatePath(paths.textToVideo),
      '/text-to-video',
    )
    assert.equal(
      generatePath(paths.imageToVideo),
      '/image-to-video',
    )
    assert.equal(
      generatePath(paths.audioToVideo),
      '/audio-to-video',
    )
    assert.equal(
      generatePath(paths.retake),
      '/retake',
    )
    assert.equal(
      generatePath(paths.extend),
      '/extend',
    )
    assert.equal(
      generatePath(paths.fetcherTool, { tool: 't2v' }),
      '/fetcher/t2v',
    )
    assert.equal(
      generatePath(paths.project, { projectId: 'project-1' }),
      '/projects/project-1',
    )
  })

  it('matches the route table', () => {
    assert.ok(matchPath(paths.home, '/'))
    assert.ok(matchPath(paths.textToVideo, '/text-to-video'))
    assert.ok(matchPath(paths.imageToVideo, '/image-to-video'))
    assert.ok(matchPath(paths.audioToVideo, '/audio-to-video'))
    assert.ok(matchPath(paths.retake, '/retake'))
    assert.ok(matchPath(paths.extend, '/extend'))
    assert.ok(matchPath(paths.dayToNight, '/day-to-night'))
    assert.ok(matchPath(paths.alphaGen, '/alpha-gen'))
    assert.ok(matchPath(paths.assets, '/assets'))
    assert.ok(matchPath(paths.dashboard, '/activity-dashboard'))
    assert.equal(matchPath(paths.fetcherTool, '/fetcher/t2v')?.params.tool, 't2v')
    assert.ok(matchPath(paths.remote, '/remote'))
    assert.ok(matchPath(paths.projects, '/projects'))
    assert.equal(
      matchPath(paths.project, '/projects/abc')?.params.projectId,
      'abc',
    )
    assert.equal(matchPath(paths.project, '/projects'), null)
  })

  it('does not expose a generic feature or explore route', () => {
    assert.equal('feature' in paths, false)
    assert.equal(
      Object.values(paths).some((path) => path.includes('explore')),
      false,
    )
    assert.equal(matchPath(paths.textToVideo, '/not-a-tool'), null)
    assert.equal(matchPath(paths.fetcherTool, '/fetcher'), null)
  })
})
