import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { filterByLevels, LOG_LEVELS, parseLogEntries, type LogLevel } from './logEntries.ts'

const TS = '2026-10-07 18:39:43,374'
const backend = (message: string, label = 'INFO') => `${TS} - ${label} - [Backend] ${message}`
const levels = (lines: string[]) => parseLogEntries(lines).map((entry) => entry.level)

describe('parseLogEntries: Python prefixes on backend lines', () => {
  it('uses the Python level instead of the INFO label Electron gives stdout lines', () => {
    assert.deepEqual(
      levels([
        backend('INFO:__main__:ready'),
        backend('WARNING:runtime_config:two checkpoints found'),
        backend('ERROR:handlers.lora:download failed'),
        backend('CRITICAL:__main__:out of memory'),
        backend('DEBUG:httpx:request'),
      ]),
      ['info', 'warning', 'error', 'error', 'debug'],
    )
  })

  it('keeps the file level for lines that are not from the backend', () => {
    assert.deepEqual(
      levels([
        `${TS} - WARNING - [Electron] Backend liveness probe failed (1/5)`,
        `${TS} - INFO - [Electron] WARNING: not a Python record`,
      ]),
      ['warning', 'info'],
    )
  })

  it('keeps the ERROR label of stderr lines without a Python prefix', () => {
    assert.deepEqual(levels([backend('`Siglip2ImageProcessorFast` is deprecated.', 'ERROR')]), ['error'])
  })
})

describe('parseLogEntries: level markers inside the message', () => {
  it('keeps the header level when the message contains another level marker', () => {
    assert.deepEqual(
      levels([
        `${TS} - INFO - [Electron] relayed: 2026-10-07 18:00:00,000 - ERROR - [Backend] boom`,
        `${TS} - DEBUG - [Electron] saw " - WARNING - " in a payload`,
      ]),
      ['info', 'debug'],
    )
  })

  it('does not read a level from a continuation line', () => {
    assert.deepEqual(levels(['plain text - ERROR - more text']), [null])
  })
})

describe('parseLogEntries: tracebacks', () => {
  const traceback = [
    backend('Traceback (most recent call last):'),
    backend('  File "/app/runner.py", line 24, in _run'),
    backend('    target()'),
    backend('    ~~~~~~^^'),
    backend('    ...<3 lines>...'),
    backend('ModuleNotFoundError: No module named "x"'),
  ]

  it('shows the whole traceback block as error, including the exception line', () => {
    assert.deepEqual(levels(traceback), Array(traceback.length).fill('error'))
  })

  it('ends the block at the exception line', () => {
    assert.deepEqual(
      levels([...traceback, backend('Application startup complete.')]).at(-1),
      'info',
    )
  })

  it('shows the chain connector lines of a chained exception as error', () => {
    assert.deepEqual(
      levels([
        backend('httpx.HTTPStatusError: Client error 403'),
        backend('The above exception was the direct cause of the following exception:'),
        backend('Traceback (most recent call last):'),
        backend('  File "/app/a.py", line 1, in f'),
        backend('GatedRepoError: 403'),
      ]).slice(1),
      ['error', 'error', 'error', 'error'],
    )
  })

  it('shows "During handling" connector lines as error', () => {
    assert.deepEqual(
      levels([backend('During handling of the above exception, another exception occurred:')]),
      ['error'],
    )
  })

  it('lets a Python prefixed record end the block', () => {
    assert.deepEqual(
      levels([backend('Traceback (most recent call last):'), backend('INFO:__main__:next')]),
      ['error', 'info'],
    )
  })
})

describe('level filter', () => {
  const lines = [
    `${TS} - ERROR - [Backend] boom`,
    'continuation of the error',
    `${TS} - INFO - [Electron] fine`,
    'continuation of the info',
  ]
  const entries = parseLogEntries(lines)
  const only = (...ids: LogLevel[]) => filterByLevels(entries, new Set(ids)).map((e) => e.text)

  it('keeps continuation lines with the line above', () => {
    assert.deepEqual(only('error'), [lines[0], lines[1]])
  })

  it('hides everything when no level is enabled', () => {
    assert.deepEqual(only(), [])
  })

  it('hides nothing when every level is enabled', () => {
    assert.deepEqual(
      filterByLevels(entries, new Set(LOG_LEVELS.map((level) => level.id))),
      entries,
    )
  })

  it('shows a traceback when only Error is enabled', () => {
    const parsed = parseLogEntries([
      backend('INFO:__main__:start'),
      backend('Traceback (most recent call last):'),
      backend('  File "/app/a.py", line 1, in f'),
      backend('ValueError: bad'),
      backend('INFO:__main__:end'),
    ])
    assert.equal(filterByLevels(parsed, new Set<LogLevel>(['error'])).length, 3)
  })
})
