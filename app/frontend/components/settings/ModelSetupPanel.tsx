import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/ds/Button/Button'
import { isH3Variant } from '@/lib/h3-models'
import { modelSetupLabels, type ModelSetupKind } from '@/lib/local-model-catalog'
import { useHfAuth } from '@/hooks/use-hf-auth'
import type { ExploreApi } from '@/ltx-io/runtime/ExploreRuntime'
import type { ApiSuccessOf } from '@/lib/api-client'
import { beginManagedModelSetup, useManagedModelSetup, type ManagedModelKind } from '@/lib/managed-model-setup'

import { SettingsFieldRow, SettingsHint, SettingsStack } from './SettingsContent'
import styles from './ModelSetupPanel.module.scss'

type Props = {
  kind: ModelSetupKind
  api: ExploreApi
  onReady?: () => void
  onClose?: () => void
  resumeSelection?: boolean
}

type FileInfo = { name: string; bytes: number; installed: boolean }
type ModelResult<T> = { ok: true; data: T } | { ok: false; error: { message: string } }
type NativeVersion = ApiSuccessOf<'getLtxVersions'>['versions'][number]

function unwrap<T>(result: ModelResult<T>): T {
  if (!result.ok) throw new Error(result.error.message)
  return result.data
}

function gb(bytes: number) {
  return (bytes / 1e9).toLocaleString(undefined, { maximumFractionDigits: 2 })
}

function useSetupState<T>(read: () => Promise<T>, polling: (state: T) => boolean) {
  const [data, setData] = useState<T | null>(null)
  const [readError, setReadError] = useState('')
  const [actionError, setActionError] = useState('')
  const [busy, setBusy] = useState(false)
  const [revision, setRevision] = useState(0)
  const refresh = () => setRevision((value) => value + 1)

  useEffect(() => {
    if (busy) return
    let live = true
    let timer: ReturnType<typeof setTimeout>
    async function load() {
      try {
        const next = await read()
        if (!live) return
        setReadError('')
        setData(next)
        if (polling(next)) timer = setTimeout(load, 2000)
      } catch (error) {
        if (live) setReadError(error instanceof Error ? error.message : String(error))
      }
    }
    void load()
    return () => { live = false; clearTimeout(timer) }
  }, [read, busy, revision, polling])

  async function act(action: () => Promise<void>) {
    setBusy(true)
    setActionError('')
    try {
      await action()
    } catch (error) {
      setActionError(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(false)
      refresh()
    }
  }
  return { data, busy, error: actionError || readError, act, refresh }
}

function Files({ files }: { files: FileInfo[] }) {
  if (!files.length) return null
  return <details className={styles.files}>
    <summary>Model files ({files.length})</summary>
    <ul>{files.map((file) => <li key={file.name}>
      <span>{file.name}</span> <span>{file.installed ? 'Installed' : `${gb(file.bytes)} GB`}</span>
    </li>)}</ul>
  </details>
}

function Progress({ done, total, file }: { done: number; total: number; file?: string | null }) {
  return <div className={styles.progress}>
    <progress aria-label="Model download progress" value={total > 0 ? done : undefined} max={total || 1} />
    <small>{gb(done)} / {gb(total)} GB{file ? ` · ${file}` : ''}</small>
  </div>
}

function HfAccess({ api, repos }: { api: ExploreApi; repos: string[] }) {
  const desktop = !!window.electronAPI
  const { hfAuthStatus, hfAuthPolling, startHuggingFaceLogin } = useHfAuth(desktop, api)
  return <div className={styles.access}>
    <SettingsHint>Sign in to Hugging Face on the desktop and accept the model terms, then retry the download.</SettingsHint>
    <SettingsFieldRow className={styles.actions}>
      {desktop && hfAuthStatus !== 'authenticated' && <Button appearance="neutral" hierarchy="secondary" size="sm"
        label={hfAuthPolling ? 'Waiting for sign-in…' : 'Sign in to Hugging Face'}
        disabled={hfAuthPolling} onClick={() => void startHuggingFaceLogin()} />}
      {repos.map((repo) => <a key={repo} href={`https://huggingface.co/${repo}`} target="_blank" rel="noreferrer"
        onClick={(e) => { if (desktop) { e.preventDefault(); void window.electronAPI.openHuggingFaceRepo({ repoId: repo }) } }}>
        {repo} terms
      </a>)}
    </SettingsFieldRow>
  </div>
}

function preferredVersion(rows: NativeVersion[], kind: ModelSetupKind) {
  const versions = rows.filter((row) => row.model_id.startsWith(kind + '-'))
  return versions.find((row) => row.installed && row.active)
    ?? versions.find((row) => row.installed)
    ?? versions.find((row) => row.model_id === 'ltx-2.3-22b-distilled-1.1')
    ?? versions[0]
}

type NativeStatus = {
  version: NativeVersion
  progress: ApiSuccessOf<'getModelDownloadProgress'> | null
  otherDownload: boolean
  files: FileInfo[]
}

const nativePolling = (value: NativeStatus) => value.otherDownload || value.progress?.status === 'downloading'

function NativeSetup({ kind, api, onReady }: Props) {
  const [terms, setTerms] = useState<string[]>([])
  const follow = useRef<string | null>(null)
  const selectAfterDownload = useRef(false)
  const read = useCallback(async (): Promise<NativeStatus> => {
    const [inventory, current] = await Promise.all([api.getLtxVersions(), api.getActiveDownload()])
    let version = preferredVersion(unwrap(inventory).versions, kind)
    if (!version) throw new Error('This backend does not offer the selected native model.')
    const running = unwrap(current)
    if (running.session_id && running.activate_model_id === version.model_id) follow.current = running.session_id
    const session = follow.current
    const progress = session ? unwrap(await api.getModelDownloadProgress({ sessionId: session })) : null
    // The inventory read can finish before the final download publishes its files.
    if (progress?.status === 'complete') {
      version = preferredVersion(unwrap(await api.getLtxVersions()).versions, kind) ?? version
      follow.current = null
    }
    const files = !version.installed && !running.session_id
      ? unwrap(await api.describeCheckpoints({ cp_ids: version.cps_to_download })).checkpoints
        .map((file) => ({ name: file.name, bytes: file.size_bytes, installed: file.downloaded }))
      : []
    return { version, progress, otherDownload: !!running.session_id && running.session_id !== session, files }
  }, [api, kind])
  const { data, busy, error, act, refresh } = useSetupState(read, nativePolling)
  useEffect(() => {
    if (data?.version.installed && data.version.active && selectAfterDownload.current) {
      selectAfterDownload.current = false
      onReady?.()
    }
  }, [data, onReady])

  async function choose() {
    if (!data) return
    const version = data.version
    if (version.installed) {
      if (!version.active) unwrap(await api.setActiveLtxModel({ model_id: version.model_id }))
      onReady?.()
      return
    }
    const permissions = unwrap(await api.checkModelAccess({ cp_ids: version.cps_to_download })).access
    const gated = Object.keys(permissions).filter((repo) => permissions[repo] !== 'authorized')
    setTerms(gated)
    if (gated.length) return
    const started = unwrap(await api.startModelDownload({
      type: 'download', cp_ids: version.cps_to_download, activate_model_id: version.model_id,
    }))
    if (started.status !== 'started') throw new Error('The model download did not start.')
    follow.current = started.sessionId
    selectAfterDownload.current = true
  }

  const progress = data?.progress
  const installed = data?.version.installed === true
  const working = !installed && progress?.status === 'downloading'
  const missing = data?.files.reduce((n, file) => n + (file.installed ? 0 : file.bytes), 0) ?? 0
  return <SettingsStack>
    <p role="status">{!data ? 'Checking native models…' : installed
      ? data.version.active ? 'Installed and selected. The model loads on the first generation.' : 'Installed. Select this model to use it on the next generation.'
      : working ? 'Downloading the native model bundle…' : data.otherDownload
        ? 'Another native model download is in progress.' : `Native model files required${missing ? ` · ${gb(missing)} GB` : ''}.`}</p>
    {progress?.status === 'downloading' && <Progress done={progress.total_downloaded_bytes} total={progress.expected_total_bytes} file={progress.current_downloading_file} />}
    {(error || progress?.status === 'error') && <p role="alert">{error || (progress?.status === 'error' ? progress.error : '')}</p>}
    {data && <Files files={data.files} />}
    {!!terms.length && <HfAccess api={api} repos={terms} />}
    <SettingsFieldRow className={styles.actions}>
      {data && (!data.version.active || !installed || onReady) && <Button size="sm"
        label={installed ? 'Use this model' : 'Download & select'} disabled={busy || working || (!installed && data.otherDownload)}
        onClick={() => void act(choose)} />}
      <Button appearance="neutral" hierarchy="secondary" size="sm" label="Refresh" disabled={busy}
        onClick={() => { if (progress?.status === 'error') follow.current = null; refresh() }} />
    </SettingsFieldRow>
  </SettingsStack>
}

type ModelBundle = Pick<ApiSuccessOf<'getComfyModels'>, 'status' | 'message' | 'models' | 'completedBytes' | 'totalBytes' | 'currentFile'>
type ComfyStatus = {
  engine: ApiSuccessOf<'getComfyEngine'>; bundle: ModelBundle; weightsReady: boolean
  downloading: boolean; licenseAcknowledged: boolean
  node: ApiSuccessOf<'getKandinskyNode'> | null; nodeError: string
}

const comfyPolling = (value: ComfyStatus) => ['installing', 'starting', 'not_installed', 'stopped'].includes(value.engine.status)
  || value.downloading || value.bundle.status === 'offline'
  || value.node?.status === 'installing' || value.node?.status === 'restarting' || value.node?.status === 'busy'

async function readSetup(api: ExploreApi, kind: ManagedModelKind, checkNode: boolean): Promise<ComfyStatus> {
  const engine = unwrap(await api.getComfyEngine())
  if (isH3Variant(kind)) {
    const h3 = unwrap(await api.getH3Models({ fresh: true }))
    const bundle = h3.variants[kind]
    return { engine, bundle, weightsReady: bundle.status === 'ready', licenseAcknowledged: h3.licenseAcknowledged,
      downloading: Object.values(h3.variants).some(row => row.status === 'downloading'), node: null, nodeError: '' }
  }
  const running = engine.status === 'ready' || engine.status === 'external'
  const [result, node] = await Promise.all([api.getComfyModels({ fresh: true }),
    running && kind === 'kandinsky' && checkNode ? api.getKandinskyNode() : null])
  const models = unwrap(result)
  const k6 = kind === 'kandinsky'
  const bundle = k6 ? { status: models.kandinskyStatus, message: models.kandinskyMessage, models: models.kandinskyModels,
    completedBytes: models.kandinskyCompletedBytes, totalBytes: models.kandinskyTotalBytes, currentFile: models.kandinskyCurrentFile } : models
  return { engine, bundle, weightsReady: k6 ? models.kandinskyWeightsReady : models.status === 'ready', licenseAcknowledged: true,
    downloading: models.status === 'downloading' || models.kandinskyStatus === 'downloading',
    node: node?.ok ? node.data : null, nodeError: node && !node.ok ? node.error.message : '' }
}

function ComfySetup({ kind, api, onReady, resumeSelection }: Props & { kind: ManagedModelKind }) {
  const k6 = kind === 'kandinsky'
  const desktop = !!window.electronAPI
  const selectAfterInstall = useRef(false)
  const usedJob = useRef<string | null>(null)
  const checkedStart = useRef(false)
  const [startError, setStartError] = useState('')
  const setup = useManagedModelSetup(kind)
  const working = !!setup && setup.phase !== 'ready' && setup.phase !== 'error'
  const read = useCallback(() => readSetup(api, kind, desktop && !working), [api, kind, desktop, working, setup?.phase])
  const [confirmedRights, setConfirmedRights] = useState(false)
  const h3 = isH3Variant(kind)
  const { data, error, busy, act, refresh } = useSetupState(read, comfyPolling)
  const engine = data?.engine
  useEffect(() => {
    if (!engine || checkedStart.current || !desktop || !resumeSelection) return
    checkedStart.current = true
    // Opening an installed model can start its private worker. Downloads always
    // require the separate setup action, even when weights turn out to be missing.
    if (!working && setup?.phase !== 'error' && engine.status === 'stopped' && engine.canStart) {
      selectAfterInstall.current = true
      void window.electronAPI.manageComfyEngine({ action: 'start' }).then((result) => {
        if (result.status === 'error') setStartError(result.message)
      }).catch((err) => setStartError(err instanceof Error ? err.message : String(err)))
    }
  }, [desktop, resumeSelection, engine, working, setup?.phase])
  const online = engine?.status === 'ready' || engine?.status === 'external'
  const managed = engine?.status === 'ready'
  const status = data?.bundle.status
  const rightsGranted = !h3 || data?.licenseAcknowledged === true
  const ready = online && rightsGranted && !working && status === 'ready' && !data?.nodeError && (!data?.node || data.node.status === 'ready')
  const message = data?.bundle.message
  const files = data?.bundle.models ?? []
  const missing = files.filter((file) => !file.installed).reduce((n, file) => n + file.sizeBytes, 0)
  const node = data?.node
  const needsWeights = !data?.weightsReady
  useEffect(() => {
    const resumed = resumeSelection && setup?.phase === 'ready' && setup.id !== usedJob.current
    if (ready && (selectAfterInstall.current || resumed)) {
      selectAfterInstall.current = false
      if (setup) usedJob.current = setup.id
      onReady?.()
    }
  }, [ready, onReady, resumeSelection, setup])

  function prepareModel() {
    setStartError('')
    selectAfterInstall.current = true
    beginManagedModelSetup(kind)
  }

  async function download() {
    if (isH3Variant(kind)) {
      const result = unwrap(await api.downloadH3Models({ variant: kind, acceptedLicense: true })).variants[kind]
      if (result.status === 'error') throw new Error(result.message)
    } else unwrap(k6 ? await api.downloadKandinskyModels() : await api.downloadComfyModels())
    selectAfterInstall.current = true
  }
  async function manageNode(action: 'install' | 'restart') {
    const result = await window.electronAPI.manageKandinskyNode({ action })
    if (result.status === 'manual' || result.status === 'busy' || result.status === 'offline') throw new Error(result.message)
    selectAfterInstall.current = true
  }

  return <SettingsStack>
    <p role="status">{engine?.message || 'Checking the local video engine…'}</p>
    {engine?.progress != null && (engine.status === 'installing' || engine.status === 'starting') && <div className={styles.progress}>
      <progress aria-label="Engine setup progress" value={engine.progress} max={100} />
      <small>{Math.floor(engine.progress)}%</small>
    </div>}
    {h3 && data && !rightsGranted && <div className={styles.access}>
      <SettingsHint>MiniMax’s standard H3 license excludes the United States, European Union, United Kingdom and South Korea. Local use in these territories requires separate permission.</SettingsHint>
      <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE" target="_blank" rel="noreferrer"
        onClick={e => { if (desktop) { e.preventDefault(); void window.electronAPI.openExternalUrl({ url: e.currentTarget.href }) } }}>Read the MiniMax H3 license</a>
      <label><input type="checkbox" checked={confirmedRights} onChange={e => setConfirmedRights(e.target.checked)} /> I have the rights to use MiniMax H3 on this computer.</label>
      <SettingsFieldRow className={styles.actions}><Button size="sm" label="Confirm model rights" disabled={busy || !confirmedRights}
        onClick={() => void act(async () => { unwrap(await api.acceptH3License()) })} /></SettingsFieldRow>
    </div>}
    {online && <SettingsHint>{message}</SettingsHint>}
    {working && <p role="status">{setup.message}</p>}
    {ready && <SettingsHint>Ready on this computer. Select this model in Generate to load its weights on the first run.</SettingsHint>}
    {status === 'downloading' && data && <Progress
      done={data.bundle.completedBytes}
      total={data.bundle.totalBytes}
      file={data.bundle.currentFile} />}
    {(error || startError || setup?.phase === 'error') && <p role="alert">{error || startError || setup?.message}</p>}
    {k6 && data?.nodeError && <p role="alert">{data.nodeError}</p>}
    {k6 && node && node.status !== 'ready' && <SettingsHint>{node.message}</SettingsHint>}
    <Files files={files.map((file) => ({ name: file.filename, installed: file.installed, bytes: file.sizeBytes }))} />
    {!!missing && status !== 'downloading' && <SettingsHint>Missing weights · {gb(missing)} GB</SettingsHint>}
    {kind === 'comfy-int8' && /license|\b401\b|\b403\b|hugging face|huggingface/i.test([error, message, setup?.message].join(' ')) &&
      <HfAccess api={api} repos={['Lightricks/LTX-2.5']} />}
    {!online && data && engine?.status !== 'unsupported' && <SettingsHint>
      LTX Desktop can install its own video engine on this computer. A complete model download is up to{' '}
      {gb(data.bundle.totalBytes)} GB, plus engine files.
    </SettingsHint>}
    <SettingsFieldRow className={styles.actions}>
      {desktop && engine && !working && !ready && engine.status !== 'unsupported' && (
        engine.canPrepare || engine.canStart || managed || engine.status === 'installing' || engine.status === 'starting'
      ) && engine.status !== 'external' && <Button size="sm"
        label={setup?.phase === 'error' ? 'Retry model setup' : engine.canPrepare && !engine.canStart ? 'Install engine & model' : 'Set up model'}
        disabled={busy || !rightsGranted} onClick={prepareModel} />}
      {desktop && engine?.status === 'external' && (engine.canPrepare || engine.canStart) && !working && <Button size="sm"
        appearance="neutral" hierarchy="secondary" label="Use LTX-managed engine" disabled={busy}
        onClick={() => void act(async () => { await window.electronAPI.manageComfyEngine({ action: engine.canStart ? 'start' : 'prepare' }) })} />}
      {data && engine?.status === 'external' && status !== 'downloading' && needsWeights && <Button size="sm"
        label={h3 ? 'Download H3 weights' : k6 ? 'Download Kandinsky weights' : 'Download INT8 weights'} disabled={busy || !rightsGranted} onClick={() => void act(download)} />}
      {desktop && !working && k6 && node?.canInstall && <Button size="sm" label="Install Kandinsky nodes" disabled={busy}
        onClick={() => void act(() => manageNode('install'))} />}
      {desktop && !working && k6 && node?.canRestart && <Button size="sm" label="Restart ComfyUI" disabled={busy || data?.downloading}
        onClick={() => void act(() => manageNode('restart'))} />}
      {ready && onReady && <Button size="sm" label="Use this model" disabled={busy} onClick={onReady} />}
      <Button appearance="neutral" hierarchy="secondary" size="sm" label="Refresh" disabled={busy} onClick={refresh} />
    </SettingsFieldRow>
    {desktop && engine?.canStop && !working && <details className={styles.files}>
      <summary>Engine controls</summary>
      <Button appearance="neutral" hierarchy="secondary" size="sm" label="Stop managed engine"
        disabled={busy || data?.downloading}
        onClick={() => void act(async () => {
          const result = await window.electronAPI.manageComfyEngine({ action: 'stop' })
          if (result.status === 'error') throw new Error(result.message)
        })} />
    </details>}
    {desktop && window.electronAPI.platform === 'win32' && engine && engine.status !== 'unsupported' && !online && !working && <details className={styles.files}>
      <summary>Use an existing ComfyUI installation</summary>
      <Button appearance="neutral" hierarchy="secondary" size="sm" label="Open existing ComfyUI Desktop"
        onClick={() => void act(async () => { await window.electronAPI.openComfyDesktop() })} />
    </details>}
    {k6 && !ready && engine?.status === 'external' && <a href="https://github.com/kandinskylab/kandinsky-6/blob/73b892b730885e02ab58fbf7fc5cb10b78e1e3da/comfyui/README.md#manual-downloads"
      target="_blank" rel="noreferrer" onClick={(event) => {
        if (desktop) { event.preventDefault(); void window.electronAPI.openExternalUrl({ url: event.currentTarget.href }) }
      }}>Kandinsky model and node setup guide</a>}
  </SettingsStack>
}

export function ModelSetupPanel(props: Props) {
  return <section className={styles.panel} aria-label={`Set up ${modelSetupLabels[props.kind]}`}>
    <div className={styles.header}>
      <h4>{modelSetupLabels[props.kind]}</h4>
      {props.onClose && <Button appearance="neutral" hierarchy="plain" size="sm" label="Close" onClick={props.onClose} />}
    </div>
    {props.kind === 'ltx-2.3' || props.kind === 'ltx-2.5'
      ? <NativeSetup key={props.kind} {...props} /> : <ComfySetup {...props} key={props.kind} kind={props.kind} />}
  </section>
}
