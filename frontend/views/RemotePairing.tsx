import { Copy, ExternalLink } from 'lucide-react'
import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from 'react'
import QRCode from 'react-qr-code'
import { Button } from '@/ds/Button/Button'
import { Skeleton } from '@/ds/Skeleton/Skeleton'
import { Text } from '@/ds/Text/Text'
import { Tooltip, TooltipProvider } from '@ds/Tooltip/Tooltip'
import CloseIcon from '@ds/assets/Icons/Close/Normal.svg?react'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { copyTextToClipboard } from '../lib/copy-to-clipboard'
import { useRemoteExposure } from '../hooks/use-remote-exposure'
import { useRemoteDevices } from '../hooks/use-remote-devices'
import { isRemoteDeviceInUse } from '../lib/remote-device-activity'
import { pairedDeviceDisplayName } from '../lib/paired-device-name'
import { remoteExposureFromKey } from '../lib/remote-exposure-from-key'
import { pairingErrorMessage } from '../lib/remote-pairing-screen'
import { useRemoteStatus, type RemoteStatus } from '../hooks/use-remote-status'
import styles from './RemotePairing.module.scss'

const PAIRING_QR_SIZE = 168
/** The visible QR/URL can go stale (grant or LAN change), so keep it fresher than the app-wide keep-awake poll. */
const PAIRING_STATUS_SETTLED_POLL_MS = 5000

/** LAN pairing URL with one-time grant (`?t=`) served on port 41955 — same string for QR and copy. */
function pairingUrlFromStatus(status: RemoteStatus | null): string | null {
  if (!status?.serving) {
    return null
  }
  const raw = status.lanUrl ?? status.url
  if (!raw) {
    return null
  }
  try {
    const parsed = new URL(raw)
    const grant = parsed.searchParams.get('t')
    if (!grant || grant.length === 0) {
      return null
    }
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return null
    }
    return raw
  } catch {
    return null
  }
}

function PairingScreenSkeleton() {
  return (
    <div className={styles.pairingSkeleton} aria-hidden="true">
      <Skeleton
        width={PAIRING_QR_SIZE}
        height={PAIRING_QR_SIZE}
        borderRadius="var(--radius-md)"
        className={styles.pairingSkeletonBlock}
      />
      <Skeleton
        height="2rem"
        borderRadius="var(--radius-md)"
        className={`${styles.urlSkeleton} ${styles.pairingSkeletonBlock}`}
      />
    </div>
  )
}

const COPY_TOAST_MS = 2400
const URL_CHIP_ICON_SIZE = 20

function CopyLinkToast({ open }: { open: boolean }) {
  if (!open) {
    return null
  }
  return (
    <div className={styles.copyToast} role="status" aria-live="polite">
      <Text as="p" variant="body" size="xs" className={styles.copyToastText}>
        Link copied
      </Text>
    </div>
  )
}

function CopyableUrl({ value }: { value: string }) {
  const [toastOpen, setToastOpen] = useState(false)
  const [toastKey, setToastKey] = useState(0)
  const inputId = useId()
  const canOpenInBrowser = typeof window.electronAPI?.openExternalUrl === 'function'

  const showCopyToast = useCallback(() => {
    setToastKey((key) => key + 1)
    setToastOpen(true)
  }, [])

  const handleCopy = useCallback(() => {
    void copyTextToClipboard(value).then((copied) => {
      if (copied) {
        showCopyToast()
      }
    })
  }, [showCopyToast, value])

  useEffect(() => {
    if (!toastOpen) {
      return
    }
    const timeoutId = window.setTimeout(() => {
      setToastOpen(false)
    }, COPY_TOAST_MS)
    return () => {
      window.clearTimeout(timeoutId)
    }
  }, [toastOpen, toastKey])

  return (
    <div className={styles.urlChip}>
      {canOpenInBrowser ? (
        <Button
          appearance="white"
          hierarchy="secondary"
          size="md"
          isIconOnly
          className={styles.urlChipButton}
          leftIcon={<ExternalLink size={URL_CHIP_ICON_SIZE} aria-hidden />}
          aria-label="Open pairing link in browser"
          onClick={() => {
            void window.electronAPI.openExternalUrl({ url: value })
          }}
        />
      ) : null}
      <div className={styles.copyButtonWrap}>
        <CopyLinkToast key={toastKey} open={toastOpen} />
        <Button
          appearance="white"
          hierarchy="secondary"
          size="md"
          isIconOnly
          className={styles.urlChipButton}
          leftIcon={<Copy size={URL_CHIP_ICON_SIZE} aria-hidden />}
          aria-label="Copy pairing link"
          onClick={handleCopy}
        />
      </div>
      <input
        id={inputId}
        readOnly
        value={value}
        title={value}
        className={styles.urlText}
        onFocus={(event) => {
          event.currentTarget.select()
        }}
        onClick={(event) => {
          event.currentTarget.select()
        }}
      />
    </div>
  )
}

function ExposureToggle({
  on,
  toggle,
}: {
  on: boolean
  toggle: () => void
}) {
  const offRef = useRef<HTMLButtonElement>(null)
  const onRef = useRef<HTMLButtonElement>(null)

  const select = (next: boolean) => {
    if (next !== on) {
      toggle()
    }
    const target = next ? onRef : offRef
    target.current?.focus()
  }

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const next = remoteExposureFromKey(on, event.key)
    if (next == null) {
      return
    }
    event.preventDefault()
    select(next)
  }

  return (
    <div
      className={styles.toggleBar}
      role="radiogroup"
      aria-label="Remote"
      aria-orientation="horizontal"
      onKeyDown={onKeyDown}
    >
      <span
        className={`${styles.toggleThumb} ${on ? styles.toggleThumbOn : ''}`}
        aria-hidden="true"
      />
      <button
        ref={offRef}
        type="button"
        role="radio"
        aria-checked={!on}
        tabIndex={on ? -1 : 0}
        className={on ? styles.toggleOption : `${styles.toggleOption} ${styles.toggleOptionActive}`}
        onClick={() => {
          select(false)
        }}
      >
        <span className={styles.toggleLabel}>
          <span className={styles.toggleLabelGhost} aria-hidden="true">
            Off
          </span>
          <span className={styles.toggleLabelText}>Off</span>
        </span>
      </button>
      <button
        ref={onRef}
        type="button"
        role="radio"
        aria-checked={on}
        tabIndex={on ? 0 : -1}
        className={on ? `${styles.toggleOption} ${styles.toggleOptionActive}` : styles.toggleOption}
        onClick={() => {
          select(true)
        }}
      >
        <span className={styles.toggleLabel}>
          <span className={styles.toggleLabelGhost} aria-hidden="true">
            Off
          </span>
          <span className={styles.toggleLabelText}>On</span>
        </span>
      </button>
    </div>
  )
}

function DeviceStage({
  open,
  children,
}: {
  open: boolean
  children: ReactNode
}) {
  return (
    <div className={`${styles.stage} ${open ? styles.open : ''}`}>
      <div
        className={styles.laptop}
        aria-hidden={!open}
        // `inert` is missing from React 18's types, so spread it to dodge excess-prop checking.
        {...(!open ? { inert: '' } : {})}
      >
        <div className={styles.lid}>
          <div className={styles.lidFront}>
            <div className={styles.bezel}>
              <span className={styles.notch} />
              {children}
            </div>
          </div>
          <div className={styles.lidBack} />
          <div className={styles.lidEdge} />
        </div>
        <div className={styles.base} />
      </div>
      {/* Closed-lid slab only — the base below it is always the real `.base`, so
          opening is a crossfade of one 0.95rem sliver instead of a whole machine. */}
      <div className={styles.clamshell} aria-hidden="true">
        <span className={styles.clamshellLid} />
        <span className={styles.clamshellSeam} />
      </div>
    </div>
  )
}

export function RemotePairing() {
  const { isLoaded } = useAppSettings()
  const { enabled: on, toggle } = useRemoteExposure()
  const { status, unreachable } = useRemoteStatus(isLoaded, PAIRING_STATUS_SETTLED_POLL_MS)
  const pairingUrl = pairingUrlFromStatus(status)
  const serving = Boolean(on && pairingUrl)
  const errorMessage = pairingErrorMessage(on, serving, status, unreachable)
  const showSkeleton = on && !serving && errorMessage == null

  return (
    <div className={styles.page}>
      <div className={styles.hero}>
        <DeviceStage open={on}>
          <div
            className={`${styles.screen} ${errorMessage ? styles.screenIdle : ''}`}
            aria-busy={showSkeleton}
          >
            {showSkeleton ? (
              <>
                <span className={styles.srOnly}>Loading pairing link…</span>
                <PairingScreenSkeleton />
              </>
            ) : null}
            {serving && pairingUrl ? (
              <>
                <div className={styles.qrFrame}>
                  <QRCode
                    value={pairingUrl}
                    size={PAIRING_QR_SIZE}
                    level="H"
                    bgColor="#FFFFFF"
                    fgColor="#09090b"
                    title="Remote pairing QR code"
                  />
                </div>
                <CopyableUrl value={pairingUrl} />
              </>
            ) : null}
            {errorMessage ? (
              <Text as="p" variant="body" size="sm" className={styles.reason}>
                {errorMessage}
              </Text>
            ) : null}
          </div>
        </DeviceStage>

        <header className={styles.header}>
          <div className={styles.titleBlock}>
            <Text as="h1" variant="heading" size="xxl" align="center" className={styles.title}>
              Remote control for LTX Desktop
            </Text>
            <Text as="p" variant="body" size="md" align="center" className={styles.lead}>
              Use LTX Desktop from your phone or another browser on the same
              Wi‑Fi network as this computer.
            </Text>
          </div>
          <ExposureToggle on={on} toggle={toggle} />
        </header>
      </div>

      {isLoaded ? <PairedDeviceList poll={on} /> : null}
    </div>
  )
}

function formatLastSeen(ms: number): string {
  return new Date(ms).toLocaleString()
}

function PairedDeviceList({ poll }: { poll: boolean }) {
  const { devices, revoke, revokingId } = useRemoteDevices({ active: true, poll })
  const [nowMs, setNowMs] = useState(() => Date.now())
  const active = devices.filter((device) => device.revoked_at == null)

  useEffect(() => {
    if (active.length === 0) {
      return
    }
    const interval = window.setInterval(() => {
      setNowMs(Date.now())
    }, 5000)
    return () => window.clearInterval(interval)
  }, [active.length])

  if (active.length === 0) {
    return null
  }

  return (
    <div className={styles.devicesMount}>
      <div className={styles.devicesCard}>
      <div className={styles.devicesHeader}>
        <Text as="h2" variant="heading" size="xs">
          Connected devices
        </Text>
      </div>
      <TooltipProvider delay={70}>
      <ul className={styles.deviceList}>
        {active.map((device) => {
          const inUse = isRemoteDeviceInUse(device.last_seen_at, nowMs)
          return (
            <li key={device.id} className={styles.deviceRow}>
              <div className={styles.deviceMeta}>
                <span
                  className={`${styles.liveDot} ${inUse ? styles.liveDotOn : ''}`}
                  aria-label={inUse ? 'In use' : 'Idle'}
                />
                <div className={styles.deviceText}>
                  <div className={styles.deviceNameRow}>
                    <Text as="p" variant="body" size="lg" className={styles.deviceName}>
                      {pairedDeviceDisplayName(device.name)}
                    </Text>
                    {device.ip ? (
                      <Text as="p" variant="body" size="lg" className={styles.deviceIp}>
                        {device.ip}
                      </Text>
                    ) : null}
                  </div>
                  <Text as="p" variant="body" size="xs" className={styles.lastSeen}>
                    {inUse ? 'In use' : `Last seen ${formatLastSeen(device.last_seen_at)}`}
                  </Text>
                </div>
              </div>
              <Tooltip content="Disconnect">
                <span>
                  <Button
                    appearance="overlay"
                    hierarchy="secondary"
                    size="md"
                    isIconOnly
                    leftIcon={<CloseIcon />}
                    aria-label="Disconnect"
                    disabled={revokingId === device.id}
                    onClick={() => {
                      void revoke(device.id)
                    }}
                  />
                </span>
              </Tooltip>
            </li>
          )
        })}
      </ul>
      </TooltipProvider>
      </div>
    </div>
  )
}
