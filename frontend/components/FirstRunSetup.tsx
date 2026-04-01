import { useState, useEffect } from 'react'
import { backendFetch } from '../lib/backend'
import { logger } from '../lib/logger'
import './FirstRunSetup.css'

interface LaunchGateProps {
  licenseOnly?: boolean
  showLicenseStep?: boolean
  onComplete: () => Promise<void>
  onAcceptLicense?: () => Promise<void>
}

type Step = 'license' | 'location' | 'installing' | 'complete'

interface DownloadProgress {
  status: 'idle' | 'downloading' | 'complete' | 'error'
  current_downloading_file: string | null
  current_file_progress: number
  total_progress: number
  total_downloaded_bytes: number
  expected_total_bytes: number
  completed_files: string[]
  all_files: string[]
  error: string | null
  speed_bytes_per_sec: number
}

interface RequiredModelsResponse {
  modelTypes: string[]
}

// Fun loading messages
const INSTALL_MESSAGES = [
  "Downloading model weights...",
  "Teaching AI to dream in 4K...",
  "Loading neural pathways...",
  "Calibrating inference engine...",
  "Almost there...",
  "Unpacking the magic...",
  "Configuring parameters...",
  "Finalizing installation..."
]


export function LaunchGate({
  licenseOnly,
  showLicenseStep = true,
  onComplete,
  onAcceptLicense,
}: LaunchGateProps) {
  const [currentStep, setCurrentStep] = useState<Step>(showLicenseStep ? 'license' : 'location')
  const [installPath, setInstallPath] = useState('')
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [downloadSessionId, setDownloadSessionId] = useState<string | null>(null)
  const [installMessage, setInstallMessage] = useState(INSTALL_MESSAGES[0])
  const [availableSpace, setAvailableSpace] = useState('...')
  const [videoPath, setVideoPath] = useState('/splash/splash.mp4')
  const [ltxApiKey, setLtxApiKey] = useState('')
  const [licenseAccepted, setLicenseAccepted] = useState(false)
  const [licenseText, setLicenseText] = useState<string | null>(null)
  const [licenseError, setLicenseError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [isActionPending, setIsActionPending] = useState(false)

  // Format bytes to human readable
  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`
  }

  // Format time remaining
  const formatTimeRemaining = (seconds: number): string => {
    if (!seconds || !isFinite(seconds) || seconds <= 0) return '--'
    if (seconds < 60) return `${Math.round(seconds)}s`
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`
    return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`
  }

  // Calculate ETA based on speed and remaining bytes
  const getTimeRemaining = (): string => {
    if (!downloadProgress || downloadProgress.speed_bytes_per_sec <= 0) return '--'
    const remainingBytes = downloadProgress.expected_total_bytes - downloadProgress.total_downloaded_bytes
    if (remainingBytes <= 0) return '--'
    const secondsRemaining = remainingBytes / downloadProgress.speed_bytes_per_sec
    return formatTimeRemaining(secondsRemaining)
  }

  // Fetch license text
  const fetchLicense = async () => {
    setLicenseError(null)
    setLicenseText(null)
    try {
      const text = await window.electronAPI.fetchLicenseText()
      setLicenseText(text)
    } catch (e) {
      setLicenseError(e instanceof Error ? e.message : 'Failed to fetch license text.')
    }
  }

  // Initialize
  useEffect(() => {
    const init = async () => {
      try {
        // Get video path for production (unpacked from asar)
        try {
          const resourcePath = await window.electronAPI.getResourcePath?.()
          if (resourcePath) {
            setVideoPath(`file://${resourcePath}/app.asar.unpacked/dist/splash/splash.mp4`)
          }
        } catch {
          // Dev mode: use relative path
          setVideoPath('/splash/splash.mp4')
        }

        // Get models path from backend
        try {
          const response = await backendFetch('/api/models/status')
          if (response.ok) {
            const data = await response.json()
            if (data.models_path) {
              setInstallPath(data.models_path)
            }
          }
        } catch (e) {
          logger.error(`Failed to get models path: ${e}`)
        }

        // TODO: Get actual available space
        setAvailableSpace('1.8 TB')
      } catch (e) {
        logger.error(`Init error: ${e}`)
      }
    }
    init()
    if (showLicenseStep) {
      void fetchLicense()
    }
  }, [showLicenseStep])

  // Cycle install messages
  useEffect(() => {
    if (currentStep !== 'installing') return
    let index = 0
    const interval = setInterval(() => {
      index = (index + 1) % INSTALL_MESSAGES.length
      setInstallMessage(INSTALL_MESSAGES[index])
    }, 4000)
    return () => clearInterval(interval)
  }, [currentStep])

  // Poll download progress during installation
  useEffect(() => {
    if (currentStep !== 'installing' || !downloadSessionId) return

    const pollProgress = async () => {
      try {
        const response = await backendFetch(`/api/models/download/progress?sessionId=${downloadSessionId}`)
        if (response.ok) {
          const progress = await response.json()
          setDownloadProgress(progress)

          if (progress.status === 'error') {
            setDownloadError(progress.error || 'Download failed.')
          } else if (progress.status === 'complete') {
            setTimeout(() => setCurrentStep('complete'), 600)
          }
        }
      } catch (e) {
        logger.error(`Progress poll error: ${e}`)
      }
    }

    pollProgress()
    const interval = setInterval(pollProgress, 500)
    return () => clearInterval(interval)
  }, [currentStep, downloadSessionId])

  // Start installation
  const startInstallation = async () => {
    setCurrentStep('installing')
    try {
      // If API key is provided, save it to settings first and skip text encoder download
      if (ltxApiKey.trim()) {
        try {
          await backendFetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ltxApiKey: ltxApiKey.trim() }),
          })
        } catch (e) {
          logger.error(`Failed to save API key: ${e}`)
        }
      }

      // Start download - skip text encoder if API key is provided
      const skipTextEncoder = !!ltxApiKey.trim()
      const requiredModelsResponse = await backendFetch(`/api/models/required-models?skipTextEncoder=${skipTextEncoder}`)
      const requiredModels = requiredModelsResponse.ok
        ? await requiredModelsResponse.json() as RequiredModelsResponse
        : { modelTypes: [] }

      const downloadResponse = await backendFetch('/api/models/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modelTypes: requiredModels.modelTypes }),
      })
      if (downloadResponse.ok) {
        const downloadData = await downloadResponse.json()
        if (downloadData.sessionId) {
          setDownloadSessionId(downloadData.sessionId)
        }
      }
    } catch (e) {
      logger.error(`Download start error: ${e}`)
      setDownloadError(e instanceof Error ? e.message : 'Failed to start model download.')
    }
  }

  const retryInstallation = () => {
    setDownloadError(null)
    startInstallation()
  }

  // Handle next button
  const handleNext = async () => {
    setActionError(null)
    if (currentStep === 'license') {
      if (!licenseAccepted) return
      setIsActionPending(true)
      try {
        if (onAcceptLicense) {
          await onAcceptLicense()
        }
        if (licenseOnly) {
          await onComplete()
          return
        }
        setCurrentStep('location')
      } catch (e) {
        setActionError(e instanceof Error ? e.message : 'Failed to accept license.')
      } finally {
        setIsActionPending(false)
      }
      return
    }
    if (currentStep === 'location') {
      startInstallation()
      return
    }
    if (currentStep === 'complete') {
      await handleFinish()
    }
  }

  const handleFinish = async () => {
    setActionError(null)
    setIsActionPending(true)
    try {
      await onComplete()
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Failed to complete setup.')
    } finally {
      setIsActionPending(false)
    }
  }

  // Get button text
  const getNextButtonText = () => {
    if (currentStep === 'license') return licenseOnly ? 'Accept' : 'Next'
    if (currentStep === 'location') return 'Install'
    if (currentStep === 'complete') return 'Finish'
    return 'Continue'
  }

  // Check if next button should be disabled
  const isNextDisabled = () => {
    if (currentStep === 'license') return !licenseAccepted || isActionPending
    if (currentStep === 'complete') return isActionPending
    return false
  }

  return (
    <div className="h-screen flex flex-col" style={{
      background: '#000000',
      fontFamily: 'Arial, Helvetica, sans-serif',
      color: '#ffffff'
    }}>
      {/* Custom Title Bar */}
      <div style={{
        height: 32,
        background: '#000000',
        display: 'flex',
        alignItems: 'center',
        paddingLeft: 80,
        borderBottom: '1px solid #1a1a1a',
        // @ts-expect-error - Electron-specific CSS property
        WebkitAppRegion: 'drag'
      }}>
      </div>

      {/* Main Container */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        overflow: 'hidden',
        minHeight: 0,
        // @ts-expect-error - Electron-specific CSS property
        WebkitAppRegion: 'no-drag'
      }}>
        {/* Header */}
        <div style={{
          padding: currentStep === 'installing' ? '12px 32px' : '16px 32px',
          borderBottom: '1px solid #1a1a1a'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            {/* LTX Logo */}
            <svg style={{ height: 24, width: 'auto' }} viewBox="0 0 75 32" xmlns="http://www.w3.org/2000/svg" fill="none">
              <path d="M0 30.0087V7.50057C0 7.09765 0.154254 6.69973 0.460162 6.43869C0.708822 6.22729 0.987356 6.12029 1.29316 6.12029H8.2339C8.63671 6.12029 9.03463 6.26205 9.31316 6.55308C9.53944 6.79174 9.65133 7.07777 9.65133 7.41345V23.198C9.65133 23.6108 9.98462 23.944 10.3974 23.944H21.4638C21.8666 23.944 22.267 24.0858 22.5431 24.3767C22.7668 24.6155 22.8812 24.9015 22.8812 25.2372V30.5856C22.8812 31.0457 22.6823 31.4982 22.3018 31.7569C22.078 31.9086 21.8244 31.9832 21.5383 31.9832L1.99199 31.9956C0.890348 31.9981 0 31.1078 0 30.0087Z" fill="white"/>
              <path d="M36.5888 31.9926C34.4062 31.9876 32.492 31.6543 30.8413 30.9878C29.1906 30.3214 27.9104 29.2346 26.9981 27.7227C26.0856 26.2132 25.6333 24.2137 25.6382 21.7269L25.6532 13.7194L21.7016 13.7119C21.3486 13.7119 21.0528 13.5876 20.8116 13.3365C20.5705 13.0853 20.4512 12.7819 20.4537 12.4164L20.4636 7.39299C20.4636 7.02744 20.5854 6.72154 20.8265 6.47288C21.0677 6.22422 21.3635 6.09983 21.7165 6.10233L25.6681 6.10983L25.6779 1.29066C25.6779 0.925114 25.7998 0.619208 26.041 0.370548C26.2821 0.121887 26.5779 0 26.9309 0L33.9065 0.0124913C34.2595 0.0124913 34.5554 0.136772 34.7965 0.387931C35.0376 0.639089 35.1569 0.944995 35.1545 1.30805L35.1445 6.12721L41.2078 6.13959C41.5608 6.13959 41.8566 6.26398 42.0977 6.51514C42.3389 6.76629 42.4582 7.0722 42.4557 7.43525L42.4458 12.4586C42.4458 12.8242 42.3239 13.1301 42.0829 13.3787C41.8417 13.6274 41.5434 13.7518 41.1928 13.7493L35.1296 13.7368L35.1171 20.8988C35.1171 21.8613 35.3061 22.6148 35.6914 23.1618C36.0767 23.7089 36.6833 23.985 37.5186 23.985L41.5608 23.9925C41.9138 23.9925 42.2096 24.1168 42.4507 24.368C42.6919 24.6192 42.8113 24.9251 42.8088 25.2881L42.7988 30.7093C42.7988 31.075 42.677 31.3808 42.4358 31.6294C42.1947 31.8782 41.8963 32.0025 41.5459 32L36.5913 31.9901L36.5888 31.9926Z" fill="white"/>
              <path d="M47.5486 31.9851C47.2282 31.9851 46.965 31.8682 46.7589 31.6369C46.5503 31.4056 46.4485 31.1395 46.4485 30.841C46.4485 30.7416 46.4634 30.6248 46.4957 30.4929C46.5279 30.3611 46.5926 30.2268 46.6869 30.0951L54.3506 18.9342C54.4648 18.7675 54.4673 18.5463 54.3556 18.3771L47.4543 8.01457C47.3896 7.91517 47.335 7.79827 47.2854 7.6664C47.2382 7.53463 47.2133 7.40036 47.2133 7.26859C47.2133 6.97017 47.3251 6.70403 47.5486 6.47275C47.7722 6.24147 48.0279 6.12458 48.316 6.12458H55.6444C56.0914 6.12458 56.4267 6.23158 56.6501 6.44787C56.8737 6.66426 57.0327 6.85328 57.1295 7.01993L60.3082 11.8169C60.5043 12.1128 60.939 12.1128 61.1352 11.8169L64.3139 7.01993C64.4405 6.85328 64.6094 6.66426 64.8156 6.44787C65.0216 6.23158 65.3494 6.12458 65.7964 6.12458H72.7896C73.0778 6.12458 73.331 6.24147 73.557 6.47275C73.7805 6.70403 73.8922 6.95268 73.8922 7.21883C73.8922 7.38547 73.8748 7.53463 73.8451 7.6664C73.8128 7.79827 73.7482 7.91517 73.6539 8.01457L66.6159 18.3747C66.4992 18.5463 66.5017 18.77 66.6209 18.9392L74.4212 30.0975C74.5181 30.2293 74.5801 30.3636 74.6124 30.4954C74.6448 30.6273 74.6596 30.744 74.6596 30.8435C74.6596 31.142 74.5479 31.4081 74.3244 31.6394C74.1008 31.8707 73.8451 31.9874 73.557 31.9874H65.8934C65.4786 31.9874 65.1756 31.888 64.9844 31.689C64.7932 31.4901 64.6317 31.3086 64.5051 31.142L60.9886 25.9544C60.7924 25.671 60.3753 25.6685 60.1766 25.9471L56.4118 31.1395C56.3149 31.3061 56.1634 31.4876 55.9573 31.6865C55.7488 31.8855 55.4383 31.9851 55.0236 31.9851H47.5486Z" fill="white"/>
            </svg>
            <span style={{ fontSize: 14, color: '#71717a', fontWeight: 500, letterSpacing: '0.02em', paddingTop: 2, paddingLeft: 6 }}>Desktop</span>
          </div>
        </div>

        {/* Content Area */}
        <div style={{
          flex: 1,
          padding: currentStep === 'installing' ? 0 : '28px 32px',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}>
          {/* Step 1: Model License */}
          {currentStep === 'license' && (
            <div style={{ animation: 'fadeIn 0.25s ease', display: 'flex', flexDirection: 'column', overflow: 'hidden', flex: 1 }}>
              <h2 style={{
                fontFamily: "'Miriam Libre', serif",
                fontSize: 24,
                fontWeight: 700,
                marginBottom: 6
              }}>
                LTX-2 Model License
              </h2>
              <p style={{ color: '#a0a0a0', fontSize: 14, marginBottom: 16 }}>
                The LTX-2 model is subject to the following license agreement. Please review and accept before downloading.
              </p>

              <div style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
                minHeight: 0
              }}>
                <div style={{
                  flex: 1,
                  overflow: 'hidden',
                  borderRadius: 8,
                  minHeight: 0
                }}>
                  {licenseError ? (
                    <div style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      height: '100%',
                      gap: 12
                    }}>
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#f87171" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                      </svg>
                      <span style={{ color: '#f87171', fontSize: 13, textAlign: 'center' }}>{licenseError}</span>
                      <button
                        onClick={fetchLicense}
                        style={{
                          padding: '6px 20px',
                          borderRadius: 9999,
                          fontSize: 13,
                          fontWeight: 600,
                          cursor: 'pointer',
                          background: 'linear-gradient(125deg, #A98BD9, #6D28D9)',
                          border: 'none',
                          color: '#ffffff',
                        }}
                      >
                        Retry
                      </button>
                    </div>
                  ) : licenseText === null ? (
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      height: '100%',
                      gap: 10
                    }}>
                      <svg width="20" height="20" viewBox="0 0 24 24" style={{ animation: 'spin 1s linear infinite' }}>
                        <circle cx="12" cy="12" r="10" stroke="#6D28D9" strokeWidth="3" fill="none" strokeDasharray="31.4 31.4" strokeLinecap="round" />
                      </svg>
                      <span style={{ color: '#a0a0a0', fontSize: 13 }}>Loading license...</span>
                    </div>
                  ) : (
                    <div style={{
                      overflowY: 'auto',
                      height: '100%',
                      background: '#1a1a1a',
                      borderRadius: 8,
                      padding: 40
                    }}>
                      <pre style={{
                        fontFamily: "'Consolas', 'Monaco', monospace",
                        fontSize: 11,
                        lineHeight: 1.5,
                        color: '#d0d0d0',
                        margin: 0,
                        whiteSpace: 'pre-line',
                        wordWrap: 'break-word',
                        width: '100%'
                      }}>
                        {licenseText?.replace(/([^\n])\n([^\n])/g, '$1 $2')}
                      </pre>
                    </div>
                  )}
                </div>

                <label style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  marginTop: 14,
                  cursor: 'pointer',
                  fontSize: 13,
                  userSelect: 'none'
                }}>
                  <input
                    type="checkbox"
                    checked={licenseAccepted}
                    onChange={(e) => setLicenseAccepted(e.target.checked)}
                    style={{
                      width: 16,
                      height: 16,
                      accentColor: '#2B61FF',
                      cursor: 'pointer',
                      flexShrink: 0
                    }}
                  />
                  <span>I have read and agree to the LTX-2 Community License Agreement</span>
                </label>
              </div>
            </div>
          )}

          {/* Step 2: Choose Location */}
          {currentStep === 'location' && (
            <div style={{ animation: 'fadeIn 0.25s ease' }}>
              <h2 style={{
                fontFamily: "'Miriam Libre', serif",
                fontSize: 24,
                fontWeight: 700,
                marginBottom: 6
              }}>
                Choose Location
              </h2>
              <p style={{ color: '#a0a0a0', fontSize: 14, marginBottom: 24 }}>
                Select where to install the model files.
              </p>

              <div style={{
                background: '#2e3445',
                borderRadius: 12,
                padding: '14px 18px'
              }}>
                <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                  <input
                    type="text"
                    value={installPath}
                    readOnly
                    style={{
                      flex: 1,
                      background: '#1a1a1a',
                      border: '1px solid #333',
                      borderRadius: 8,
                      padding: '12px 14px',
                      color: '#ffffff',
                      fontSize: 13,
                      fontFamily: "'Consolas', 'Monaco', monospace"
                    }}
                  />
                  <button
                    onClick={async () => {
                      const result = await window.electronAPI?.openModelsDirChangeDialog()
                      if (result?.success && result.path) {
                        setInstallPath(result.path)
                      }
                    }}
                    style={{
                      padding: '10px 28px',
                      borderRadius: 9999,
                      fontSize: 13,
                      fontWeight: 600,
                      cursor: 'pointer',
                      background: 'transparent',
                      border: '1px solid #444',
                      color: '#ffffff',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    Browse
                  </button>
                </div>

                <div style={{
                  display: 'flex',
                  justifyContent: 'flex-end',
                  fontSize: 12,
                  color: '#a0a0a0',
                  marginTop: 10
                }}>
                  <span>Available: <strong style={{ color: '#fff' }}>{availableSpace}</strong></span>
                </div>
              </div>

              {/* LTX API Key - Optional but saves ~25 GB download */}
              <div style={{
                marginTop: 24,
                background: '#2e3445',
                borderRadius: 12,
                padding: '14px 18px'
              }}>
                <div style={{ marginBottom: 8 }}>
                  <label style={{ fontSize: 13, fontWeight: 600, color: '#ffffff' }}>
                    LTX API Key
                    <span style={{
                      fontSize: 11,
                      color: '#A98BD9',
                      marginLeft: 8,
                      fontWeight: 400
                    }}>
                      Optional - Saves ~25 GB download
                    </span>
                  </label>
                </div>
                <input
                  type="password"
                  value={ltxApiKey}
                  onChange={(e) => setLtxApiKey(e.target.value)}
                  placeholder="Enter API key to skip text encoder download..."
                  style={{
                    width: '100%',
                    background: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: 8,
                    padding: '12px 14px',
                    color: '#ffffff',
                    fontSize: 13,
                    boxSizing: 'border-box'
                  }}
                />
                <p style={{ fontSize: 11, color: '#888', marginTop: 8 }}>
                  {ltxApiKey ? (
                    <span style={{ color: '#6D28D9' }}>
                      ✓ Text encoder download will be skipped (using API instead)
                    </span>
                  ) : (
                    'If you have an LTX API key, entering it here skips the 25 GB text encoder download. ' +
                    'The API provides faster text encoding (~1s vs 23s local).'
                  )}
                </p>
              </div>
            </div>
          )}

          {/* Step 3: Installing */}
          {currentStep === 'installing' && (
            <div style={{
              position: 'relative',
              height: '100%',
              animation: 'fadeIn 0.25s ease'
            }}>
              {/* Video Section - fills container but leaves room for progress */}
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 140,
                background: '#0a0a0a',
                overflow: 'hidden'
              }}>
                {/* Splash Video */}
                <video
                  key={videoPath}
                  autoPlay
                  loop
                  muted
                  playsInline
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    display: 'block'
                  }}
                >
                  <source src={videoPath} type="video/mp4" />
                </video>

                {/* Video Credit */}
                <div style={{
                  position: 'absolute',
                  bottom: 20,
                  left: 24,
                  fontFamily: "'Miriam Libre', serif",
                  fontSize: 13,
                  color: 'rgba(255,255,255,0.75)',
                  textShadow: '0 1px 4px rgba(0,0,0,0.9)',
                  zIndex: 10
                }}>
                  Generated by PongFlongo
                </div>
              </div>

              {/* Progress Section - fixed at bottom */}
              <div style={{
                position: 'absolute',
                left: 0,
                right: 0,
                bottom: 0,
                height: 140,
                background: '#0d0d0d',
                padding: '16px 24px',
                borderTop: '1px solid #2a2a2a'
              }}>
              {downloadError ? (
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '100%',
                  gap: 10,
                }}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#f87171" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                  </svg>
                  <span style={{ color: '#f87171', fontSize: 13, textAlign: 'center', maxWidth: 400 }}>{downloadError}</span>
                  <div style={{ display: 'flex', gap: 10 }}>
                    <button
                      onClick={() => { setDownloadError(null); setCurrentStep('location') }}
                      style={{
                        padding: '6px 20px',
                        borderRadius: 9999,
                        fontSize: 13,
                        fontWeight: 600,
                        cursor: 'pointer',
                        background: 'transparent',
                        border: '1px solid #444',
                        color: '#ffffff',
                      }}
                    >
                      Back
                    </button>
                    <button
                      onClick={retryInstallation}
                      style={{
                        padding: '6px 20px',
                        borderRadius: 9999,
                        fontSize: 13,
                        fontWeight: 600,
                        cursor: 'pointer',
                        background: 'linear-gradient(125deg, #A98BD9, #6D28D9)',
                        border: 'none',
                        color: '#ffffff',
                      }}
                    >
                      Retry
                    </button>
                  </div>
                </div>
              ) : (
              <>
                {/* Header row with status and percentage */}
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: 8
                }}>
                  <span style={{ fontSize: 13, fontWeight: 500 }}>
                    {(downloadProgress?.total_progress || 0) > 85 ? 'Installing...' : 'Downloading...'}
                  </span>
                  <span style={{ fontSize: 13, color: '#A98BD9', fontWeight: 600 }}>
                    {Math.round(downloadProgress?.total_progress || 0)}%
                  </span>
                </div>

                {/* Progress Bar */}
                <div style={{
                  height: 6,
                  background: '#1a1a1a',
                  borderRadius: 3,
                  overflow: 'hidden'
                }}>
                  <div style={{
                    height: '100%',
                    background: 'linear-gradient(125deg, #A98BD9, #6D28D9, #194DF9)',
                    backgroundSize: '200% 200%',
                    animation: 'gradientShift 3s ease infinite',
                    borderRadius: 3,
                    width: `${downloadProgress?.total_progress || 0}%`,
                    transition: 'width 0.3s ease'
                  }} />
                </div>

                {/* Download stats row */}
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginTop: 10,
                  fontSize: 12,
                  color: '#a0a0a0'
                }}>
                  {/* Current file */}
                  <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {downloadProgress?.current_downloading_file || installMessage}
                  </span>

                  {/* Speed and ETA */}
                  <div style={{ display: 'flex', gap: 16, marginLeft: 16, flexShrink: 0 }}>
                    {downloadProgress && downloadProgress.speed_bytes_per_sec > 0 && (
                      <span style={{ color: '#6D28D9', fontWeight: 500 }}>
                        {(downloadProgress.speed_bytes_per_sec / (1024 * 1024)).toFixed(1)} MB/s
                      </span>
                    )}
                    {downloadProgress && downloadProgress.expected_total_bytes > 0 && (
                      <span>
                        {formatBytes(downloadProgress.total_downloaded_bytes)} / {formatBytes(downloadProgress.expected_total_bytes)}
                      </span>
                    )}
                    {downloadProgress && downloadProgress.speed_bytes_per_sec > 0 && (
                      <span>
                        ETA: {getTimeRemaining()}
                      </span>
                    )}
                  </div>
                </div>

                {/* Files progress */}
                {downloadProgress && downloadProgress.all_files.length > 0 && (
                  <div style={{
                    marginTop: 6,
                    fontSize: 11,
                    color: '#666'
                  }}>
                    File {downloadProgress.completed_files.length + 1} of {downloadProgress.all_files.length}
                  </div>
                )}
              </>
              )}
              </div>
            </div>
          )}

          {/* Step 4: Complete */}
          {currentStep === 'complete' && (
            <div style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              textAlign: 'center',
              animation: 'fadeIn 0.25s ease'
            }}>
              {/* Success Icon */}
              <div style={{
                width: 72,
                height: 72,
                background: 'linear-gradient(125deg, #A98BD9, #6D28D9, #194DF9)',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: 20
              }}>
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                  <polyline points="20 6 9 17 4 12"/>
                </svg>
              </div>

              <h2 style={{
                fontFamily: "'Miriam Libre', serif",
                fontSize: 26,
                fontWeight: 700,
                marginBottom: 8
              }}>
                Ready to Create
              </h2>
              <p style={{ color: '#a0a0a0', fontSize: 14, maxWidth: 320 }}>
                LTX Video is installed. Start generating.
              </p>

              {/* Install Summary */}
              <div style={{
                background: '#2e3445',
                borderRadius: 12,
                padding: '16px 28px',
                marginTop: 20,
                textAlign: 'left',
                minWidth: 260
              }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  padding: '8px 0',
                  fontSize: 13
                }}>
                  <span style={{ color: '#a0a0a0' }}>Location</span>
                  <span style={{ fontWeight: 500, maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {installPath.split('\\').pop() || installPath}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{
          padding: currentStep === 'installing' ? '12px 24px' : '16px 32px',
          borderTop: '1px solid #1a1a1a',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div style={{ fontSize: 11, color: '#666' }}>© 2026 Lightricks</div>

          <div style={{ display: 'flex', gap: 10 }}>
            {/* Next/Install/Finish Button */}
            {currentStep !== 'installing' && (
              <button
                onClick={() => void handleNext()}
                disabled={isNextDisabled()}
                style={{
                  padding: '10px 28px',
                  borderRadius: 8,
                  fontSize: 13,
                  fontWeight: 700,
                  cursor: isNextDisabled() ? 'not-allowed' : 'pointer',
                  background: isNextDisabled() ? '#555' : '#2B61FF',
                  border: 'none',
                  color: '#ffffff',
                  transition: 'all 0.2s ease',
                  opacity: isNextDisabled() ? 0.6 : 1
                }}
              >
                {getNextButtonText()}
              </button>
            )}
          </div>
        </div>
        {actionError && (
          <div style={{ padding: '0 32px 12px 32px', color: '#fca5a5', fontSize: 12 }}>
            {actionError}
          </div>
        )}
      </div>

    </div>
  )
}
