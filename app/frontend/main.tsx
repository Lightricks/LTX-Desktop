import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { macTrafficLightsOnEnd } from '../shared/window-chrome'
import { installProjectStorageDevtools } from './lib/project-storage-devtools'
import './index.css'

installProjectStorageDevtools()

if (window.electronAPI?.platform) {
  document.documentElement.dataset.platform = window.electronAPI.platform
}

if (macTrafficLightsOnEnd(window.electronAPI?.platform, navigator.language)) {
  document.documentElement.dataset.trafficLights = 'end'
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
