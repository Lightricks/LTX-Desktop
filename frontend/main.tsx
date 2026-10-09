import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { installProjectStorageDevtools } from './lib/project-storage-devtools'
import './index.css'

installProjectStorageDevtools()

if (window.electronAPI?.platform) {
  document.documentElement.dataset.platform = window.electronAPI.platform
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
