import { AlertCircle } from 'lucide-react'
import { LogViewer } from './LogViewer'
import { Button } from './ui/button'

/** Full-window screen for a backend that crashed and could not restart. Shows the logs. */
export function BackendCrashedScreen() {
  return (
    <div className="h-screen bg-surface-secondary flex items-center justify-center p-6">
      <div className="w-full max-w-5xl rounded-xl border border-separator bg-surface-primary p-6 shadow-2xl">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 text-fg-danger mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-fg-primary mb-2">The backend process crashed and could not be restarted</h2>
          <p className="text-base text-fg-secondary mb-4">Review the logs below and restart the application.</p>
        </div>
        <div className="h-[50vh]">
          <LogViewer isOpen={true} onClose={() => {}} embedded={true} />
        </div>
        <div className="mt-4 flex justify-center">
          <Button onClick={() => window.location.reload()}>Restart Application</Button>
        </div>
      </div>
    </div>
  )
}
