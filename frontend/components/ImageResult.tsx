import { useState } from 'react'
import { Download, RefreshCw, ImageIcon, Video, Heart, Pencil, MoreHorizontal } from 'lucide-react'
import { Button } from './ui/button'

const MODEL_DISPLAY_NAMES: Record<string, string> = {
  'flux-klein-9b': 'FLUX.2 Klein 9B',
  'z-image-turbo': 'Z-Image Turbo',
  'nano-banana-2': 'Nano Banana 2',
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

interface ImageResultProps {
  imageUrl: string | null
  isGenerating: boolean
  progress: number
  statusMessage: string
  elapsedSeconds?: number
  estimatedSeconds?: number | null
  onCreateVideo: () => void
  onEdit?: (imagePath: string) => void
  modelName?: string | null
}

export function ImageResult({
  imageUrl,
  isGenerating,
  progress,
  statusMessage,
  elapsedSeconds,
  estimatedSeconds,
  onCreateVideo,
  onEdit,
  modelName
}: ImageResultProps) {
  const [isHovered, setIsHovered] = useState(false)

  const handleDownload = () => {
    if (imageUrl) {
      const a = document.createElement('a')
      a.href = imageUrl
      a.download = `zit-image-${Date.now()}.png`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    }
  }

  return (
    <div className="w-full h-full flex flex-col">
      <label className="block text-[12px] font-semibold text-zinc-500 mb-2 uppercase leading-4">
        Result
      </label>
      
      <div className="flex-1 bg-zinc-900 rounded-lg border border-zinc-800 overflow-hidden flex items-center justify-center relative min-h-[400px]">
        {isGenerating ? (
          <div className="flex flex-col items-center justify-center p-8 text-center">
            <RefreshCw className="h-12 w-12 text-primary animate-spin mb-4" />
            <p className="text-lg font-medium text-foreground mb-2">
              {statusMessage || 'Generating Image...'}
            </p>
            <div className="w-64">
              <div className="h-2 bg-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-300"
                  style={{ width: `${Math.max(progress, 2)}%` }}
                />
              </div>
              <div className="flex justify-between items-center mt-2">
                <p className="text-xs text-muted-foreground">
                  {elapsedSeconds != null && elapsedSeconds > 0
                    ? `Elapsed: ${formatTime(elapsedSeconds)}`
                    : progress > 0 ? `${Math.round(progress)}%` : 'Starting...'
                  }
                </p>
                {estimatedSeconds != null && elapsedSeconds != null && elapsedSeconds > 0 && (
                  <p className="text-xs text-muted-foreground">
                    {estimatedSeconds > elapsedSeconds
                      ? `~${formatTime(estimatedSeconds - elapsedSeconds)} left`
                      : 'Finishing up...'
                    }
                  </p>
                )}
              </div>
            </div>
          </div>
        ) : imageUrl ? (
          <div 
            className="relative w-full h-full flex items-center justify-center bg-black"
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
          >
            {/* Image display */}
            <img
              src={imageUrl}
              alt="Generated image"
              className="max-w-full max-h-full object-contain"
            />

            {/* Model badge */}
            {modelName && (
              <div className="absolute bottom-3 left-3 flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur-sm border border-white/10 z-10">
                <div className="w-2 h-2 rounded-full bg-blue-400" />
                <span className="text-[11px] font-medium text-white/80">{MODEL_DISPLAY_NAMES[modelName] || modelName}</span>
              </div>
            )}
            
            {/* Hover overlay - LTX Studio style */}
            <div 
              className={`absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/40 transition-opacity duration-200 ${
                isHovered ? 'opacity-100' : 'opacity-0'
              }`}
            >
              {/* Top toolbar */}
              <div className="absolute top-4 left-4 right-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-9 w-9 bg-black/50 hover:bg-black/70 text-white rounded-full backdrop-blur-sm"
                    title="Favorite"
                  >
                    <Heart className="h-4 w-4" />
                  </Button>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    size="icon"
                    variant="ghost"
                    onClick={handleDownload}
                    className="h-9 w-9 bg-black/50 hover:bg-black/70 text-white rounded-full backdrop-blur-sm"
                    title="Download"
                  >
                    <Download className="h-4 w-4" />
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-9 w-9 bg-black/50 hover:bg-black/70 text-white rounded-full backdrop-blur-sm"
                    title="More options"
                  >
                    <MoreHorizontal className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              
              {/* Center action buttons */}
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 flex items-center gap-3">
                <Button
                  variant="ghost"
                  className="h-10 px-4 bg-black/50 hover:bg-black/70 text-white rounded-full backdrop-blur-sm flex items-center gap-2"
                  title="Edit image"
                  onClick={() => imageUrl && onEdit?.(imageUrl)}
                >
                  <Pencil className="h-4 w-4" />
                  <span className="text-sm font-medium">Edit</span>
                </Button>
                
                <Button
                  onClick={onCreateVideo}
                  className="h-10 px-4 bg-blue-600 hover:bg-blue-500 text-white rounded-full flex items-center gap-2"
                  title="Create video from this image"
                >
                  <Video className="h-4 w-4" />
                  <span className="text-sm font-medium">Create video</span>
                </Button>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center text-zinc-500">
            <div className="w-16 h-16 rounded-full bg-zinc-800 flex items-center justify-center mb-4">
              <ImageIcon className="h-8 w-8 text-zinc-400" />
            </div>
            <p className="text-sm">Generated image will appear here</p>
          </div>
        )}
      </div>
    </div>
  )
}
