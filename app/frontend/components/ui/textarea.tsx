import * as React from 'react'
import { cn } from '@/lib/utils'

interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string
  helperText?: string
  charCount?: number
  maxChars?: number
}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, label, helperText, charCount, maxChars, ...props }, ref) => {
    return (
      <div className="w-full">
        {label && (
          <label className="block text-[12px] font-semibold text-zinc-500 mb-2 uppercase leading-4">
            {label}
          </label>
        )}
        <textarea
          className={cn(
            'flex min-h-[120px] w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-3 text-sm text-white',
            'placeholder:text-zinc-500',
            'focus:outline-none focus:ring-1 focus:ring-zinc-500 focus:border-zinc-500',
            'disabled:cursor-not-allowed disabled:opacity-50',
            'resize-y',
            className
          )}
          ref={ref}
          {...props}
        />
        <div className="flex justify-between mt-2">
          {helperText && (
            <span className="text-xs text-zinc-500">{helperText}</span>
          )}
          {maxChars !== undefined && (
            <span className="text-xs text-zinc-500 ml-auto">
              {charCount ?? 0}/{maxChars}
            </span>
          )}
        </div>
      </div>
    )
  }
)
Textarea.displayName = 'Textarea'

export { Textarea }
