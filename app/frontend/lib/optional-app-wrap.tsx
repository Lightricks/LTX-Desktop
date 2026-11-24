import type { ComponentType, ReactNode } from 'react'

export type BootOverlayContext = {
  documentMode: 'install' | 'app'
  mainMode: 'install' | 'app' | null
  phaseKind: string
  setDocumentWindowMode: (mode: 'install' | 'app') => void
  setSplashFinished: (finished: boolean) => void
  children: ReactNode
}

/** Identity wrap. Vite aliases this module to a local overlay when present. */
export function wrapApp(App: ComponentType): ComponentType {
  return App
}

/** Pass-through. Vite aliases this module to a local overlay when present. */
export function OptionalBootGate({ children }: BootOverlayContext): ReactNode {
  return children
}
