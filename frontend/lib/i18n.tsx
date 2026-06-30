import React, { createContext, useContext, useState, useCallback, useEffect, useMemo } from 'react'

// ============================================================
// LTX Desktop i18n — lightweight React Context translation
// ============================================================

export type Language = 'en' | 'zh'

// Dynamic loader: imported by the bundler, not embedded inline
type TranslationMap = Record<string, string>

const localeModules: Record<Language, () => Promise<TranslationMap>> = {
  en: () => import('./locales/en.json').then(m => m.default || m),
  zh: () => import('./locales/zh.json').then(m => m.default || m),
}

interface I18nContextValue {
  language: Language
  setLanguage: (lang: Language) => void
  t: (key: string, vars?: Record<string, string | number>, fallback?: string) => string
  loading: boolean
}

const I18nContext = createContext<I18nContextValue>({
  language: 'en',
  setLanguage: () => {},
  t: (key: string, _vars?: Record<string, string | number>, fallback?: string) => fallback ?? key,
  loading: false,
})

const STORAGE_KEY = 'ltx-language'

function getSavedLanguage(): Language {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === 'zh' || stored === 'en') return stored
    // Detect system language
    const navLang = navigator.language?.toLowerCase() || ''
    if (navLang.startsWith('zh')) return 'zh'
  } catch {}
  return 'en'
}

function saveLanguage(lang: Language): void {
  try { localStorage.setItem(STORAGE_KEY, lang) } catch {}
}

export function I18nProvider({ children, defaultLanguage }: {
  children: React.ReactNode
  defaultLanguage?: Language
}) {
  const [language, setLanguageState] = useState<Language>(defaultLanguage ?? getSavedLanguage)
  const [messages, setMessages] = useState<TranslationMap | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    localeModules[language]()
      .then(module => {
        if (!cancelled) {
          setMessages(module)
          setLoading(false)
        }
      })
      .catch(() => {
        if (!cancelled) {
          // Fallback: use empty dict (keys become display text)
          setMessages({})
          setLoading(false)
        }
      })
    return () => { cancelled = true }
  }, [language])

  const setLanguage = useCallback((lang: Language) => {
    setLanguageState(lang)
    saveLanguage(lang)
  }, [])

  const t = useCallback((key: string, vars?: Record<string, string | number>, fallback?: string): string => {
    if (!messages) return fallback ?? key
    let value = messages[key] ?? fallback ?? key
    if (vars) {
      value = value.replace(/\{\{(\w+)\}\}/g, (_, v) => String(vars[v] ?? ''))
    }
    return value
  }, [messages])

  const value = useMemo(() => ({ language, setLanguage, t, loading }), [language, setLanguage, t, loading])

  return React.createElement(I18nContext.Provider, { value }, children)
}

export function useI18n(): I18nContextValue {
  return useContext(I18nContext)
}

// Convenience shorthand
export function useT() {
  const { t, language, setLanguage } = useI18n()
  return { t, language, setLanguage }
}
