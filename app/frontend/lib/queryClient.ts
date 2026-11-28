import { QueryClient } from '@tanstack/react-query'

/**
 * App-wide react-query client. A single cache is shared across every route
 * (home features, projects, editor) so data isn't siloed per layout subtree.
 * Retries and window-focus refetches are off — the backend is local and we
 * poll explicitly where freshness matters.
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false, refetchOnWindowFocus: false },
    mutations: { retry: false },
  },
})
