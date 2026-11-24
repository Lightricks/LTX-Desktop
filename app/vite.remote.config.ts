import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import svgr from 'vite-plugin-svgr'
import path from 'path'

export default defineConfig({
  plugins: [
    react(),
    svgr({ include: '**/*.svg?react' }),
  ],
  root: path.resolve(__dirname, 'frontend/remote'),
  base: '/',
  css: {
    preprocessorOptions: {
      scss: {
        api: 'modern-compiler',
        silenceDeprecations: ['import'],
      },
    },
  },
  resolve: {
    alias: {
      '@ds': path.resolve(__dirname, './frontend/ds'),
      '@': path.resolve(__dirname, './frontend'),
    },
  },
  server: {
    fs: {
      allow: [path.resolve(__dirname)],
    },
  },
  build: {
    outDir: path.resolve(__dirname, 'dist-remote'),
    emptyOutDir: true,
    assetsDir: 'static',
  },
})
