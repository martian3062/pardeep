import { defineConfig } from 'vite'
import { tanstackStart } from '@tanstack/react-start/plugin/vite'
import viteTsConfigPaths from 'vite-tsconfig-paths'

export default defineConfig({
  server: {
    port: 3000,
    proxy: {
      // the Python API owns /api; the browser never talks to it cross-origin in dev
      '/api': { target: 'http://127.0.0.1:8100', changeOrigin: true },
    },
  },
  plugins: [viteTsConfigPaths({ projects: ['./tsconfig.json'] }), tanstackStart()],
})
