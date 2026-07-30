import { defineConfig } from 'vite'
import { tanstackStart } from '@tanstack/react-start/plugin/vite'
import viteReact from '@vitejs/plugin-react'
import viteTsConfigPaths from 'vite-tsconfig-paths'

export default defineConfig({
  server: {
    port: 3000,
    proxy: {
      // the Python API owns /api; the browser never talks to it cross-origin in dev
      '/api': { target: 'http://127.0.0.1:8100', changeOrigin: true },
    },
  },
  plugins: [
    viteTsConfigPaths({ projects: ['./tsconfig.json'] }),
    tanstackStart(),
    // Required, and its absence is silent in a damaging way: without React
    // Refresh the client bundle fails to load, hydration never runs, and the
    // page sits on its server-rendered HTML forever — so a route that fetches in
    // useEffect looks like it is stuck loading rather than broken.
    viteReact(),
  ],
})
