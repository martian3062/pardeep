import { defineConfig } from 'vite'
import { tanstackStart } from '@tanstack/react-start/plugin/vite'
import basicSsl from '@vitejs/plugin-basic-ssl'
import viteReact from '@vitejs/plugin-react'
import viteTsConfigPaths from 'vite-tsconfig-paths'

/**
 * `npm run dev`     — localhost only
 * `npm run dev:lan` — reachable from a phone on the same Wi-Fi, over HTTPS
 *
 * HTTPS is not decoration. getUserMedia only works in a secure context, so on
 * plain http://192.168.x.x a phone silently refuses the microphone — the page
 * loads, the button works, and no audio is ever captured. A self-signed cert
 * makes the origin secure once the warning is accepted.
 */
const lan = process.env.LAN === '1'

export default defineConfig({
  server: {
    port: 3000,
    host: lan ? true : 'localhost',
    proxy: {
      // The browser stays same-origin; Vite forwards to the Python API
      // server-side, so the API itself never needs exposing to the network.
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
    ...(lan ? [basicSsl()] : []),
  ],
})
