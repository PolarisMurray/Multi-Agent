import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// /api/* goes to the Gin gateway, so the browser talks to one origin and needs no CORS.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: 'http://localhost:8080', rewrite: (path) => path.replace(/^\/api/, '') },
    },
  },
})
