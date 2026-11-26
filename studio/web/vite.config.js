import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// During development, proxy API calls to the home server.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://localhost:8765' } },
})
