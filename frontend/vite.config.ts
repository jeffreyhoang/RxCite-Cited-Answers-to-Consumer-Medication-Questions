import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // The API's CORS allows only port 5173
  server: { port: 5173, strictPort: true },
})
