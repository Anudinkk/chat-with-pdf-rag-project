import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // This makes the server accessible on your network, which is needed for Docker
    host: '0.0.0.0', 
    port: 5173 
  }
})
