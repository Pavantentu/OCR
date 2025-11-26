import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/OCR/',  // ***IMPORTANT for GitHub Pages***
  plugins: [react()],
  server: {
    open: true,
    proxy: {
      '/api': {
        target: 'https://api.stemverse.api/OCR',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    outDir: 'dist', // Vite default (works with gh-pages -d dist)
  },
});
