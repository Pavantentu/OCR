# OCR React App

Lightweight Vite + React scaffold containing the provided App component for uploading documents, loading locations from CSV, and displaying OCR results.

Quick start (Windows, bash):

1. Install dependencies

```bash
cd "c:/Users/T A NAIDU/OneDrive/Desktop/OCR"
npm install
```

2. Start dev server

```bash
npm run dev
```

3. Open the URL printed by Vite (usually http://localhost:5173)

Notes:
- The project uses PapaParse for CSV parsing. 
- There's no backend included. The upload POST to `/api/upload` is a placeholder—replace or proxy to your server.
