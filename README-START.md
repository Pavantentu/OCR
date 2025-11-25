# Development Setup - Auto-Refresh & Auto-Open

## Quick Start

### Option 1: Start Both Servers (Recommended)
```bash
npm run dev:all
```

This will:
- ✅ Start the Flask backend on http://localhost:5000 (with auto-reload on file changes)
- ✅ Start the Vite frontend on http://localhost:5173 (with auto-refresh on file changes)
- ✅ Automatically open your browser to the frontend

### Option 2: Windows Batch Script
Double-click `start-dev.bat` or run:
```bash
start-dev.bat
```

### Option 3: Start Servers Separately
```bash
# Terminal 1: Backend
npm run dev:backend

# Terminal 2: Frontend  
npm run dev:frontend
```

## Auto-Refresh Features

### Frontend (Vite)
- ✅ **Hot Module Replacement (HMR)**: Changes to React components update instantly without full page reload
- ✅ **Auto-open browser**: Browser opens automatically when dev server starts
- ✅ **Auto-refresh**: Any file changes trigger automatic browser refresh

### Backend (Flask)
- ✅ **Auto-reload**: Flask automatically restarts when Python files are saved
- ✅ **Debug mode**: Full error messages and debugging enabled

## Development URLs

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:5000
- **Health Check**: http://localhost:5000/api/health

## Notes

- Make sure you have Python and Node.js installed
- Backend requires a virtual environment (venv) with dependencies installed
- Frontend requires npm packages installed (`npm install`)
- Both servers will automatically reload when you save code changes

