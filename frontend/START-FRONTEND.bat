@echo off
setlocal
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Install Node.js 18 or newer, then run this launcher again.
  pause
  exit /b 1
)
where npm.cmd >nul 2>nul
if errorlevel 1 (
  echo npm was not found. Install Node.js with npm and try again.
  pause
  exit /b 1
)
if not exist node_modules\vite (
  echo Installing frontend dependencies...
  call npm.cmd install
  if errorlevel 1 (
    echo Installation failed. Check your connection and try again.
    pause
    exit /b 1
  )
)
echo Start the existing nutrition backend on port 8000 for assessments and plans.
echo Opening the frontend at http://localhost:5173
call npm.cmd run dev -- --host 127.0.0.1
pause
