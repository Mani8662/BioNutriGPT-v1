@echo off
title BioNutriGPT Clinical Application Launcher
echo ===================================================================
echo             BIONUTRIGPT CLINICAL DECISION SUPPORT PLATFORM
echo ===================================================================
echo  Starting FastAPI Backend (Port 8000) and Vite React Frontend (Port 5173)...
echo ===================================================================

set "APP_DIR=%~dp0"
if not exist "%APP_DIR%backend" (
    set "APP_DIR=%~dp0Models\Less Data\"
)

start "BioNutriGPT Backend (FastAPI)" cmd /k "cd /d "%APP_DIR%backend" && python main.py"
timeout /t 3 /nobreak >nul

start "BioNutriGPT Frontend (React + Vite)" cmd /k "cd /d "%APP_DIR%frontend" && npm.cmd run dev"
timeout /t 2 /nobreak >nul

echo.
echo BioNutriGPT Full-Stack System Launched!
echo   - Backend API Docs : http://localhost:8000/docs
echo   - Frontend UI      : http://localhost:5173
echo.
pause
