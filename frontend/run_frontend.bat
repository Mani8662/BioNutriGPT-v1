@echo off
echo ===================================================
echo Starting BioNutriGPT React Frontend on port 5173...
echo ===================================================
cd /d "%~dp0"
call npm.cmd run dev
pause
