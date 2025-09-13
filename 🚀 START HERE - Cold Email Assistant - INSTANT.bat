@echo off
title Cold Email Assistant - Instant Launch
color 0B

echo ================================================================
echo    Cold Email Assistant - Premium Edition (INSTANT)
echo      Professional AI Email Generation Tool
echo ================================================================
echo.
echo ✓ Everything is pre-installed - launching immediately!
echo ✓ No waiting, no installation - just pure performance!
echo.
echo - The AI model will download automatically when you first use the app
echo - Keep this window open while using the app
echo.
pause

echo.
echo Starting Cold Email Assistant...
echo ================================================================

:: Check if models directory exists
if not exist "models" (
    echo Creating models directory...
    mkdir models
)

:: Activate pre-configured virtual environment and start the application
echo Activating portable environment...
call venv-portable\Scripts\activate.bat

echo Starting app with pre-configured environment...
python -m streamlit run app.py --server.port 8501

echo.
echo ================================================================
echo  Cold Email Assistant session ended
echo  To restart: Just run this file again!
echo ================================================================
pause
