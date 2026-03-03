@echo off
title The Marionette

echo.
echo  ==========================================
echo   The Marionette V2  ^|  Comment Generator
echo  ==========================================
echo.

:: Check uv
where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv not found. Installing...
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo [FAIL] Could not install uv.
        echo        Please install manually: https://docs.astral.sh/uv/
        pause
        exit /b 1
    )
    echo [OK] uv installed.
    echo.
)

:: Check .env
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo [NOTICE] .env created from .env.example
        echo          Please open .env and fill in your API keys, then run this script again.
        echo.
        start notepad ".env"
        pause
        exit /b 0
    )
)

:: Sync dependencies
echo [1/2] Syncing dependencies...
uv sync --quiet
if errorlevel 1 (
    echo [ERROR] Dependency sync failed. Check your network or pyproject.toml.
    pause
    exit /b 1
)
echo [OK] Environment ready.
echo.

:: Launch app
echo [2/2] Starting app...
echo        URL: http://localhost:8501
echo        Press Ctrl+C to stop.
echo.
uv run streamlit run app.py --server.headless false --browser.gatherUsageStats false

pause
