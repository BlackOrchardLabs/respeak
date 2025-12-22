@echo off
REM Re:speak TTS GUI Launcher (Orchard Theme)
REM Requires Python 3.11 for Coqui TTS voice cloning

echo Starting Re:speak - Voice Kernel Studio...

cd /d "%~dp0"

REM Try Python 3.11 first (required for Coqui TTS)
py -3.11 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Using Python 3.11...
    py -3.11 gui\tts_gui_orchard.py
    goto :end
)

REM Fall back to default Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11 from https://www.python.org/
    pause
    exit /b 1
)

echo Using default Python...
python gui\tts_gui_orchard.py

:end
if %errorlevel% neq 0 (
    echo.
    echo ERROR: GUI failed to start
    echo Make sure you have the required dependencies:
    echo   pip install TTS soundfile sounddevice scipy numpy pyttsx3
    pause
    exit /b 1
)
