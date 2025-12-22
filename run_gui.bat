@echo off
REM Re:speak TTS GUI Launcher (Basic)

echo Starting Re:speak TTS...

cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)

REM Check dependencies
python -c "import pyttsx3" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing basic dependencies...
    pip install pyttsx3
)

REM Launch the TTS GUI
python gui\tts_gui.py

if %errorlevel% neq 0 (
    echo ERROR: TTS GUI failed to start
    pause
    exit /b 1
)
