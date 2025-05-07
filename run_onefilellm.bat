@echo off
REM --- OneFileLLM Launcher (Windows) ---

REM IMPORTANT: Configure these paths if your setup differs!
REM Option 1: If venv is in the project directory:
SET "VENV_PATH=%~dp0.venv"
REM Option 2: If venv is elsewhere (example):
REM SET "VENV_PATH=C:\path\to\your\venvs\onefilellm_env"

REM Path to the onefilellm.py script (assuming it's in the same directory as this .bat file)
SET "SCRIPT_PATH=%~dp0onefilellm.py"

REM --- Activation and Execution ---
echo Activating virtual environment...
IF EXIST "%VENV_PATH%\Scripts\activate.bat" (
    CALL "%VENV_PATH%\Scripts\activate.bat"
) ELSE (
    echo [WARNING] Virtual environment activation script not found at "%VENV_PATH%\Scripts\activate.bat".
    echo Attempting to run directly. Ensure Python and dependencies are in your system PATH.
)

echo Running OneFileLLM...
python "%SCRIPT_PATH%" %*

echo.
echo OneFileLLM finished.
REM Deactivate is usually not strictly necessary here as the cmd window might close,
REM but can be added if desired:
REM IF EXIST "%VENV_PATH%\Scripts\deactivate.bat" (
REM CALL "%VENV_PATH%\Scripts\deactivate.bat"
REM )
pause