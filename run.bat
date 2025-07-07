@echo off
REM Run script for Transformer Training Interface on Windows

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Running setup first...
    python setup.py
    if errorlevel 1 (
        echo Setup failed. Please run 'python setup.py' manually.
        exit /b 1
    )
)

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Failed to activate virtual environment
    exit /b 1
)

REM Check if dependencies are installed
python -c "import torch" 2>nul
if errorlevel 1 (
    echo Dependencies not installed. Installing...
    pip install -r requirements.txt
)

REM Run the application
echo Starting Transformer Training Interface...
echo Open your browser at http://localhost:8000
python run_ui.py