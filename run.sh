#!/bin/bash
# Run script for Transformer Training Interface

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup first..."
    python3 setup.py
    if [ $? -ne 0 ]; then
        echo "Setup failed. Please run 'python3 setup.py' manually."
        exit 1
    fi
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Failed to activate virtual environment"
    exit 1
fi

# Check if dependencies are installed
if ! python -c "import torch" 2>/dev/null; then
    echo "Dependencies not installed. Installing..."
    pip install -r requirements.txt
fi

# Run the application
echo "Starting Transformer Training Interface..."
echo "Open your browser at http://localhost:8000"
python run_ui.py