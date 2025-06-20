#!/usr/bin/env python3
"""
Launch both FastAPI (for training visualization) and Streamlit (for LoRA) UIs.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def main():
    print("🚀 Starting Transformer Training Suite...")
    print("=" * 50)
    
    # Change to src directory
    src_dir = Path(__file__).parent / "src"
    os.chdir(src_dir)
    
    # Start FastAPI server
    print("Starting FastAPI server on http://localhost:8000")
    fastapi_proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "web.app:app", 
        "--reload", 
        "--port", "8000"
    ])
    
    # Give FastAPI time to start
    time.sleep(3)
    
    # Start Streamlit app
    print("\nStarting Streamlit app on http://localhost:8501")
    streamlit_proc = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", 
        "web/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ])
    
    print("\n" + "=" * 50)
    print("Both UIs are running!")
    print("- FastAPI (Training): http://localhost:8000")
    print("- Streamlit (LoRA): http://localhost:8501")
    print("\nPress Ctrl+C to stop both servers")
    print("=" * 50)
    
    try:
        # Wait for interrupt
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down servers...")
        fastapi_proc.terminate()
        streamlit_proc.terminate()
        
        # Wait for processes to finish
        fastapi_proc.wait()
        streamlit_proc.wait()
        
        print("Servers stopped. Goodbye!")

if __name__ == "__main__":
    main()