#!/usr/bin/env python3
"""
Launch all three UIs:
- FastAPI (training visualization) on port 8000
- Streamlit (custom model LoRA) on port 8501  
- Streamlit (LLaMA LoRA) on port 8502
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def main():
    print("🚀 Starting Complete Training Suite...")
    print("=" * 50)
    
    # Change to src directory
    src_dir = Path(__file__).parent / "src"
    os.chdir(src_dir)
    
    processes = []
    
    # Start FastAPI server
    print("Starting FastAPI server on http://localhost:8000")
    fastapi_proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "web.app:app", 
        "--reload", 
        "--port", "8000"
    ])
    processes.append(("FastAPI", fastapi_proc))
    
    # Give FastAPI time to start
    time.sleep(3)
    
    # Start Streamlit app for custom models
    print("\nStarting Streamlit (Custom LoRA) on http://localhost:8501")
    streamlit_proc = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", 
        "web/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ])
    processes.append(("Streamlit Custom", streamlit_proc))
    
    # Give it time to start
    time.sleep(2)
    
    # Start Streamlit app for LLaMA
    print("\nStarting Streamlit (LLaMA LoRA) on http://localhost:8502")
    llama_proc = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", 
        "web/streamlit_llama.py",
        "--server.port", "8502",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ])
    processes.append(("Streamlit LLaMA", llama_proc))
    
    print("\n" + "=" * 50)
    print("All UIs are running!")
    print("\n📌 Available Interfaces:")
    print("  1. FastAPI (Training Visualization): http://localhost:8000")
    print("     - Base model training with real-time visualization")
    print("     - Architecture explorer")
    print("     - Attention pattern viewer")
    print("\n  2. Streamlit (Custom Model LoRA): http://localhost:8501")
    print("     - LoRA fine-tuning for custom BigramLM")
    print("     - Character-level tokenization")
    print("     - Good for learning/experimentation")
    print("\n  3. Streamlit (LLaMA LoRA): http://localhost:8502")
    print("     - LoRA fine-tuning for LLaMA/Mistral models")
    print("     - Supports 7B, 13B, 70B models")
    print("     - 8-bit/4-bit quantization")
    print("     - Production-ready fine-tuning")
    print("\n🛑 Press Ctrl+C to stop all servers")
    print("=" * 50)
    
    try:
        # Wait for interrupt
        while True:
            time.sleep(1)
            # Check if any process has died
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\n⚠️  {name} has stopped unexpectedly!")
                    
    except KeyboardInterrupt:
        print("\n\nShutting down servers...")
        for name, proc in processes:
            print(f"Stopping {name}...")
            proc.terminate()
        
        # Wait for processes to finish
        for name, proc in processes:
            proc.wait()
        
        print("All servers stopped. Goodbye!")

if __name__ == "__main__":
    main()