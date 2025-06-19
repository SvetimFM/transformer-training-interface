#!/usr/bin/env python3
import uvicorn
import sys
import os

# Add src to path so imports work correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    print("Starting Transformer Training UI...")
    print("Open your browser at http://localhost:8000")
    
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["src"]
    )