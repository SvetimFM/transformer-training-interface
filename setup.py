#!/usr/bin/env python3
"""
Setup script for Transformer PCN UI
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.10 or higher"""
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} detected")

def create_virtual_environment():
    """Create a virtual environment if it doesn't exist"""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✓ Virtual environment created")
    else:
        print("✓ Virtual environment already exists")

def get_pip_command():
    """Get the correct pip command for the current OS"""
    if os.name == 'nt':  # Windows
        return os.path.join("venv", "Scripts", "pip")
    else:  # Unix-like
        return os.path.join("venv", "bin", "pip")

def install_dependencies():
    """Install required dependencies"""
    pip_cmd = get_pip_command()
    
    print("\nInstalling dependencies...")
    print("This may take a few minutes, especially for PyTorch...")
    
    # Upgrade pip first
    subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
    
    # Install requirements
    subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
    print("✓ Dependencies installed successfully")

def download_dataset():
    """Download the Shakespeare dataset if not present"""
    dataset_path = Path("shakespeare.txt")
    if not dataset_path.exists():
        print("\nDownloading Shakespeare dataset...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, "shakespeare.txt")
        print("✓ Dataset downloaded")
    else:
        print("✓ Dataset already exists")

def create_directories():
    """Create necessary directories"""
    dirs = ["checkpoints", "logs", "data"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print("✓ Created necessary directories")

def print_next_steps():
    """Print instructions for running the application"""
    print("\n" + "="*50)
    print("Setup completed successfully!")
    print("="*50)
    print("\nTo run the application:")
    
    if os.name == 'nt':  # Windows
        print("1. Activate the virtual environment:")
        print("   .\\venv\\Scripts\\activate")
        print("\n2. Run the application:")
        print("   python run_ui.py")
    else:  # Unix-like
        print("1. Activate the virtual environment:")
        print("   source venv/bin/activate")
        print("\n2. Run the application:")
        print("   python run_ui.py")
    
    print("\n3. Open your browser to http://localhost:8000")
    print("\nFor more information, see README.md")

def main():
    """Main setup function"""
    print("Transformer PCN UI Setup")
    print("========================\n")
    
    try:
        check_python_version()
        create_virtual_environment()
        install_dependencies()
        download_dataset()
        create_directories()
        print_next_steps()
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during setup: {e}")
        print("Please check the error messages above and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()