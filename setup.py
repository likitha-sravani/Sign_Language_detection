#!/usr/bin/env python3
"""
Setup script for ASL Sign Language Detection Project
"""

import subprocess
import sys
import os

def check_python_version():
    if sys.version_info < (3, 8):
        print("Warning: Python 3.8 or higher is recommended for this project.")
        return False
    return True

def check_pip():
    try:
        import pip
        return True
    except ImportError:
        print("Error: pip is not installed. Please install pip and rerun this script.")
        return False

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'tensorflow',
        'opencv-python',
        'numpy',
        'matplotlib',
        'pillow',
        'scikit-learn',
        'pandas',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("✓ All required packages are installed!")
        return True

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'models', 'src']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory already exists: {directory}")

def main():
    print("ASL Sign Language Detection - Setup")
    print("=" * 40)
    print(f"Current working directory: {os.getcwd()}")
    
    # Check Python version
    check_python_version()
    
    # Check pip
    if not check_pip():
        return
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Check dependencies
    print("\n2. Checking dependencies...")
    if not check_dependencies():
        print("\nInstalling missing dependencies...")
        if not install_requirements():
            print("Failed to install dependencies. Please install them manually:")
            print("pip install -r requirements.txt")
            return
    
    print("\n3. Setup complete!")
    print("\nNext steps:")
    print("1. Download the ASL Alphabet dataset from Kaggle")
    print("2. Extract it to the 'data/' folder")
    print("3. Run: python src/train.py (to train the model)")
    print("4. Run: python src/gui.py (to start the desktop app)")
    print("5. Or run: python src/detect.py (for command-line detection)")
    print("\nAll set! 🚀")

if __name__ == "__main__":
    main() 