#!/usr/bin/env python3
"""
Portrait Wireframe Generator Setup Script

Simple setup and verification script for the complete pipeline.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.10+"""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_conda():
    """Check if conda is available"""
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        print(f"✅ Conda available: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ Conda not found. Please install Miniconda or Anaconda.")
        return False

def create_environment():
    """Create conda environment"""
    print("\n🔧 Creating conda environment...")
    try:
        subprocess.run(['conda', 'env', 'create', '-f', 'environment.yml'], check=True)
        print("✅ Environment created successfully")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Environment may already exist or creation failed")
        return False

def verify_installation():
    """Verify key dependencies"""
    print("\n🔍 Verifying installation...")
    
    # Test imports
    test_imports = [
        ('mediapipe', 'MediaPipe'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('skimage', 'scikit-image')
    ]
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
        except ImportError:
            print(f"❌ {name} import failed")
            return False
    
    return True

def main():
    print("🎨 Portrait Wireframe Generator Setup")
    print("=" * 40)
    
    # Check prerequisites
    if not check_python_version():
        return 1
    
    if not check_conda():
        return 1
    
    # Create environment
    create_environment()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Activate environment: conda activate portrait_outline")
    print("2. Run complete pipeline: python scripts/run_complete_pipeline.py")
    print("3. Or run individual steps as described in README.md")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())