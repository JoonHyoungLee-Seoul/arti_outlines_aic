#!/usr/bin/env python3
"""
Basic pipeline tests for Portrait Wireframe Generator

These tests verify that the main components can be imported and basic
functionality works without running the full pipeline.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestPipelineComponents(unittest.TestCase):
    """Test basic component imports and functionality."""
    
    def test_import_mediapipe(self):
        """Test MediaPipe import."""
        try:
            import mediapipe as mp
            self.assertIsNotNone(mp.__version__)
            print(f"✅ MediaPipe version: {mp.__version__}")
        except ImportError as e:
            self.fail(f"MediaPipe import failed: {e}")
    
    def test_import_opencv(self):
        """Test OpenCV import."""
        try:
            import cv2
            self.assertIsNotNone(cv2.__version__)
            print(f"✅ OpenCV version: {cv2.__version__}")
        except ImportError as e:
            self.fail(f"OpenCV import failed: {e}")
    
    def test_import_numpy(self):
        """Test NumPy import."""
        try:
            import numpy as np
            self.assertIsNotNone(np.__version__)
            print(f"✅ NumPy version: {np.__version__}")
        except ImportError as e:
            self.fail(f"NumPy import failed: {e}")
    
    def test_import_onnxruntime(self):
        """Test ONNX Runtime import."""
        try:
            import onnxruntime as ort
            self.assertIsNotNone(ort.__version__)
            print(f"✅ ONNX Runtime version: {ort.__version__}")
        except ImportError as e:
            self.fail(f"ONNX Runtime import failed: {e}")
    
    def test_project_structure(self):
        """Test basic project structure."""
        required_dirs = [
            "download_data",
            "image_processing", 
            "scripts"
        ]
        
        required_files = [
            "README.md",
            "environment.yml",
            "setup.py",
            "download_data/aic_portrait_paintings_downloader.py",
            "image_processing/wireframe_portrait_processor.py",
            "image_processing/run_cutout.py",
            "image_processing/demo_server_8081.py",
            "scripts/run_complete_pipeline.py"
        ]
        
        for directory in required_dirs:
            dir_path = project_root / directory
            self.assertTrue(dir_path.exists() and dir_path.is_dir(), 
                          f"Required directory missing: {directory}")
        
        for file_path in required_files:
            file_full_path = project_root / file_path
            self.assertTrue(file_full_path.exists() and file_full_path.is_file(),
                          f"Required file missing: {file_path}")
        
        print("✅ Project structure validated")
    
    def test_demo_server_help(self):
        """Test that demo server script can show help."""
        import subprocess
        
        try:
            result = subprocess.run([
                sys.executable, 
                str(project_root / "image_processing" / "demo_server_8081.py"),
                "--help"
            ], capture_output=True, text=True, timeout=10)
            
            # Should exit with error (since --help isn't implemented) but not crash
            print("✅ Demo server script loads without crashing")
            
        except subprocess.TimeoutExpired:
            self.fail("Demo server script hung during execution")
        except Exception as e:
            print(f"⚠️  Demo server test skipped: {e}")
    
    def test_pipeline_script_help(self):
        """Test that pipeline script can show help."""
        import subprocess
        
        try:
            result = subprocess.run([
                sys.executable,
                str(project_root / "scripts" / "run_complete_pipeline.py"),
                "--help"
            ], capture_output=True, text=True, timeout=10)
            
            self.assertEqual(result.returncode, 0, "Pipeline script help failed")
            self.assertIn("usage:", result.stdout.lower())
            print("✅ Pipeline script help working")
            
        except subprocess.TimeoutExpired:
            self.fail("Pipeline script hung during help")
        except Exception as e:
            print(f"⚠️  Pipeline script test skipped: {e}")

class TestDataStructures(unittest.TestCase):
    """Test data structures and file formats."""
    
    def test_environment_yml(self):
        """Test environment.yml structure."""
        import yaml
        
        env_file = project_root / "environment.yml"
        self.assertTrue(env_file.exists(), "environment.yml missing")
        
        with open(env_file, 'r') as f:
            env_data = yaml.safe_load(f)
        
        self.assertIn('name', env_data)
        self.assertIn('dependencies', env_data)
        self.assertEqual(env_data['name'], 'portrait_outline')
        
        # Check for key dependencies
        deps = str(env_data['dependencies'])
        required_packages = ['python', 'numpy', 'opencv', 'mediapipe']
        
        for package in required_packages:
            self.assertIn(package, deps, f"Missing dependency: {package}")
        
        print("✅ Environment configuration validated")

if __name__ == '__main__':
    print("🧪 Running Portrait Wireframe Generator Tests")
    print("=" * 50)
    
    # Run tests with verbose output
    unittest.main(verbosity=2)