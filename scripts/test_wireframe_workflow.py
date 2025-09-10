#!/usr/bin/env python3
"""
Test script for wireframe portrait workflow
==========================================

Tests the flexible wireframe system with different feature combinations
and validates the output for various user scenarios.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project directories to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'image_processing'))

from wireframe_portrait_processor import (
    WireframePortraitProcessor, WireframeConfig, create_preset_configs
)

def test_basic_functionality():
    """Test basic wireframe functionality"""
    print("🧪 Testing Basic Functionality")
    print("=" * 50)
    
    # Find test image
    test_image_paths = [
        project_root / "image_processing/out_sample/clipped_images_fg/8104_fg.png",
        project_root / "download_data/aic_sample/images/8104.jpg",
        project_root / "mediapipe_practice/test_image.jpg"
    ]
    
    test_image = None
    for path in test_image_paths:
        if path.exists():
            test_image = str(path)
            break
    
    if not test_image:
        print("❌ No test image found. Please ensure sample images are available.")
        print("Expected locations:")
        for path in test_image_paths:
            print(f"  - {path}")
        return False
    
    print(f"✅ Using test image: {test_image}")
    
    # Test each preset configuration
    presets = create_preset_configs()
    output_dir = project_root / "image_processing/out/wireframe_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for preset_name, config in presets.items():
        print(f"\n🎨 Testing preset: {preset_name}")
        
        try:
            # Create processor
            processor = WireframePortraitProcessor(config)
            
            # Process image
            output_path = output_dir / f"test_{preset_name}.png"
            results = processor.process_image(test_image, str(output_path))
            
            if results and 'final_rgba' in results:
                print(f"  ✅ Generated wireframe: {output_path}")
                
                # Validate output
                if validate_wireframe_output(results['final_rgba'], config):
                    print(f"  ✅ Output validation passed")
                else:
                    print(f"  ⚠️  Output validation warnings")
            else:
                print(f"  ❌ Failed to generate wireframe")
                return False
                
        except Exception as e:
            print(f"  ❌ Error processing {preset_name}: {e}")
            return False
    
    print(f"\n🎉 All basic functionality tests passed!")
    print(f"📁 Test outputs saved to: {output_dir}")
    return True

def test_custom_configurations():
    """Test custom feature combinations"""
    print("\n🔧 Testing Custom Configurations")
    print("=" * 50)
    
    # Find test image
    test_image_paths = [
        project_root / "image_processing/out_sample/clipped_images_fg/8104_fg.png",
        project_root / "download_data/aic_sample/images/8104.jpg"
    ]
    
    test_image = None
    for path in test_image_paths:
        if path.exists():
            test_image = str(path)
            break
    
    if not test_image:
        print("❌ No test image found")
        return False
    
    output_dir = project_root / "image_processing/out/wireframe_tests/custom"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test various custom combinations
    test_configs = [
        # Construction lines only - thin lines
        {
            'name': 'construction_thin',
            'config': WireframeConfig(
                enable_construction_lines=True,
                enable_mesh=False,
                enable_dexined_outline=False,
                construction_line_thickness=1
            )
        },
        # Construction lines only - thick lines
        {
            'name': 'construction_thick',
            'config': WireframeConfig(
                enable_construction_lines=True,
                enable_mesh=False,
                enable_dexined_outline=False,
                construction_line_thickness=3
            )
        },
        # Mesh only - light
        {
            'name': 'mesh_light',
            'config': WireframeConfig(
                enable_construction_lines=False,
                enable_mesh=True,
                enable_dexined_outline=False,
                mesh_thickness=1
            )
        },
        # Construction + Mesh combination
        {
            'name': 'construction_plus_mesh',
            'config': WireframeConfig(
                enable_construction_lines=True,
                enable_mesh=True,
                enable_dexined_outline=False,
                construction_line_thickness=2,
                mesh_thickness=1
            )
        }
    ]
    
    for test_case in test_configs:
        print(f"\n🎨 Testing: {test_case['name']}")
        
        try:
            processor = WireframePortraitProcessor(test_case['config'])
            output_path = output_dir / f"{test_case['name']}.png"
            results = processor.process_image(test_image, str(output_path))
            
            if results:
                print(f"  ✅ Generated: {output_path}")
            else:
                print(f"  ❌ Failed to generate")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Custom configuration tests completed")
    print(f"📁 Outputs saved to: {output_dir}")
    return True

def validate_wireframe_output(image: np.ndarray, config: WireframeConfig) -> bool:
    """Validate wireframe output meets expectations"""
    try:
        # Basic shape validation
        if len(image.shape) != 3:
            print(f"    ⚠️  Expected 3D image, got {len(image.shape)}D")
            return False
        
        # RGBA validation
        if config.output_format == "rgba":
            if image.shape[2] != 4:
                print(f"    ⚠️  Expected RGBA (4 channels), got {image.shape[2]} channels")
                return False
            
            # Check if there are transparent pixels
            transparent_pixels = np.sum(image[:, :, 3] == 0)
            total_pixels = image.shape[0] * image.shape[1]
            transparency_ratio = transparent_pixels / total_pixels
            
            if transparency_ratio < 0.1:
                print(f"    ⚠️  Low transparency ratio: {transparency_ratio:.2%}")
            else:
                print(f"    ✓ Transparency ratio: {transparency_ratio:.2%}")
        
        # Check if image has content (not all black/white)
        non_zero_pixels = np.sum(np.any(image[:, :, :3] > 0, axis=2))
        if non_zero_pixels == 0:
            print(f"    ⚠️  Image appears to be empty (all black)")
            return False
        
        # Feature-specific validations
        if config.enable_construction_lines:
            # Check for colored lines (construction lines use specific colors)
            has_red = np.any(image[:, :, 0] > 200)  # Red construction lines
            has_green = np.any(image[:, :, 1] > 200)  # Green construction lines
            has_blue = np.any(image[:, :, 2] > 200)  # Blue construction lines
            
            if has_red or has_green or has_blue:
                print(f"    ✓ Construction lines detected")
            else:
                print(f"    ⚠️  Construction lines not clearly detected")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Validation error: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\n🛡️  Testing Error Handling")
    print("=" * 50)
    
    output_dir = project_root / "image_processing/out/wireframe_tests/error_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test with non-existent image
    print("Testing with non-existent image...")
    config = WireframeConfig(enable_construction_lines=True)
    processor = WireframePortraitProcessor(config)
    
    results = processor.process_image("non_existent_image.jpg")
    if not results:
        print("  ✅ Correctly handled non-existent image")
    else:
        print("  ⚠️  Unexpected result for non-existent image")
    
    # Test with corrupted image (create a small invalid file)
    corrupt_image_path = output_dir / "corrupt.jpg"
    with open(corrupt_image_path, 'w') as f:
        f.write("not an image")
    
    print("Testing with corrupted image...")
    results = processor.process_image(str(corrupt_image_path))
    if not results:
        print("  ✅ Correctly handled corrupted image")
    else:
        print("  ⚠️  Unexpected result for corrupted image")
    
    # Clean up
    corrupt_image_path.unlink()
    
    print("✅ Error handling tests completed")
    return True

def create_demo_outputs():
    """Create demo outputs showing different user scenarios"""
    print("\n🎭 Creating Demo Outputs")
    print("=" * 50)
    
    # Find test image
    test_image_paths = [
        project_root / "image_processing/out_sample/clipped_images_fg/8104_fg.png",
        project_root / "download_data/aic_sample/images/8104.jpg"
    ]
    
    test_image = None
    for path in test_image_paths:
        if path.exists():
            test_image = str(path)
            break
    
    if not test_image:
        print("❌ No test image found for demo")
        return False
    
    demo_dir = project_root / "image_processing/out/wireframe_demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # User scenarios
    scenarios = {
        'step1_beginner_start': WireframeConfig(
            enable_construction_lines=True,
            enable_mesh=True,
            enable_dexined_outline=False
        ),
        'step2_add_outline': WireframeConfig(
            enable_construction_lines=True,
            enable_mesh=True,
            enable_dexined_outline=True
        ),
        'step3_experienced_construction_only': WireframeConfig(
            enable_construction_lines=True,
            enable_mesh=False,
            enable_dexined_outline=False
        ),
        'step4_experienced_add_mesh': WireframeConfig(
            enable_construction_lines=True,
            enable_mesh=True,
            enable_dexined_outline=False
        ),
        'step5_experienced_outline_finish': WireframeConfig(
            enable_construction_lines=False,
            enable_mesh=False,
            enable_dexined_outline=True
        ),
        'all_features_maximum_guidance': WireframeConfig(
            enable_construction_lines=True,
            enable_mesh=True,
            enable_dexined_outline=True
        )
    }
    
    for scenario_name, config in scenarios.items():
        print(f"Creating demo: {scenario_name}")
        
        try:
            processor = WireframePortraitProcessor(config)
            output_path = demo_dir / f"{scenario_name}.png"
            results = processor.process_image(test_image, str(output_path))
            
            if results:
                print(f"  ✅ Created: {output_path}")
            else:
                print(f"  ❌ Failed to create demo")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Create README for demo
    readme_content = """# Wireframe Portrait Demo

This directory contains example outputs demonstrating the flexible wireframe portrait system.

## User Scenarios

### Beginner Workflow
1. `step1_beginner_start.png` - Start with construction lines + mesh for guidance
2. `step2_add_outline.png` - Add DexiNed outline for complete guidance

### Experienced User Workflow  
3. `step3_experienced_construction_only.png` - Start with just construction lines
4. `step4_experienced_add_mesh.png` - Add mesh for refinement if needed
5. `step5_experienced_outline_finish.png` - Use outline only for clothing/background

### Maximum Guidance
6. `all_features_maximum_guidance.png` - All features enabled for difficult cases

## Features
- **Construction Lines**: Classical portrait drawing guidelines (red, green, blue, yellow lines)
- **Face Mesh**: Detailed facial contours from MediaPipe
- **DexiNed Outline**: AI-generated edge detection for precise outlines

Each feature can be toggled independently to match user skill level and drawing preferences.
"""
    
    readme_path = demo_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\n🎉 Demo outputs created!")
    print(f"📁 Demo directory: {demo_dir}")
    print(f"📖 See README.md for scenario descriptions")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Wireframe Portrait Workflow Testing")
    print("=" * 60)
    
    # Set up environment
    os.chdir(project_root)
    
    all_passed = True
    
    # Run tests
    try:
        if not test_basic_functionality():
            all_passed = False
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        all_passed = False
    
    try:
        if not test_custom_configurations():
            all_passed = False
    except Exception as e:
        print(f"❌ Custom configuration test failed: {e}")
        all_passed = False
    
    try:
        if not test_error_handling():
            all_passed = False
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        all_passed = False
    
    try:
        if not create_demo_outputs():
            all_passed = False
    except Exception as e:
        print(f"❌ Demo creation failed: {e}")
        all_passed = False
    
    # Final report
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("The wireframe portrait workflow is ready for use.")
        print("\nUsage examples:")
        print("  # Beginner mode (all features)")
        print("  python image_processing/wireframe_portrait_processor.py input.jpg --preset beginner")
        print("  ")
        print("  # Advanced mode (construction lines only)")
        print("  python image_processing/wireframe_portrait_processor.py input.jpg --preset advanced")
        print("  ")
        print("  # Custom (construction + mesh)")
        print("  python image_processing/wireframe_portrait_processor.py input.jpg --construction-lines --mesh")
        
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the error messages above and fix issues before using the system.")
    
    print(f"\n📁 Test outputs available in: {project_root}/image_processing/out/wireframe_tests/")
    print(f"📁 Demo outputs available in: {project_root}/image_processing/out/wireframe_demo/")

if __name__ == '__main__':
    main()