#!/usr/bin/env python3
"""
Complete Portrait Wireframe Generation Pipeline

This script runs the entire pipeline from downloading portraits to generating
wireframes and launching the demo.

Usage:
    python scripts/run_complete_pipeline.py [--skip-download] [--skip-segmentation]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description, cwd=None):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, cwd=cwd, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run complete portrait wireframe pipeline")
    parser.add_argument("--skip-download", action="store_true", 
                       help="Skip downloading portraits (use existing data)")
    parser.add_argument("--skip-segmentation", action="store_true",
                       help="Skip segmentation (use existing segmented images)")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of sample portraits to process (default: 5)")
    
    args = parser.parse_args()
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🎨 Portrait Wireframe Generator - Complete Pipeline")
    print("=" * 50)
    
    # Step 1: Download portraits
    if not args.skip_download:
        if not run_command(
            "python download_data/aic_portrait_paintings_downloader.py",
            "Downloading portrait dataset from Art Institute of Chicago"
        ):
            print("Failed to download portraits. Exiting.")
            return 1
    else:
        print("⏭️  Skipping portrait download")
    
    # Step 2: Segmentation
    if not args.skip_segmentation:
        if not run_command(
            "python image_processing/run_cutout.py -b download_data/aic_sample/images/",
            "Running BiRefNet segmentation on portraits"
        ):
            print("Failed to run segmentation. Exiting.")
            return 1
    else:
        print("⏭️  Skipping segmentation")
    
    # Step 3: Generate sample wireframes
    print(f"\n🎯 Processing {args.num_samples} sample portraits through wireframe pipeline")
    
    # Get available foreground images
    fg_dir = Path("image_processing/out/clipped_images_fg")
    if not fg_dir.exists():
        print("❌ No segmented images found. Please run segmentation first.")
        return 1
    
    fg_images = list(fg_dir.glob("*_fg.png"))[:args.num_samples]
    
    for i, fg_image in enumerate(fg_images, 1):
        image_id = fg_image.stem.replace("_fg", "")
        print(f"\n📷 Processing portrait {i}/{len(fg_images)}: {image_id}")
        
        # Generate wireframe
        cmd = f"""python image_processing/wireframe_portrait_processor.py {fg_image} \
  --construction-lines --mesh --pose-landmarks \
  --svg --svg-output image_processing/beginner_output_svg/{image_id}_output.svg \
  --background-merge \
  --foreground-dir image_processing/out/clipped_images_fg/ \
  --background-dir image_processing/out/clipped_images_bg/ \
  --foreground-transparency 100 --background-transparency 30 \
  -o image_processing/output/{image_id}_complete.png"""
        
        if not run_command(cmd, f"Generating wireframe for portrait {image_id}"):
            print(f"⚠️  Skipping {image_id} due to processing error")
            continue
        
        # Copy to demo directory
        fg_src = f"image_processing/out/clipped_images_fg/{image_id}_fg.png"
        bg_src = f"image_processing/out/clipped_images_bg/{image_id}_bg.png"
        fg_dst = f"image_processing/out_sample/clipped_images_fg/"
        bg_dst = f"image_processing/out_sample/clipped_images_bg/"
        
        run_command(f"cp {fg_src} {fg_dst}", f"Copying {image_id} to demo directory")
        run_command(f"cp {bg_src} {bg_dst}", f"Copying {image_id} background to demo directory")
    
    # Step 4: Launch demo
    print("\n🌐 Pipeline complete! Ready to launch interactive demo.")
    print("\nTo start the demo server:")
    print("    cd image_processing")
    print("    python demo_server_8081.py")
    print("\nThen open: http://localhost:8081/wireframe_demo_working.html")
    
    launch_demo = input("\nLaunch demo now? (y/N): ").lower().strip()
    if launch_demo == 'y':
        print("\n🚀 Starting demo server...")
        print("Press Ctrl+C to stop the server")
        try:
            subprocess.run("python image_processing/demo_server_8081.py", shell=True)
        except KeyboardInterrupt:
            print("\n👋 Demo server stopped.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())