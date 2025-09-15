# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Now, the hardware is AMD environment and has **Rocm**. Please set this environment as priority. However, when you create the code please make it flexible so it can work with **cuda** and **mac**. 

## Project Overview

This is a Portrait Outline Generator project that creates artistic wireframe portraits from portrait images using computer vision and deep learning techniques. The system processes input portraits (JPG/PNG) and outputs:
- `*_construction.png` - Construction guides with facial landmarks-based guidelines
- `*_mesh.png` - Face mesh wireframes with detailed contours
- `*_outline.png` - DexiNed-processed edge outlines
- `*.svg` - **Scalable vector graphics with infinite zoom capability** (NEW)
- High-resolution outputs (4K, 8K, print quality)

## Current Implementation Status ✅

**Completed Milestones:**
- **✅ Data Acquisition Pipeline** - Successfully downloaded 298 public domain portrait paintings from Art Institute of Chicago API with comprehensive metadata (160KB curator cards, 225KB metadata.jsonl)
- **✅ Image Segmentation Pipeline** - Implemented BiRefNet ONNX-based foreground/background separation with GPU acceleration, processing complete with 596 output files (298 originals → 298 foreground + 298 background images)
- **✅ Environment Setup** - Fully configured conda environment with all dependencies including onnxruntime, opencv-python, and BiRefNet model integration
- **✅ GPU Acceleration Setup** - Complete virtual display and ROCm GPU acceleration configuration for MediaPipe face landmark detection with 2-10x performance improvement
- **✅ MediaPipe Integration** - Face landmark detection working with proper RGBA transparency handling and GPU acceleration support
- **✅ Wireframe Portrait System** - Complete flexible wireframe generation system with toggleable features (construction lines, face mesh, DexiNed outlines)
- **✅ SVG Export System** - Scalable vector graphics export with infinite zoom capability, web-ready format for frontend integration
- **✅ High-Resolution Support** - 4K, 8K, and print quality (A4 300DPI) wireframe processing with adaptive scaling
- **✅ MediaPipe Pose Landmarker Integration** - Body skeleton detection with 33 pose landmarks, filtered to exclude face/hand details (landmarks 0-10, 17-22)

**Active Components:**
- **Wireframe Portrait Processor** (`image_processing/wireframe_portrait_processor.py`) - Main system for generating wireframe portraits with configurable features
- **High-Resolution Processor** (`image_processing/high_resolution_wireframe_processor.py`) - 4K/8K/print quality wireframe generation with adaptive scaling
- **SVG Generator** (`image_processing/svg_generator.py`) - Scalable vector graphics export with infinite zoom capability
- **Image Segmentation Tool** (`image_processing/run_cutout.py`) - Production-ready BiRefNet segmentation with batch processing capabilities
- **Data Download Tool** (`download_data/aic_portrait_paintings_downloader.py`) - Automated portrait data acquisition from AIC API
- **MediaPipe Face Landmarks** (`mediapipe_practice/face_landmark.ipynb`) - GPU-accelerated face landmark detection with RGBA transparency support
- **GPU Acceleration Scripts** - Complete virtual display and ROCm configuration for optimal performance

## Common Development Commands

### Environment Setup (One-time)
```bash
# Install tools and setup environment
./scripts/setup/install_tools.sh

# Activate project environment  
source activate.sh

# Setup GPU acceleration (one-time)
./scripts/setup/install_gpu_support.sh
```

### GPU Acceleration (for MediaPipe)
```bash
# Start virtual display server for GPU acceleration
./scripts/gpu/start_virtual_display.sh

# Setup GPU environment variables
source scripts/runtime/setup_gpu_env.sh

# Verify GPU acceleration
rocm-smi --showproductname --showtemp
```

### Wireframe Portrait Generation
```bash
# Generate wireframe portrait with preset configuration
cd image_processing
python wireframe_portrait_processor.py input.jpg --preset beginner -o output.png

# Generate with SVG export for web integration
python wireframe_portrait_processor.py input.jpg --preset intermediate --svg --svg-output output.svg -o output.png

# SVG-only output for frontend applications
python wireframe_portrait_processor.py input.jpg --preset advanced --output-format svg -o output.svg

# High-quality SVG with DexiNed outline (production ready)
python wireframe_portrait_processor.py input.jpg --preset beginner --svg --svg-output high_quality.svg -o output.png

# High-resolution processing (4K, 8K, print quality)
python high_resolution_wireframe_processor.py input.jpg --target-resolution 3840x2160 --preset beginner

# Custom feature configuration
python wireframe_portrait_processor.py input.jpg --construction-lines --mesh --dexined --pose-landmarks -o custom.png

# Pose landmarks only for body skeleton analysis
python wireframe_portrait_processor.py input.jpg --pose-landmarks -o skeleton.png
```

### Data Operations
```bash
# Download sample portrait data from AIC API
cd download_data
python aic_portrait_paintings_downloader.py
# ✅ Status: Complete (298 images, 160KB curator cards, 225KB metadata.jsonl)

# Process single image (foreground/background separation)
python image_processing/run_cutout.py -i path/to/image.jpg

# Batch process images (✅ Completed: 298 → 596 outputs)
cd image_processing
python run_cutout.py -b ../download_data/aic_sample/images/
# Outputs: ./out/foreground/ and ./out/background/ directories

# Face landmark detection with GPU acceleration
cd mediapipe_practice
jupyter notebook face_landmark.ipynb
# Requires: Virtual display running + GPU environment setup
```

## Architecture

The wireframe portrait generation system follows a modular, feature-based architecture:

### 🎯 **Wireframe Generation Pipeline**
1. **Face Detection** (MediaPipe: GPU-accelerated landmark detection with 468 facial points)
2. **Pose Detection** (MediaPipe: Body skeleton detection with 33 pose landmarks, filtered for body focus)
3. **Construction Lines** (Classical portrait guidelines based on actual facial landmarks)
4. **Face Mesh** (Detailed wireframe contours using MediaPipe connections)
5. **Pose Skeleton** (Body structure wireframes excluding face and hand details)
6. **Edge Detection** (DexiNed AI-powered outline generation with ROCm acceleration)
7. **Vector Export** (SVG generation with infinite scalability and web integration)
8. **High-Resolution Scaling** (Adaptive processing for 4K, 8K, and print quality)

### 🔧 **System Components**
- **Configurable Features**: Toggle construction lines, face mesh, pose landmarks, and outlines independently
- **Preset System**: Beginner (all features), intermediate (lines + mesh + pose), advanced (lines + pose only)
- **Multi-Format Output**: PNG, SVG, high-resolution variants
- **GPU Acceleration**: ROCm/CUDA support for MediaPipe and DexiNed processing
- **Web Integration**: Standards-compliant SVG with zoom, animation, and interaction support

## Environment Management

The project uses conda environment isolation:
- **Environment name**: `portrait_outline`
- **Python version**: 3.10+
- **GPU Support**: CUDA/ROCm for BiRefNet ONNX inference
- **Key dependencies**: opencv-python, numpy, mediapipe, onnxruntime, scikit-image

All external tools (SuperClaude, etc.) are installed within the `portrait_outline` conda environment to prevent conflicts.

## Key Files and Directories

### 🎨 **Wireframe System**
- `image_processing/wireframe_portrait_processor.py` - Main wireframe generation system with preset configurations
- `image_processing/high_resolution_wireframe_processor.py` - 4K/8K/print quality processing with adaptive scaling
- `image_processing/svg_generator.py` - SVG export system with infinite zoom and web integration
- `image_processing/SVG_EXPORT_DOCUMENTATION.md` - Comprehensive SVG integration guide

### 🔧 **Core Components**
- `image_processing/run_cutout.py` - BiRefNet ONNX segmentation tool for background removal
- `image_processing/models/BiRefNet-general-epoch_244.onnx` - Pre-trained segmentation model
- `mediapipe_practice/face_landmark.ipynb` - GPU-accelerated face landmark detection notebook
- `mediapipe_practice/face_landmarker.task` - MediaPipe face landmark model

### 📁 **Data and Assets**
- `download_data/aic_portrait_paintings_downloader.py` - Data acquisition from Art Institute of Chicago API
- `download_data/aic_sample/` - Portrait dataset (298 public domain paintings)
- `out/` - Generated wireframe outputs (PNG, SVG, high-resolution)
- `requirements.txt` - Python package dependencies

### GPU Acceleration Scripts
- `scripts/setup/install_gpu_support.sh` - One-time GPU dependencies installation
- `scripts/gpu/start_virtual_display.sh` - Virtual display server for GPU acceleration  
- `scripts/gpu/stop_virtual_display.sh` - Clean shutdown of virtual display
- `scripts/runtime/setup_gpu_env.sh` - GPU environment variables configuration
- `docs/GPU_Setup.md` - Complete GPU setup documentation

## Development Constraints

- **Original form fidelity** - Maintain faithful representation of portrait features
- **Commercial license safety** - All components must be commercially safe (public domain data, open-source models)
- **GPU optimization** - Leverage CUDA/ROCm when available for performance
- **Cross-platform compatibility** - Support Linux/Windows environments

## Data Source

Portrait images are sourced from the Art Institute of Chicago API with strict filtering:
- Only public domain artworks (`is_public_domain = True`)
- Paintings with portrait subjects
- Available high-resolution images via IIIF
- Comprehensive metadata for curatorial context

## Current Status & Future Enhancements

### ✅ **Production Ready**
- Complete wireframe portrait generation system with preset configurations
- SVG export with infinite scalability for web/mobile applications
- **Enhanced DexiNed SVG Quality**: Production-grade edge outline processing with 300-700+ contours
- High-resolution processing (4K, 8K, print quality) with adaptive scaling
- GPU-accelerated processing with ROCm/CUDA support
- Landmark-accurate construction lines based on MediaPipe face detection

### 🔮 **Future Enhancements**
- **Interactive Web Components**: React/Vue/Angular components for wireframe viewers
- **Animation Presets**: CSS/SVG animations for progressive wireframe reveals  
- **Batch Processing**: Multi-image wireframe generation with progress tracking
- **API Server**: REST/GraphQL endpoints for wireframe generation services
- **Mobile Optimization**: Lightweight processing for mobile device integration

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
