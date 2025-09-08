# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Portrait Outline Generator project that creates artistic outlines from portrait images using computer vision and deep learning techniques. The system processes input portraits (JPG/PNG) and outputs:
- `*_construction.png` - Construction guides with center vertical line and eye height line
- `*_outline.png` - Clean outline drawings  
- `*_outline.svg` - Vector outlines (optional)

## Current Implementation Status

The project currently contains:
- **Image Clipping Tool** (`image_clipping/run_cutout.py`) - BiRefNet-based foreground/background separation with GPU acceleration
- **Data Download Tool** (`download_data/aic_portrait_paintings_downloader.py`) - Downloads public domain portrait paintings from Art Institute of Chicago API
- **Environment Setup Scripts** - Automated conda environment and tool installation

## Common Development Commands

### Environment Setup (One-time)
```bash
# Install tools and setup environment
./install_tools.sh

# Activate project environment  
source activate.sh
```

### Data Operations
```bash
# Download sample portrait data from AIC API
cd download_data
python aic_portrait_paintings_downloader.py

# Process single image (foreground/background separation)
python image_clipping/run_cutout.py -i path/to/image.jpg

# Batch process images
python image_clipping/run_cutout.py -b ./input_directory/
```

## Architecture

The intended pipeline-based architecture includes these main stages:

1. **Preprocessing** (OpenCV: resize, color convert, edge-friendly filtering) 
2. **Person Segmentation** (Currently: BiRefNet ONNX, Future: Mediapipe SelfieSeg)
3. **Edge Detection** (Planned: PiDiNet/DexiNed or Canny)
4. **Post-processing** (Planned: Morphology + small component removal)
5. **Contourization** (Planned: findContours)
6. **Polyline Simplification** (Planned: Douglas-Peucker)
7. **Curve Smoothing** (Planned: Chaikin)
8. **Guide Generation** (Planned: Face detection & eye keypoints via Mediapipe)

## Environment Management

The project uses conda environment isolation:
- **Environment name**: `portrait_outline`
- **Python version**: 3.10+
- **GPU Support**: CUDA/ROCm for BiRefNet ONNX inference
- **Key dependencies**: opencv-python, numpy, mediapipe, onnxruntime, scikit-image

All external tools (SuperClaude, etc.) are installed within the `portrait_outline` conda environment to prevent conflicts.

## Key Files and Directories

- `image_clipping/run_cutout.py` - Main image processing tool using BiRefNet ONNX model
- `image_clipping/models/BiRefNet-general-epoch_244.onnx` - Pre-trained segmentation model
- `download_data/aic_portrait_paintings_downloader.py` - Data acquisition from AIC API
- `Technical_Guideline.md` - Detailed technical specifications
- `requirements.txt` - Python package dependencies
- `.tools/` - Project-specific tool installations
- `out/` - Generated output files
- `src/stages/` - Planned pipeline implementation modules (future development)

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

## Current Limitations

- Full pipeline implementation is incomplete (only segmentation stage working)
- No edge detection, vectorization, or guide generation implemented yet
- Limited to BiRefNet segmentation model (Mediapipe integration planned)
- No automated testing framework

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
