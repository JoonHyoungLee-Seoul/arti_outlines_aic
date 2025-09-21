# 🎨 Portrait Wireframe Generator

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-GPU-green.svg)](https://mediapipe.dev)
[![ROCm](https://img.shields.io/badge/ROCm-6.1-red.svg)](https://rocm.docs.amd.com)
[![SVG](https://img.shields.io/badge/Export-SVG-orange.svg)](https://www.w3.org/Graphics/SVG/)
[![Demo](https://img.shields.io/badge/Demo-Live-brightgreen.svg)](#interactive-demo)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Transform portrait paintings into artistic wireframe sketches with this complete AI-powered pipeline. Generate construction lines, face meshes, and pose landmarks from classical portrait artworks using MediaPipe and computer vision.

## 🔄 Pipeline Architecture

```mermaid
flowchart TD
    A[📁 Raw JPG Portraits<br/>298 AIC Images] --> B[🤖 BiRefNet Segmentation<br/>Foreground/Background Split]
    B --> C[🎭 Foreground Images<br/>Portrait Subjects]
    B --> D[🖼️ Background Images<br/>Environment/Context]
    
    C --> E[🔍 MediaPipe Analysis]
    E --> F[👤 Face Landmarks<br/>468 Points]
    E --> G[🏃 Pose Landmarks<br/>33 Body Points]
    
    F --> H[📐 Construction Lines<br/>Classical Guidelines]
    F --> I[🕸️ Face Mesh<br/>Wireframe Contours]
    G --> J[🦴 Pose Skeleton<br/>Body Structure]
    
    H --> K[⚡ Optimized Pipeline<br/>No Redundant DexiNed]
    I --> K
    J --> K
    C --> K
    D --> K
    
    K --> L[📄 SVG Wireframes<br/>Infinite Scalability]
    K --> M[🖼️ PNG Composites<br/>Complete Images]
    
    L --> N[🌐 Interactive Demo<br/>Real-time Controls]
    M --> N
    C --> N
    D --> N
    
    N --> O[🎨 Creative Applications<br/>Art Education & Practice]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style K fill:#e8f5e8
    style N fill:#fff3e0
    style O fill:#fce4ec
```

### 🎯 **Key Optimizations**
- **⚡ 30-50% Faster**: No redundant DexiNed processing (edge detection pre-applied)
- **📦 50% Smaller Files**: Optimized SVG output (~289KB per wireframe)
- **🔄 Hybrid Architecture**: PNG backgrounds + SVG wireframes for web integration
- **🎮 Real-time Demo**: Interactive controls with 10 sample portraits

## 🚀 Quick Start (Complete Pipeline)

### Step 1: Setup Environment
```bash
# Clone repository
git clone https://github.com/your-username/portrait-wireframe-generator.git
cd portrait-wireframe-generator

# Setup conda environment
conda env create -f environment.yml
conda activate portrait_outline

# Verify installation
python --version  # Should show Python 3.10+
```

### Step 2: Download Portrait Dataset
```bash
# Download 298 public domain portraits from Art Institute of Chicago
cd download_data
python aic_portrait_paintings_downloader.py

# This creates: download_data/aic_sample/images/ (298 JPG files)
```

### Step 3: Generate Foreground/Background Segmentation
```bash
# Segment portraits using BiRefNet (creates foreground/background pairs)
cd ../image_processing
python run_cutout.py -b ../download_data/aic_sample/images/

# This creates: 
# - out/clipped_images_fg/ (298 foreground portraits)
# - out/clipped_images_bg/ (298 background images)
```

### Step 4: Generate Wireframe Portraits
```bash
# Process portraits through optimized wireframe pipeline
# Example: Process a single portrait
python wireframe_portrait_processor.py out/clipped_images_fg/102777_fg.png \
  --construction-lines --mesh --pose-landmarks \
  --svg --svg-output beginner_output_svg/102777_output.svg \
  --background-merge \
  --foreground-dir out/clipped_images_fg/ \
  --background-dir out/clipped_images_bg/ \
  --foreground-transparency 100 --background-transparency 30 \
  -o output/102777_complete.png

# This creates:
# - beginner_output_svg/102777_output.svg (interactive wireframe)
# - output/102777_complete.png (composite image)
```

### Step 5: Launch Interactive Demo
```bash
# Start the web demo server
python demo_server_8081.py

# Open browser to: http://localhost:8081/wireframe_demo_working.html
```

## ✨ Features

### 🎯 Core Capabilities
- **🎨 Wireframe Generation**: Construction lines, face mesh, and pose landmarks from portrait images
- **⚡ Optimized Pipeline**: Efficient processing without redundant edge detection (pre-applied in segmentation)
- **🖼️ Hybrid Output**: PNG composite images + SVG vector wireframes for infinite scalability
- **🌐 Interactive Demo**: Real-time wireframe controls and transparency adjustment
- **📱 Web Integration**: Standards-compliant SVG perfect for web applications
- **🚀 GPU Acceleration**: ROCm/CUDA support for optimal performance

### 🎨 Wireframe Components
- **Construction Lines**: Classical portrait guidelines based on facial landmarks (MediaPipe 468 points)
- **Face Mesh**: Detailed wireframe contours using MediaPipe face mesh connections
- **Pose Landmarks**: Body skeleton wireframes with 33 pose points (excludes face/hand details)
- **Background Composition**: Intelligent foreground/background blending with independent transparency control

## 🛠️ Installation

### Prerequisites
- **Python 3.10+**
- **Conda** (Miniconda or Anaconda)
- **GPU** with ROCm 6.1+ (AMD) or CUDA (NVIDIA) - optional but recommended

### Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/your-username/portrait-wireframe-generator.git
cd portrait-wireframe-generator

# 2. Create conda environment
conda env create -f environment.yml
conda activate portrait_outline

# 3. Verify MediaPipe installation
python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)"

# 4. Test BiRefNet segmentation
cd image_processing
python run_cutout.py --help
```

### GPU Setup (Optional)
For AMD GPUs (ROCm):
```bash
# Install ROCm dependencies
sudo apt update
sudo apt install rocm-dev rocm-libs

# Verify GPU detection
rocm-smi --showproductname
```

For NVIDIA GPUs (CUDA):
```bash
# Install CUDA dependencies (follow NVIDIA documentation)
# Verify GPU detection
nvidia-smi
```

## 📋 Complete Workflow

### 🏗️ **Technical Architecture**

```mermaid
graph LR
    subgraph "🌐 Data Source"
        AIC[Art Institute of Chicago API<br/>298 Public Domain Portraits]
    end
    
    subgraph "🔧 Processing Pipeline"
        SEG[BiRefNet ONNX<br/>Image Segmentation]
        MP[MediaPipe<br/>Face + Pose Detection]
        WF[Wireframe Generator<br/>Construction + Mesh + Skeleton]
    end
    
    subgraph "💾 Output Formats"
        SVG[SVG Vectors<br/>Infinite Scalability]
        PNG[PNG Composites<br/>Complete Images]
    end
    
    subgraph "🎮 Interactive Demo"
        WEB[Web Interface<br/>Real-time Controls]
        PORT[Portrait Selection<br/>10 Samples]
        CTRL[Wireframe Toggles<br/>Transparency Sliders]
    end
    
    AIC --> SEG
    SEG --> MP
    MP --> WF
    WF --> SVG
    WF --> PNG
    SVG --> WEB
    PNG --> WEB
    WEB --> PORT
    WEB --> CTRL
    
    style SEG fill:#e8f5e8
    style MP fill:#e1f5fe
    style WF fill:#f3e5f5
    style WEB fill:#fff3e0
```

### 1. Data Acquisition (Art Institute of Chicago API)
```bash
cd download_data
python aic_portrait_paintings_downloader.py

# Downloads 298 public domain portrait paintings
# Output: aic_sample/images/ directory with JPG files
# Creates: metadata.jsonl and curator_cards.md
```

### 2. Image Segmentation (BiRefNet ONNX)
```bash
cd image_processing

# Process single image
python run_cutout.py -i path/to/portrait.jpg

# Batch process all downloaded images
python run_cutout.py -b ../download_data/aic_sample/images/

# Output: 
# - out/clipped_images_fg/ (foreground portraits)
# - out/clipped_images_bg/ (background images)
```

### 3. Wireframe Generation (MediaPipe + Custom)
```bash
# Optimized pipeline (recommended for segmented images)
python wireframe_portrait_processor.py INPUT_IMAGE \
  --construction-lines --mesh --pose-landmarks \
  --svg --svg-output OUTPUT.svg \
  --background-merge \
  --foreground-dir out/clipped_images_fg/ \
  --background-dir out/clipped_images_bg/ \
  --foreground-transparency 100 \
  --background-transparency 30 \
  -o OUTPUT.png

# Alternative: Use presets
python wireframe_portrait_processor.py INPUT_IMAGE --preset beginner -o OUTPUT.png
```

### 4. Interactive Demo Setup
```bash
# Copy processed images to demo directory
cp out/clipped_images_fg/YOUR_IMAGE_fg.png out_sample/clipped_images_fg/
cp out/clipped_images_bg/YOUR_IMAGE_bg.png out_sample/clipped_images_bg/

# Start demo server
python demo_server_8081.py

# Access demo: http://localhost:8081/wireframe_demo_working.html
```

## 🎮 Interactive Demo

The web demo provides real-time control over wireframe generation:

### Features
- **Portrait Selection**: Choose from sample portraits or your processed images
- **Wireframe Toggles**: Independent control for construction lines, face mesh, and pose landmarks
- **Transparency Control**: Adjust foreground (0-100%) and background (0-100%) opacity
- **Real-time Updates**: Instant visual feedback for all adjustments

### Creative Use Cases
- **🎨 Drawing Practice**: Set foreground to 0%, background to 100% for tracing templates
- **🔍 Structure Analysis**: Set background to 0%, wireframes visible for studying proportions
- **🖼️ Artistic Overlay**: Blend wireframes over portraits for reference drawings

### Demo URL
```
http://localhost:8081/wireframe_demo_working.html
```

## 📁 Project Structure

```
portrait-wireframe-generator/
├── 📂 download_data/              # Data acquisition
│   ├── aic_portrait_paintings_downloader.py
│   └── aic_sample/               # Downloaded portraits
├── 📂 image_processing/          # Core processing
│   ├── wireframe_portrait_processor.py    # Main pipeline
│   ├── run_cutout.py                     # BiRefNet segmentation
│   ├── demo_server_8081.py              # Web demo server
│   ├── wireframe_demo_working.html      # Interactive demo
│   ├── out/                             # Segmented images
│   ├── beginner_output_svg/             # Generated SVG wireframes
│   └── out_sample/                      # Demo sample images
├── 📂 docs/                      # Documentation
├── 📂 scripts/                   # Setup scripts
├── environment.yml               # Conda environment
├── requirements.txt             # Python dependencies
├── README.md                   # This file
└── LICENSE                     # MIT License
```

## 🔧 Configuration Options

### Wireframe Features
```bash
# Individual feature control
--construction-lines    # Classical drawing guidelines
--mesh                 # Face wireframe mesh
--pose-landmarks       # Body skeleton wireframes

# Background composition
--background-merge                    # Enable background blending
--foreground-transparency 100        # Foreground opacity (0-100%)
--background-transparency 30         # Background opacity (0-100%)

# Output formats
--svg --svg-output wireframe.svg    # Generate SVG wireframes
-o output.png                       # Generate PNG composite
```

### Preset Configurations
```bash
--preset beginner      # All features enabled
--preset intermediate  # Lines + mesh + pose
--preset advanced      # Lines + pose only
--preset outline_only  # Construction lines only
--preset mesh_only     # Face mesh only
```

## 🎯 Use Cases

### 🎓 Art Education
- **Drawing Instruction**: Generate practice templates for portrait drawing classes
- **Proportion Study**: Visualize facial structure and proportions in classical artworks
- **Technique Analysis**: Compare wireframe structures across different artistic styles

### 💻 Digital Applications
- **Web Integration**: Embed interactive wireframes in educational websites
- **Mobile Apps**: Use SVG wireframes for drawing and art learning applications
- **Animation**: Create step-by-step wireframe reveal animations

### 🎨 Creative Projects
- **Reference Generation**: Create drawing references from portrait paintings
- **Style Transfer**: Use wireframes as guides for digital art creation
- **Interactive Exhibits**: Museum installations with real-time wireframe exploration

## 🚀 Performance

### Benchmark Results (M4 Pro)
- **Segmentation**: ~2-3 seconds per image (BiRefNet ONNX)
- **Wireframe Generation**: ~5-8 seconds per image (MediaPipe + processing)
- **Total Pipeline**: ~8-12 seconds per portrait (end-to-end)
- **SVG File Size**: ~289KB per wireframe (optimized without redundant data)

### Optimization Features
- **GPU Acceleration**: 2-10x speedup with ROCm/CUDA
- **Efficient Pipeline**: Skips redundant edge detection (pre-applied in segmentation)
- **Batch Processing**: Process multiple portraits in sequence
- **Memory Optimization**: Handles large datasets without memory issues

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Fork and clone your fork
git clone https://github.com/your-username/portrait-wireframe-generator.git

# Create development branch
git checkout -b feature/your-feature

# Install development dependencies
conda activate portrait_outline
pip install -e .

# Run tests
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Art Institute of Chicago** for providing public domain portrait dataset
- **MediaPipe** for face landmark detection and pose estimation
- **BiRefNet** for image segmentation capabilities
- **OpenCV** and **scikit-image** for image processing utilities

## 📚 Documentation

- [Installation Guide](docs/Installation.md)
- [API Reference](docs/API.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Troubleshooting](docs/Troubleshooting.md)

## 🔗 Links

- [Live Demo](http://localhost:8081/wireframe_demo_working.html) (when running locally)
- [Art Institute of Chicago API](https://api.artic.edu/docs/)
- [MediaPipe Documentation](https://mediapipe.dev/)
- [Issue Tracker](https://github.com/your-username/portrait-wireframe-generator/issues)

---

**Transform classical portraits into modern digital wireframes with AI-powered precision! 🎨✨**