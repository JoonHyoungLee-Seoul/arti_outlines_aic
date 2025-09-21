# 🎨 Interactive Wireframe Portrait Demo

This interactive web demo showcases the complete **Hybrid PNG/SVG Architecture** for wireframe portrait generation, providing real-time control over all wireframe features and transparency settings.

## 🚀 Quick Start

1. **Start the demo server:**
   ```bash
   cd image_processing
   python demo_server_8081.py
   ```

2. **Access the demo:**
   - **URL**: `http://localhost:8081/wireframe_demo_working.html`
   - **Mobile-friendly**: Responsive design for desktop and mobile devices
   - **CORS-enabled**: Local file access with proper security headers

## 🎯 Features

### Portrait Selection
- 6 sample portraits from the Art Institute of Chicago collection
- High-quality foreground and background separated images
- Automatic loading of corresponding wireframe SVG files

### Interactive Wireframe Controls

#### Toggle Features (On/Off)
- **Construction Lines** - Classical portrait drawing guidelines based on facial landmarks
- **Face Mesh** - Detailed wireframe contours using MediaPipe connections  
- **DexiNed Outlines** - AI-powered edge detection outlines
- **Pose Landmarks** - Body skeleton wireframes (excluding face/hand details)

#### Transparency Sliders (0-100%)
- **Foreground Transparency** - Control opacity of the portrait subject
- **Background Transparency** - Control opacity of the background image

### Creative Use Cases

#### 🎨 Drawing Practice Mode
- Set **Foreground: 0%**, **Background: 100%**
- Creates clean background with white person silhouette for tracing

#### 🔍 Wireframe Analysis Mode  
- Set **Foreground: 100%**, **Background: 0%**
- Shows only wireframe overlay for studying facial structure

#### 🖼️ Artistic Overlay Mode
- Set **Foreground: 30%**, **Background: 80%**
- Subtle wireframe blend over background image

## 🏗️ Architecture

This demo implements the **Hybrid PNG/SVG Architecture** described in the main project:

- **PNG Raster Layers**: Background and foreground images for rich visual content
- **SVG Vector Overlays**: Wireframe elements for infinite scalability
- **Real-time Composition**: Dynamic layer blending with transparency controls
- **Web-Ready Format**: Standards-compliant output for frontend integration

## 📁 Production File Structure

```
image_processing/
├── wireframe_demo_working.html  # Interactive demo webpage (WORKING VERSION)
├── demo_server_8081.py         # Production HTTP server on port 8081 with CORS
├── out_sample/                 # Sample portrait dataset
│   ├── clipped_images_fg/     # Foreground portraits (6 samples)
│   └── clipped_images_bg/     # Background images (6 samples)
└── beginner_output_svg/       # Composite wireframe SVG files (PRODUCTION READY)
    ├── 15714_output.svg       # Complete wireframe composite with grouped layers
    ├── 16151_output.svg       # All wireframe components in single file
    └── ...                    # Additional portrait wireframes
```

### Composite SVG Architecture (CONFIRMED WORKING)

The `beginner_output_svg/` directory contains **composite SVG files** that work correctly in the demo:

- **Pattern**: `{portrait_id}_output.svg`
- **Architecture**: Single SVG file with grouped wireframe components:
  - `<g id="construction-lines">` - Classical drawing guidelines
  - `<g id="face-mesh">` - Detailed facial wireframes (468 landmarks)  
  - `<g id="pose-landmarks">` - Body skeleton wireframes (33 landmarks)
- **Group Control**: JavaScript controls visibility of each group within the composite SVG
- **Sample IDs**: 15714, 16151, 16281, 16298, 8104, 864

## 🔧 Technical Details

### Browser Compatibility
- Modern browsers with SVG and CSS3 support
- Responsive design for desktop and mobile
- CORS-enabled server for local file access

### Performance Features
- Lazy loading of images and SVG content
- Smooth transitions and animations
- Optimized layer composition for 60fps updates

### Customization
The demo code is fully customizable:
- Modify `wireframe_demo_working.html` for UI changes
- Adjust `demo_server_8081.py` for server configuration  
- Extend JavaScript for additional features
- Use composite SVG approach for proper wireframe alignment

## 🎮 Controls Reference

| Control | Type | Range | Description |
|---------|------|-------|-------------|
| Portrait Selection | Dropdown | 6 options | Choose from sample portraits |
| Construction Lines | Toggle | On/Off | Classical drawing guidelines |
| Face Mesh | Toggle | On/Off | Detailed facial wireframes |
| DexiNed Outlines | Toggle | On/Off | AI edge detection |
| Pose Landmarks | Toggle | On/Off | Body skeleton wireframes |
| Foreground Transparency | Slider | 0-100% | Portrait subject opacity |
| Background Transparency | Slider | 0-100% | Background image opacity |

## 💡 Tips

- **For Drawing Practice**: Turn off foreground, keep background visible
- **For Structure Study**: Turn off background, keep wireframes visible  
- **For Artistic Effect**: Blend both layers with partial transparency
- **For Web Integration**: Use the SVG overlay approach shown in the code

---

This demo showcases the powerful combination of traditional image processing with modern web technologies for creative portrait analysis and artistic applications.