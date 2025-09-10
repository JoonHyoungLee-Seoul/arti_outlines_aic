# 📋 SVG Export Quick Reference

Essential commands and code snippets for using the SVG wireframe export system.

## 🚀 Command Line Quick Start

```bash
# Basic SVG export with PNG
python wireframe_portrait_processor.py input.jpg --preset intermediate --svg --svg-output wireframe.svg -o wireframe.png

# SVG-only export
python wireframe_portrait_processor.py input.jpg --preset advanced --output-format svg -o wireframe.svg

# High-resolution SVG
python high_resolution_wireframe_processor.py input.jpg --target-resolution 3840x2160 --svg
```

## 🎨 Available Presets

| Preset | Construction Lines | Face Mesh | DexiNed Outline | Best For |
|--------|:------------------:|:---------:|:---------------:|----------|
| `beginner` | ✅ | ✅ | ✅ | Learning, complete guidance |
| `intermediate` | ✅ | ✅ | ❌ | Practice, balanced detail |
| `advanced` | ✅ | ❌ | ❌ | Experts, minimal guidelines |

## 💻 Python API

### Basic Usage
```python
from wireframe_portrait_processor import WireframeConfig, WireframePortraitProcessor

config = WireframeConfig(
    enable_construction_lines=True,
    enable_mesh=True,
    enable_svg_export=True,
    svg_output_path="output.svg"
)

processor = WireframePortraitProcessor(config)
results = processor.process_image("portrait.jpg", "wireframe.png")
svg_content = results['svg_content']
```

### Custom SVG Configuration
```python
from svg_generator import SVGWireframeConfig

# Print quality preset
svg_config = SVGWireframeConfig.print_quality_preset()

# Custom colors and thickness
svg_config.construction_lines['color'] = '#FF0000'
svg_config.face_mesh['color'] = '#00FF00'  
svg_config.construction_lines['thickness'] = 3
```

## 🌐 Web Integration

### HTML Embedding
```html
<!-- Direct SVG embedding -->
<div class="wireframe-viewer">
    <object data="wireframe.svg" type="image/svg+xml"></object>
</div>

<!-- Inline SVG for full control -->
<div class="wireframe-container">
    <!-- Paste SVG content here -->
</div>
```

### CSS Styling
```css
/* Responsive scaling */
.wireframe-container svg {
    width: 100%;
    height: auto;
    max-width: 600px;
}

/* Individual component styling */
#construction-lines line { stroke: #ff0000; stroke-width: 2px; }
#face-mesh line { stroke: #808080; opacity: 0.7; }

/* Animation */
#construction-lines line {
    stroke-dasharray: 1000;
    stroke-dashoffset: 1000;
    animation: draw 2s ease-in-out forwards;
}

@keyframes draw {
    to { stroke-dashoffset: 0; }
}
```

### JavaScript Interaction
```javascript
// Toggle wireframe layers
function toggleConstructionLines() {
    const group = document.getElementById('construction-lines');
    group.style.display = group.style.display === 'none' ? 'block' : 'none';
}

// Zoom functionality
function zoomWireframe(factor) {
    const svg = document.querySelector('.wireframe-container svg');
    const [x, y, w, h] = svg.getAttribute('viewBox').split(' ');
    const newW = w / factor;
    const newH = h / factor;
    const newX = x + (w - newW) / 2;
    const newY = y + (h - newH) / 2;
    svg.setAttribute('viewBox', `${newX} ${newY} ${newW} ${newH}`);
}

// Interactive highlighting
document.querySelectorAll('#face-mesh line').forEach(line => {
    line.addEventListener('mouseover', () => {
        line.setAttribute('stroke-width', '3');
        line.setAttribute('stroke', '#ff0000');
    });
    line.addEventListener('mouseout', () => {
        line.setAttribute('stroke-width', '1');
        line.setAttribute('stroke', '#808080');
    });
});
```

## 📱 React Component Example

```jsx
import React, { useState, useRef } from 'react';

const WireframeViewer = ({ svgContent }) => {
    const [zoom, setZoom] = useState(1);
    const svgRef = useRef(null);

    const handleZoom = (factor) => {
        setZoom(factor);
        if (svgRef.current) {
            const svg = svgRef.current;
            const viewBox = svg.getAttribute('viewBox').split(' ');
            const [x, y, w, h] = viewBox.map(Number);
            const newW = w / factor;
            const newH = h / factor;
            svg.setAttribute('viewBox', `${x} ${y} ${newW} ${newH}`);
        }
    };

    const toggleLayer = (layerId) => {
        const layer = svgRef.current?.getElementById(layerId);
        if (layer) {
            layer.style.display = layer.style.display === 'none' ? 'block' : 'none';
        }
    };

    return (
        <div className="wireframe-viewer">
            <div className="controls">
                <button onClick={() => handleZoom(0.5)}>50%</button>
                <button onClick={() => handleZoom(1)}>100%</button>
                <button onClick={() => handleZoom(2)}>200%</button>
                <button onClick={() => toggleLayer('construction-lines')}>
                    Toggle Guidelines
                </button>
                <button onClick={() => toggleLayer('face-mesh')}>
                    Toggle Mesh
                </button>
            </div>
            <div 
                ref={svgRef}
                dangerouslySetInnerHTML={{ __html: svgContent }}
                className="svg-container"
            />
        </div>
    );
};

export default WireframeViewer;
```

## 📊 File Size Guidelines

| Content | Typical Size | Use Case |
|---------|-------------|----------|
| Construction lines only | 1-2 KB | Minimal guidelines |
| Lines + Face mesh | 50-200 KB | Detailed wireframes |
| All features | 100-300 KB | Complete guidance |

## 🎯 Best Practices

### Performance
- Use `construction-lines` preset for fastest loading
- Enable compression for web delivery
- Consider lazy loading for multiple wireframes

### Quality
- Use high-resolution source images (1080p+)
- Ensure good face detection before processing
- Test SVG output across different browsers

### Accessibility  
- Include `<desc>` elements for screen readers
- Use semantic grouping with `<g>` elements
- Provide text alternatives for complex wireframes

## ⚠️ Troubleshooting

### Common Issues
```bash
# Face not detected
python wireframe_portrait_processor.py image.jpg --preset beginner
# Check console output for face detection status

# SVG not generating
python wireframe_portrait_processor.py image.jpg --preset intermediate --svg-output test.svg --output-format svg
# Verify SVG file is created

# Large file sizes
python wireframe_portrait_processor.py image.jpg --preset advanced --svg-output minimal.svg
# Use minimal preset to reduce file size
```

### Browser Compatibility
- ✅ Chrome/Edge: Full support
- ✅ Firefox: Full support  
- ✅ Safari: Full support
- ✅ Mobile: Responsive scaling works
- ✅ Print: Vector-perfect quality

---

*Quick reference for SVG wireframe integration* 🎨📋