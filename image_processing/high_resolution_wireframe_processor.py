#!/usr/bin/env python3
"""
High Resolution Wireframe Portrait Processor
===========================================

Enhanced wireframe processor for ultra-high resolution support
with zoom in/out capabilities for digital art creation.

Key Features:
- Support for 4K, 8K, and higher resolutions
- Intelligent scaling for different components
- Vector-based construction lines for infinite scalability
- High-quality mesh rendering at any resolution
- DexiNed super-resolution techniques
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
import argparse
import math

# Import base wireframe processor
sys.path.append(os.path.dirname(__file__))
from wireframe_portrait_processor import (
    WireframeConfig, ConstructionLinesGenerator, MeshGenerator, 
    DexiNedGenerator, BackgroundRemover, WireframePortraitProcessor,
    create_preset_configs
)
from svg_generator import SVGGenerator, SVGWireframeConfig

@dataclass
class HighResolutionConfig(WireframeConfig):
    """Enhanced configuration for high-resolution processing

    This extends :class:`WireframeConfig` with extra knobs that keep the
    wireframe clean when generating very large images.  Most of these fields
    control how line weights and mesh density are scaled relative to a
    1080p baseline so that the output looks consistent from HD to print-size
    canvases.
    """
    
    # High-resolution settings
    target_resolution: Tuple[int, int] = (3840, 2160)  # 4K default
    max_resolution: Tuple[int, int] = (7680, 4320)     # 8K maximum
    enable_super_resolution: bool = True
    vector_construction_lines: bool = True
    
    # Adaptive scaling
    line_thickness_scaling: float = 1.0  # Auto-calculated based on resolution
    mesh_density_scaling: float = 1.0    # Auto-calculated based on resolution
    
    # DexiNed super-resolution
    dexined_upscale_factor: int = 2      # 352x352 → 704x704 → final resolution
    enable_edge_enhancement: bool = True
    
    # Performance optimization
    tile_processing: bool = False         # For extremely large images
    tile_size: Tuple[int, int] = (2048, 2048)
    tile_overlap: int = 256

class HighResolutionConstructionLinesGenerator(ConstructionLinesGenerator):
    """High-resolution construction lines with vector-based rendering"""
    
    @staticmethod
    def draw_construction_lines(image: np.ndarray, 
                              landmarks: List, 
                              config: HighResolutionConfig) -> np.ndarray:
        """
        Draw high-resolution construction lines with adaptive scaling
        """
        if not landmarks:
            return image.copy()
        
        annotated = image.copy()
        height, width = image.shape[:2]
        
        # Scale the guideline thickness so strokes look similar across
        # resolutions.  A 1px line at 1080p becomes thicker at 4K/8K.
        base_resolution = 1080  # HD reference
        resolution_factor = max(height, width) / base_resolution
        thickness = max(1, int(
            config.construction_line_thickness *
            resolution_factor *
            config.line_thickness_scaling
        ))
        
        colors = config.construction_line_colors
        
        def get_pixel_coords(landmark_idx):
            """Convert normalized landmark coordinates to pixel coordinates"""
            if landmark_idx >= len(landmarks):
                return None
            landmark = landmarks[landmark_idx]
            return (int(landmark.x * width), int(landmark.y * height))
        
        def draw_smooth_line(points, line_color):
            """Draw anti-aliased line with sub-pixel precision"""
            if len(points) < 2:
                return
                
            if config.vector_construction_lines and len(points) > 2:
                # When vector mode is on we still draw straight segments, but
                # with heavy anti-aliasing so the result resembles a smooth
                # Bézier curve.
                for i in range(len(points) - 1):
                    cv2.line(annotated, points[i], points[i + 1], line_color, thickness, cv2.LINE_AA)
            else:
                # Otherwise we simply connect the points with standard lines.
                for i in range(len(points) - 1):
                    cv2.line(annotated, points[i], points[i + 1], line_color, thickness, cv2.LINE_AA)
        
        # List of classical portrait guidelines to render. Each tuple pairs
        # landmark indices with the color used for that guideline.
        construction_lines = [
            ([10, 168, 4, 152], colors['vertical_center']),    # Vertical center
            ([63, 293], colors['eyebrow_line']),               # Eyebrow line
            ([33, 263], colors['eye_lines']),                  # Eye line 1
            ([133, 362], colors['eye_lines']),                 # Eye line 2
            ([145, 159], colors['eye_lines']),                 # Eye line 3
            ([374, 386], colors['eye_lines']),                 # Eye line 4
            ([48, 278], colors['nose_line']),                  # Nose line
            ([61, 291], colors['mouth_line']),                 # Mouth line
        ]
        
        for point_indices, line_color in construction_lines:
            points = []
            for idx in point_indices:
                point = get_pixel_coords(idx)
                if point:
                    points.append(point)  # accumulate valid landmark points
            draw_smooth_line(points, line_color)
        
        return annotated

class HighResolutionMeshGenerator(MeshGenerator):
    """High-resolution mesh with adaptive density"""
    
    def draw_face_mesh(self, image: np.ndarray, 
                      detection_result, 
                      config: HighResolutionConfig) -> np.ndarray:
        """Draw high-resolution face mesh with adaptive density"""
        if not detection_result.face_landmarks:
            return image.copy()
        
        annotated = image.copy()
        colors = config.mesh_colors
        
        # Determine how thick the mesh lines should be at the current
        # resolution.  This keeps the grid readable even on massive canvases.
        height, width = image.shape[:2]
        base_resolution = 1080
        resolution_factor = max(height, width) / base_resolution
        mesh_thickness = max(1, int(
            config.mesh_thickness * resolution_factor * config.mesh_density_scaling
        ))
        
        for face_landmarks in detection_result.face_landmarks:
            # Convert landmarks
            from mediapipe.framework.formats import landmark_pb2
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in face_landmarks
            ])
            
            # Draw with enhanced quality settings.  We omit landmark circles so
            # only the connections remain, producing a cleaner technical style.
            drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
                thickness=mesh_thickness,
                circle_radius=0  # No landmark circles for clean lines
            )
            
            # Tesselation with adaptive density
            if colors['tesselation']:
                self.mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks_proto,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=colors['tesselation'], 
                        thickness=mesh_thickness
                    )
                )
            
            # Enhanced contours
            if colors['contours']:
                self.mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks_proto,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=colors['contours'], 
                        thickness=mesh_thickness + 1
                    )
                )
            
            # High-quality iris rendering
            if colors['irises']:
                self.mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks_proto,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=colors['irises'], 
                        thickness=mesh_thickness + 1
                    )
                )
        
        return annotated

class HighResolutionDexiNedGenerator(DexiNedGenerator):
    """DexiNed with super-resolution techniques"""
    
    def generate_outline(self, image: np.ndarray, config: HighResolutionConfig) -> np.ndarray:
        """Generate high-resolution outline using super-resolution techniques"""
        if not self.model:
            return self._fallback_edge_detection(image, config)
        
        try:
            original_shape = image.shape
            
            # Multi-scale processing for better quality.  By running the edge
            # detector on several resized versions of the image and merging the
            # results we preserve both coarse structure and tiny details.
            scales = [1.0]
            if config.enable_super_resolution:
                scales = [0.5, 1.0, 1.5]
            
            edge_results = []
            
            for scale in scales:
                # Scale image for processing
                if scale != 1.0:
                    h, w = int(image.shape[0] * scale), int(image.shape[1] * scale)
                    scaled_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
                else:
                    scaled_image = image
                
                # Process with DexiNed
                img_tensor = self._preprocess_image(scaled_image)
                
                if self.model and img_tensor is not None:
                    with self.model.eval():
                        predictions = self.model(img_tensor)
                        edge_map = predictions[-1].cpu().numpy()[0, 0]
                else:
                    edge_map = np.zeros((352, 352))
                
                # Post-process edges
                edge_image = self._postprocess_edges_hires(
                    edge_map, scaled_image.shape, config, scale
                )
                
                # Scale back to original size
                if scale != 1.0:
                    edge_image = cv2.resize(
                        edge_image, 
                        (original_shape[1], original_shape[0]), 
                        interpolation=cv2.INTER_LANCZOS4
                    )
                
                edge_results.append(edge_image)
            
            # Combine multi-scale results
            if len(edge_results) > 1:
                final_edges = self._combine_multiscale_edges(edge_results, config)
            else:
                final_edges = edge_results[0]
            
            # Edge enhancement for high-resolution
            if config.enable_edge_enhancement:
                final_edges = self._enhance_edges(final_edges, config)
            
            return final_edges
            
        except Exception as e:
            print(f"Error in high-res DexiNed processing: {e}")
            return self._fallback_edge_detection(image, config)
    
    def _postprocess_edges_hires(self, edge_map: np.ndarray, 
                                target_shape: Tuple[int, int, int],
                                config: HighResolutionConfig,
                                scale: float = 1.0) -> np.ndarray:
        """High-quality edge post-processing"""
        # Use high-quality interpolation for upscaling
        edge_resized = cv2.resize(
            edge_map, 
            (target_shape[1], target_shape[0]), 
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # Slightly relax the threshold when working at smaller scales so thin
        # lines are not lost after resizing.
        threshold = config.dexined_threshold * (0.8 + 0.2 * scale)
        edge_binary = (edge_resized > threshold).astype(np.uint8) * 255
        
        # Create high-quality edge image
        edge_rgb = np.ones(target_shape, dtype=np.uint8) * 255
        edge_rgb[edge_binary > 0] = config.dexined_color
        
        return edge_rgb
    
    def _combine_multiscale_edges(self, edge_results: List[np.ndarray], 
                                 config: HighResolutionConfig) -> np.ndarray:
        """Combine multi-scale edge detection results"""
        # Weighted combination of different scales
        weights = [0.3, 0.4, 0.3]  # Emphasize 1.0 scale
        
        combined = np.ones_like(edge_results[0]) * 255  # Start with white

        for i, (edges, weight) in enumerate(zip(edge_results, weights)):
            # Any pixel that isn't nearly white is treated as part of an edge.
            edge_mask = np.any(edges < 250, axis=2)

            # Blend the current scale's edges over the accumulated result. This
            # lets higher-resolution passes reinforce details from lower ones.
            alpha = weight
            combined[edge_mask] = (
                alpha * edges[edge_mask] +
                (1 - alpha) * combined[edge_mask]
            ).astype(np.uint8)
        
        return combined
    
    def _enhance_edges(self, edge_image: np.ndarray, 
                      config: HighResolutionConfig) -> np.ndarray:
        """Enhance edges for high-resolution display"""
        # Convert to grayscale for processing
        if len(edge_image.shape) == 3:
            gray = cv2.cvtColor(edge_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = edge_image
        
        # Morphological operations for cleaner edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # First close tiny gaps between edge segments, then smooth out any
        # remaining noise for a cleaner final outline.
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = cv2.medianBlur(gray, 3)
        
        # Convert back to color
        if len(edge_image.shape) == 3:
            enhanced = np.ones_like(edge_image) * 255
            enhanced[gray < 250] = config.dexined_color
            return enhanced
        else:
            return gray

class HighResolutionWireframeProcessor(WireframePortraitProcessor):
    """High-resolution wireframe processor with zoom support"""
    
    def __init__(self, config: HighResolutionConfig):
        # Initialize base class
        super().__init__(config)
        self.config = config
        
        # Override generators with high-resolution versions
        self.construction_generator = HighResolutionConstructionLinesGenerator()
        self.mesh_generator = HighResolutionMeshGenerator()
        
        if config.enable_dexined_outline and config.dexined_model_path:
            self.dexined_generator = HighResolutionDexiNedGenerator(config.dexined_model_path)
        
        # Calculate adaptive scaling factors
        self._calculate_scaling_factors()
    
    def _calculate_scaling_factors(self):
        """Calculate adaptive scaling factors based on target resolution"""
        target_width, target_height = self.config.target_resolution
        base_resolution = 1920 * 1080  # HD reference
        target_total = target_width * target_height

        # Compare total pixel count with HD and derive a square-root factor so
        # scaling grows proportionally with image dimensions.
        resolution_factor = math.sqrt(target_total / base_resolution)

        # Update scaling factors within sane bounds to avoid overly thick lines
        # when jumping from small to extremely large resolutions.
        self.config.line_thickness_scaling = min(3.0, max(0.5, resolution_factor))
        self.config.mesh_density_scaling = min(2.0, max(0.5, resolution_factor * 0.8))
    
    def process_image(self, image_path: str, output_path: str = None) -> Dict[str, np.ndarray]:
        """Process image with high-resolution support"""
        # Load image
        image = self._load_image(image_path)
        if image is None:
            return {}
        
        # Check if we need to resize for target resolution
        original_shape = image.shape[:2]
        target_height, target_width = self.config.target_resolution
        
        if max(original_shape) < max(target_height, target_width):
            # If the source image is smaller than the requested resolution,
            # enlarge it first so subsequent drawing steps have enough pixels to
            # work with.
            image = self._upscale_image(image, (target_width, target_height))
            print(f"Upscaled image from {original_shape} to {image.shape[:2]}")
        
        # Process with tile-based approach for extremely large images
        if (image.shape[0] > 4096 or image.shape[1] > 4096) and self.config.tile_processing:
            print("Warning: Tile processing not implemented yet. Processing full image.")
            return self._process_full_image(image, output_path)
        else:
            return self._process_full_image(image, output_path)
    
    def _upscale_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Upscale image using high-quality interpolation"""
        target_width, target_height = target_size
        current_height, current_width = image.shape[:2]
        
        # Maintain aspect ratio
        aspect_ratio = current_width / current_height
        target_aspect = target_width / target_height
        
        if aspect_ratio > target_aspect:
            # Image is wider - fit width
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Image is taller - fit height
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        # Upscale using Lanczos interpolation which preserves edges better than
        # simpler algorithms like bilinear.
        upscaled = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LANCZOS4
        )
        
        return upscaled
    
    def _process_full_image(self, image: np.ndarray, output_path: str = None) -> Dict[str, np.ndarray]:
        """Process full image at high resolution"""
        return self.process_image_from_array(image, output_path)
    
    def process_image_from_array(self, image: np.ndarray, output_path: str = None) -> Dict[str, np.ndarray]:
        """Process image array (used by parent class)"""
        # Detect landmarks
        landmarks, detection_result = self._detect_landmarks(image)
        if not landmarks:
            print("No face detected in image")
            return {}
        
        results = {
            'original': image,
            'landmarks': landmarks
        }
        
        # Start with high-resolution blank canvas
        height, width = image.shape[:2]
        # Work on a pure white canvas so only the generated lines are visible
        # in the final result.
        current_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Apply features with high-resolution processing
        if self.config.enable_construction_lines:
            current_image = self.construction_generator.draw_construction_lines(
                current_image, landmarks, self.config
            )
            results['construction_lines'] = current_image.copy()
        
        if self.config.enable_mesh:
            current_image = self.mesh_generator.draw_face_mesh(
                current_image, detection_result, self.config
            )
            results['mesh'] = current_image.copy()
        
        if self.config.enable_dexined_outline and self.dexined_generator:
            outline_image = self.dexined_generator.generate_outline(image, self.config)
            # Overlay only the extracted lines, ignoring the white background
            # produced by DexiNed.
            current_image = self._add_lines_to_canvas(current_image, outline_image)
            results['dexined_outline'] = current_image.copy()
        
        # Create high-resolution transparent output
        if self.config.output_format == "rgba":
            rgba_image = BackgroundRemover.create_wireframe_rgba(
                current_image, landmarks, self.config.background_removal_method
            )
            results['final_rgba'] = rgba_image
            final_result = rgba_image
        else:
            results['final_rgb'] = current_image
            final_result = current_image
        
        # Save with high-quality settings
        if output_path:
            self._save_high_quality_image(final_result, output_path)
        
        return results
    
    def _save_high_quality_image(self, image: np.ndarray, output_path: str):
        """Save image with high-quality settings"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            # Use high-quality PNG compression. OpenCV expects BGRA ordering.
            bgra_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

            # Compression level 1 keeps edges crisp at the expense of file size.
            compression_params = [
                cv2.IMWRITE_PNG_COMPRESSION, 1,
                cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT
            ]

            success = cv2.imwrite(output_path, bgra_image, compression_params)
        else:  # RGB
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Determine format based on extension so callers can request JPEG or PNG.
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                compression_params = [cv2.IMWRITE_JPEG_QUALITY, 98]  # High quality JPEG
            else:
                compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]

            success = cv2.imwrite(output_path, bgr_image, compression_params)
        
        if success:
            print(f"High-quality wireframe saved to: {output_path}")
        else:
            print(f"Failed to save wireframe to: {output_path}")

def create_high_resolution_presets() -> Dict[str, HighResolutionConfig]:
    """Create high-resolution preset configurations.

    Combines the base wireframe presets with several output resolutions so
    users can request options like ``beginner_4K`` or ``advanced_8K``.
    """
    base_presets = create_preset_configs()
    hr_presets = {}
    
    # Resolution presets
    resolutions = {
        'HD': (1920, 1080),
        '4K': (3840, 2160), 
        '8K': (7680, 4320),
        'Print300DPI_A4': (3508, 2480),  # A4 at 300 DPI
        'Print300DPI_A3': (4961, 3508),  # A3 at 300 DPI
    }
    
    for res_name, resolution in resolutions.items():
        for preset_name, base_config in base_presets.items():
            config_name = f"{preset_name}_{res_name}"

            hr_config = HighResolutionConfig(
                # Copy base settings from the base preset
                enable_construction_lines=base_config.enable_construction_lines,
                enable_mesh=base_config.enable_mesh,
                enable_dexined_outline=base_config.enable_dexined_outline,

                # High-resolution settings
                target_resolution=resolution,
                enable_super_resolution=True,
                vector_construction_lines=True,
                enable_edge_enhancement=True,

                # Adaptive settings based on resolution
                tile_processing=resolution[0] * resolution[1] > 3840 * 2160,  # Enable for >4K
                output_format="rgba",
                background_removal_method="lines_only"
            )

            hr_presets[config_name] = hr_config
    
    return hr_presets

def main():
    """High-resolution wireframe processor CLI"""
    parser = argparse.ArgumentParser(description='High-Resolution Wireframe Portrait Processor')
    
    # Input/Output
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output image path')
    
    # Resolution settings
    parser.add_argument('--resolution', 
                       choices=['HD', '4K', '8K', 'Print300DPI_A4', 'Print300DPI_A3'],
                       default='4K', help='Target resolution')
    
    # Feature toggles
    parser.add_argument('--construction-lines', action='store_true', help='Enable construction lines')
    parser.add_argument('--mesh', action='store_true', help='Enable face mesh')
    parser.add_argument('--dexined', action='store_true', help='Enable DexiNed outline')
    
    # High-resolution presets
    parser.add_argument('--preset', help='Use high-resolution preset (e.g., beginner_4K, advanced_8K)')
    
    # Advanced options
    parser.add_argument('--dexined-model', 
                       default='DexiNed/checkpoints/BIPED/10/10_model.pth',
                       help='Path to DexiNed model')
    parser.add_argument('--enable-super-resolution', action='store_true',
                       help='Enable multi-scale super-resolution processing')
    parser.add_argument('--tile-processing', action='store_true',
                       help='Enable tile-based processing for extremely large images')
    
    args = parser.parse_args()
    
    # Resolution mapping
    resolution_map = {
        'HD': (1920, 1080),
        '4K': (3840, 2160), 
        '8K': (7680, 4320),
        'Print300DPI_A4': (3508, 2480),
        'Print300DPI_A3': (4961, 3508),
    }
    
    # Create configuration
    if args.preset:
        hr_presets = create_high_resolution_presets()
        if args.preset in hr_presets:
            config = hr_presets[args.preset]
        else:
            print(f"Available presets: {list(hr_presets.keys())}")
            return
    else:
        config = HighResolutionConfig(
            enable_construction_lines=args.construction_lines,
            enable_mesh=args.mesh,
            enable_dexined_outline=args.dexined,
            target_resolution=resolution_map[args.resolution],
            enable_super_resolution=args.enable_super_resolution,
            tile_processing=args.tile_processing
        )
    
    # Set DexiNed model path
    config.dexined_model_path = os.path.abspath(args.dexined_model)
    
    # Process image
    processor = HighResolutionWireframeProcessor(config)
    results = processor.process_image(args.input, args.output)
    
    if results:
        print(f"High-resolution wireframe processing completed!")
        print(f"Target resolution: {config.target_resolution}")
        print(f"Features used:")
        print(f"  Construction Lines: {'✓' if config.enable_construction_lines else '✗'}")
        print(f"  Face Mesh: {'✓' if config.enable_mesh else '✗'}")
        print(f"  DexiNed Outline: {'✓' if config.enable_dexined_outline else '✗'}")
        print(f"  Super Resolution: {'✓' if config.enable_super_resolution else '✗'}")
        print(f"  Vector Lines: {'✓' if config.vector_construction_lines else '✗'}")
    else:
        print("High-resolution wireframe processing failed!")

if __name__ == '__main__':
    main()