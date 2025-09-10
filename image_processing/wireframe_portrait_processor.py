#!/usr/bin/env python3
"""
Wireframe Portrait Processor
============================

A flexible system for generating wireframe portraits with toggleable features:
- Construction lines (facial guidelines)
- Face mesh (detailed contours)  
- DexiNed outline (edge detection)

Features can be enabled/disabled independently for different user skill levels.
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import argparse
import json
from datetime import datetime

# Import SVG generator
from svg_generator import SVGGenerator, SVGWireframeConfig

# Add DexiNed to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DexiNed'))

try:
    import torch
    from model import DexiNed
    DEXINED_AVAILABLE = True
except ImportError:
    print("Warning: DexiNed dependencies not available. DexiNed outline feature disabled.")
    DEXINED_AVAILABLE = False
    torch = None

class FeatureType(Enum):
    """Available wireframe features"""
    CONSTRUCTION_LINES = "construction_lines"
    MESH = "mesh"
    DEXINED_OUTLINE = "dexined_outline"

@dataclass
class WireframeConfig:
    """Configuration for wireframe generation"""
    # Feature toggles
    enable_construction_lines: bool = True
    enable_mesh: bool = False
    enable_dexined_outline: bool = False
    
    # Construction lines settings
    construction_line_thickness: int = 2
    construction_line_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'vertical_center': (255, 0, 0),    # Red
        'eyebrow_line': (0, 255, 0),       # Green
        'eye_lines': (0, 0, 255),          # Blue
        'nose_line': (255, 255, 0),        # Yellow
        'mouth_line': (255, 0, 255),       # Magenta
    })
    
    # Mesh settings
    mesh_thickness: int = 1
    mesh_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'tesselation': (128, 128, 128),    # Gray
        'contours': (64, 64, 64),          # Dark gray
        'irises': (0, 0, 0),               # Black
    })
    
    # DexiNed settings
    dexined_model_path: str = ""
    dexined_threshold: float = 0.5
    dexined_line_thickness: int = 1
    dexined_color: Tuple[int, int, int] = (0, 0, 0)  # Black
    
    # Output settings
    output_format: str = "rgba"  # "rgba", "rgb", "lines_only", "svg"
    background_removal_method: str = "lines_only"  # "lines_only", "face_mask", "color_diff", "color_filter"
    save_intermediate_steps: bool = False
    
    # SVG settings
    enable_svg_export: bool = False
    svg_output_path: str = ""

class ConstructionLinesGenerator:
    """Generates portrait construction lines based on MediaPipe landmarks"""
    
    @staticmethod
    def draw_construction_lines(image: np.ndarray, 
                              landmarks: List, 
                              config: WireframeConfig) -> np.ndarray:
        """
        Draw portrait construction lines following classical drawing guidelines
        
        Args:
            image: Input RGB image
            landmarks: MediaPipe face landmarks
            config: Wireframe configuration
            
        Returns:
            Image with construction lines drawn
        """
        if not landmarks:
            return image.copy()
        
        annotated = image.copy()
        height, width = image.shape[:2]
        thickness = config.construction_line_thickness
        colors = config.construction_line_colors
        
        def get_pixel_coords(landmark_idx):
            """Convert normalized landmark coordinates to pixel coordinates"""
            if landmark_idx >= len(landmarks):
                return None
            landmark = landmarks[landmark_idx]
            return (int(landmark.x * width), int(landmark.y * height))
        
        def draw_line_between_points(point_indices, line_color):
            """Draw line connecting multiple points"""
            points = []
            for idx in point_indices:
                point = get_pixel_coords(idx)
                if point:
                    points.append(point)
            
            # Draw line through all points
            for i in range(len(points) - 1):
                cv2.line(annotated, points[i], points[i + 1], line_color, thickness)
            
            return points
        
        # 1. Vertical Center Line [10→168→4→152]
        draw_line_between_points([10, 168, 4, 152], colors['vertical_center'])
        
        # 2. Eyebrow Line [63→293] 
        draw_line_between_points([63, 293], colors['eyebrow_line'])
        
        # 3. Eye Lines [33→263], [133→362], [145→159], [374→386]
        draw_line_between_points([33, 263], colors['eye_lines'])
        draw_line_between_points([133, 362], colors['eye_lines'])
        draw_line_between_points([145, 159], colors['eye_lines'])
        draw_line_between_points([374, 386], colors['eye_lines'])
        
        # 4. Nose Line [48→278]
        draw_line_between_points([48, 278], colors['nose_line'])
        
        # 5. Mouth Line [61→291]
        draw_line_between_points([61, 291], colors['mouth_line'])
        
        return annotated

class MeshGenerator:
    """Generates face mesh overlay using MediaPipe"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def draw_face_mesh(self, image: np.ndarray, 
                      detection_result, 
                      config: WireframeConfig) -> np.ndarray:
        """
        Draw face mesh on image
        
        Args:
            image: Input RGB image
            detection_result: MediaPipe detection result
            config: Wireframe configuration
            
        Returns:
            Image with face mesh drawn
        """
        if not detection_result.face_landmarks:
            return image.copy()
        
        annotated = image.copy()
        colors = config.mesh_colors
        
        for face_landmarks in detection_result.face_landmarks:
            # Convert landmarks for drawing
            from mediapipe.framework.formats import landmark_pb2
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in face_landmarks
            ])
            
            # Draw tesselation
            if colors['tesselation']:
                self.mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks_proto,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=colors['tesselation'], thickness=config.mesh_thickness
                    )
                )
            
            # Draw contours
            if colors['contours']:
                self.mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks_proto,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=colors['contours'], thickness=config.mesh_thickness + 1
                    )
                )
            
            # Draw irises
            if colors['irises']:
                self.mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks_proto,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=colors['irises'], thickness=config.mesh_thickness + 1
                    )
                )
        
        return annotated

class DexiNedGenerator:
    """Generates edge outlines using DexiNed model"""
    
    def __init__(self, model_path: str = ""):
        self.model = None
        self.device = None
        self.model_path = model_path
        
        if DEXINED_AVAILABLE and model_path and os.path.exists(model_path):
            self._load_model()
    
    def _load_model(self):
        """Load DexiNed model"""
        try:
            if torch is not None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model = DexiNed().to(self.device)
                
                if os.path.exists(self.model_path):
                    self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    self.model.eval()
                    print(f"DexiNed model loaded from {self.model_path}")
                else:
                    print(f"Warning: DexiNed model not found at {self.model_path}")
                    self.model = None
            else:
                self.model = None
        except Exception as e:
            print(f"Error loading DexiNed model: {e}")
            self.model = None
    
    def generate_outline(self, image: np.ndarray, config: WireframeConfig) -> np.ndarray:
        """
        Generate edge outline using DexiNed
        
        Args:
            image: Input RGB image
            config: Wireframe configuration
            
        Returns:
            Image with edge outline
        """
        if not DEXINED_AVAILABLE or self.model is None:
            # Fallback to simple edge detection
            return self._fallback_edge_detection(image, config)
        
        try:
            # Preprocess image for DexiNed
            img_tensor = self._preprocess_image(image)
            
            if torch is not None:
                with torch.no_grad():
                    predictions = self.model(img_tensor)
                    edge_map = predictions[-1].cpu().numpy()[0, 0]
            else:
                edge_map = np.zeros((352, 352))
            
            # Convert edge map to RGB image
            edge_image = self._postprocess_edges(edge_map, image.shape, config)
            
            return edge_image
            
        except Exception as e:
            print(f"Error in DexiNed processing: {e}")
            return self._fallback_edge_detection(image, config)
    
    def _preprocess_image(self, image: np.ndarray):
        """Preprocess image for DexiNed model"""
        # Resize to model input size (typically 352x352)
        img_resized = cv2.resize(image, (352, 352))
        
        # Convert to BGR and normalize
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        img_float = img_bgr.astype(np.float32)
        
        # Apply mean subtraction (DexiNed specific)
        mean_bgr = [103.939, 116.779, 123.68]
        for i in range(3):
            img_float[:, :, i] -= mean_bgr[i]
        
        # Convert to tensor and add batch dimension
        if torch is not None:
            img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0)
            
            if self.device:
                img_tensor = img_tensor.to(self.device)
            
            return img_tensor
        else:
            return img_float
    
    def _postprocess_edges(self, edge_map: np.ndarray, 
                          target_shape: Tuple[int, int, int],
                          config: WireframeConfig) -> np.ndarray:
        """Convert edge map to RGB image with white background"""
        # Resize edge map to target shape
        edge_resized = cv2.resize(edge_map, (target_shape[1], target_shape[0]))
        
        # Apply threshold
        edge_binary = (edge_resized > config.dexined_threshold).astype(np.uint8) * 255
        
        # Create RGB image with WHITE background (for wireframe mode)
        edge_rgb = np.ones(target_shape, dtype=np.uint8) * 255  # White background
        
        # Draw edges in specified color
        edge_rgb[edge_binary > 0] = config.dexined_color
        
        return edge_rgb
    
    def _fallback_edge_detection(self, image: np.ndarray, config: WireframeConfig) -> np.ndarray:
        """Fallback edge detection using Canny with white background"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Convert to RGB with WHITE background
        edge_rgb = np.ones_like(image) * 255  # White background
        edge_rgb[edges > 0] = config.dexined_color
        
        return edge_rgb

class BackgroundRemover:
    """Removes background to create transparent wireframe"""
    
    @staticmethod
    def create_wireframe_rgba(image: np.ndarray, 
                            landmarks: List,
                            method: str = "lines_only") -> np.ndarray:
        """
        Create RGBA wireframe by removing background
        
        Args:
            image: Input RGB image with lines drawn on white canvas
            landmarks: MediaPipe face landmarks (not used for lines_only method)
            method: Background removal method
            
        Returns:
            RGBA image with transparent background
        """
        if method == "lines_only":
            return BackgroundRemover._lines_only_method(image)
        elif method == "face_mask":
            return BackgroundRemover._face_mask_method(image, landmarks)
        elif method == "color_diff":
            return BackgroundRemover._color_diff_method(image)
        elif method == "color_filter":
            return BackgroundRemover._color_filter_method(image)
        else:
            # Default: use lines_only for wireframe
            return BackgroundRemover._lines_only_method(image)
    
    @staticmethod
    def _lines_only_method(image: np.ndarray) -> np.ndarray:
        """Convert white canvas to transparent, keeping only line elements"""
        # Convert RGB to RGBA
        rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        
        # Find white/near-white pixels (canvas background)
        white_threshold = 250  # Allow slight variations in white
        white_mask = np.all(image >= white_threshold, axis=2)
        
        # Make white areas transparent
        rgba_image[white_mask, 3] = 0
        
        return rgba_image
    
    @staticmethod
    def _face_mask_method(image: np.ndarray, landmarks: List) -> np.ndarray:
        """Remove background using face contour mask"""
        height, width = image.shape[:2]
        
        # Define face oval points
        face_oval_points = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        # Convert landmarks to pixel coordinates
        face_points = []
        for idx in face_oval_points:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                face_points.append((x, y))
        
        # Create face mask
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(face_points) > 0:
            face_contour = np.array(face_points, dtype=np.int32)
            cv2.fillPoly(mask, [face_contour], 255)
            
            # Erode mask slightly
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=2)
        
        # Create RGBA image
        result = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        result[mask > 0, 3] = 0  # Make face area transparent
        
        return result
    
    @staticmethod
    def _color_diff_method(image: np.ndarray, threshold: int = 30) -> np.ndarray:
        """Remove background using color difference detection"""
        # This would need an original image for comparison
        # For now, return basic RGBA conversion
        return cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    
    @staticmethod
    def _color_filter_method(image: np.ndarray) -> np.ndarray:
        """Remove background by filtering specific colors"""
        result = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        
        # Define colors to keep (line colors)
        colors_to_keep = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (128, 128, 128)
        ]
        
        # Create mask for colors to keep
        keep_mask = np.zeros(image.shape[:2], dtype=bool)
        tolerance = 20
        
        for color in colors_to_keep:
            lower = np.array([max(0, c - tolerance) for c in color])
            upper = np.array([min(255, c + tolerance) for c in color])
            color_mask = cv2.inRange(image, lower, upper)
            keep_mask = keep_mask | (color_mask > 0)
        
        # Make non-line areas transparent
        result[~keep_mask, 3] = 0
        
        return result

class WireframePortraitProcessor:
    """Main processor for wireframe portrait generation"""
    
    def __init__(self, config: WireframeConfig):
        self.config = config
        
        # Initialize generators
        self.construction_generator = ConstructionLinesGenerator()
        self.mesh_generator = MeshGenerator()
        self.dexined_generator = None
        
        if config.enable_dexined_outline and config.dexined_model_path:
            self.dexined_generator = DexiNedGenerator(config.dexined_model_path)
        
        # Initialize MediaPipe
        self.mp_face_landmarker = mp.solutions.face_mesh
        self._setup_face_detector()
    
    def _setup_face_detector(self):
        """Setup MediaPipe face detection"""
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        model_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'mediapipe_practice', 'face_landmarker.task'
        )
        
        if os.path.exists(model_path):
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        else:
            self.detector = None
            print(f"Warning: Face landmarker model not found at {model_path}")
    
    def process_image(self, image_path: str, output_path: str = None) -> Dict[str, np.ndarray]:
        """
        Process single image to generate wireframe portrait
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save result
            
        Returns:
            Dictionary containing generated images and intermediate steps
        """
        # Load and preprocess image
        image = self._load_image(image_path)
        if image is None:
            return {}
        
        # Detect face landmarks
        landmarks, detection_result = self._detect_landmarks(image)
        if not landmarks:
            print("No face detected in image")
            return {}
        
        results = {
            'original': image,
            'landmarks': landmarks
        }
        
        # Start with blank canvas (white background) - NO original image
        height, width = image.shape[:2]
        current_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White canvas
        
        # Apply features based on configuration
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
            # Add outline to canvas (only the lines)
            current_image = self._add_lines_to_canvas(current_image, outline_image)
            results['dexined_outline'] = current_image.copy()
        
        # Apply background removal if needed
        if self.config.output_format == "rgba":
            rgba_image = BackgroundRemover.create_wireframe_rgba(
                current_image, landmarks, self.config.background_removal_method
            )
            results['final_rgba'] = rgba_image
            final_result = rgba_image
        else:
            results['final_rgb'] = current_image
            final_result = current_image
        
        # Generate SVG if enabled
        svg_content = None
        if self.config.enable_svg_export or self.config.output_format == "svg":
            svg_content = self._generate_svg(image, landmarks, detection_result)
            results['svg_content'] = svg_content
            
            # Save SVG file
            if self.config.svg_output_path:
                svg_path = self.config.svg_output_path
            elif output_path:
                svg_path = os.path.splitext(output_path)[0] + '.svg'
            else:
                svg_path = None
                
            if svg_path and svg_content:
                with open(svg_path, 'w', encoding='utf-8') as f:
                    f.write(svg_content)
                print(f"Saved SVG wireframe to: {svg_path}")
        
        # Save raster result if output path specified and not SVG-only mode
        if output_path and self.config.output_format not in ["svg"]:
            self._save_image(final_result, output_path)
        elif output_path and self.config.output_format == "svg":
            if not svg_content:
                # If SVG format requested but no SVG generated, save as PNG instead
                png_path = os.path.splitext(output_path)[0] + '.png'
                self._save_image(final_result, png_path)
                print(f"Note: SVG generation failed, saved as PNG: {png_path}")
            else:
                print(f"SVG-only mode: raster output skipped, SVG saved to: {svg_path if 'svg_path' in locals() else 'specified path'}")
        
        return results
    
    def _load_image(self, image_path: str) -> Any:
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        # Read image with alpha channel support
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
        
        # Handle different channel counts
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # BGRA
                # Handle transparency by blending with white background
                image_rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                height, width = image_rgba.shape[:2]
                background = np.ones((height, width, 3), dtype=np.uint8) * 255
                
                alpha = image_rgba[:, :, 3] / 255.0
                for c in range(3):
                    background[:, :, c] = (alpha * image_rgba[:, :, c] + 
                                         (1 - alpha) * background[:, :, c])
                
                image_rgb = background.astype(np.uint8)
            elif image.shape[2] == 3:  # BGR
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            # Grayscale
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return image_rgb
    
    def _detect_landmarks(self, image: np.ndarray) -> Tuple[Any, Any]:
        """Detect face landmarks using MediaPipe"""
        if self.detector is None:
            return None, None
        
        try:
            # Convert to MediaPipe format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            
            # Detect landmarks
            detection_result = self.detector.detect(mp_image)
            
            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                return landmarks, detection_result
            else:
                return None, None
                
        except Exception as e:
            print(f"Error in landmark detection: {e}")
            return None, None
    
    def _add_lines_to_canvas(self, canvas: np.ndarray, line_image: np.ndarray) -> np.ndarray:
        """Add line elements to canvas - only the lines, not background"""
        result = canvas.copy()
        
        # Find line pixels (non-white/non-background pixels)
        if len(line_image.shape) == 3:
            # Color image - find any non-white pixels
            line_mask = ~np.all(line_image == [255, 255, 255], axis=2)
        else:
            # Grayscale - find any non-white pixels  
            line_mask = line_image < 255
        
        # Add line pixels to canvas
        if len(line_image.shape) == 3:
            result[line_mask] = line_image[line_mask]
        else:
            # Convert grayscale lines to color
            result[line_mask] = self.config.dexined_color
        
        return result
    
    def _generate_svg(self, image: np.ndarray, landmarks: List, detection_result) -> str:
        """
        Generate SVG representation of wireframe elements
        
        Args:
            image: Original input image
            landmarks: Face landmarks
            detection_result: MediaPipe detection result
            
        Returns:
            SVG content as string
        """
        height, width = image.shape[:2]
        
        # Create SVG generator
        svg_generator = SVGGenerator(width, height, "white")
        
        # Add metadata
        metadata = {
            'features': [],
            'timestamp': datetime.now().isoformat(),
            'resolution': f'{width}x{height}'
        }
        
        # Convert landmarks to numpy array for SVG processing
        landmark_coords = []
        if landmarks:
            for landmark in landmarks:
                landmark_coords.append([landmark.x, landmark.y, landmark.z])
        landmark_array = np.array(landmark_coords)
        
        # Add construction lines if enabled
        if self.config.enable_construction_lines:
            construction_config = {
                'color': f'rgb{self.config.construction_line_colors["vertical_center"]}',
                'thickness': self.config.construction_line_thickness
            }
            svg_generator.add_construction_lines(landmark_array, construction_config)
            metadata['features'].append('construction_lines')
        
        # Add face mesh if enabled
        if self.config.enable_mesh and detection_result.face_landmarks:
            # Get MediaPipe face mesh connections
            mp_face_mesh = mp.solutions.face_mesh
            connections = list(mp_face_mesh.FACEMESH_TESSELATION)
            
            mesh_config = {
                'color': f'rgb{self.config.mesh_colors["tesselation"]}',
                'thickness': self.config.mesh_thickness
            }
            svg_generator.add_face_mesh(landmark_array, connections, mesh_config)
            metadata['features'].append('face_mesh')
        
        # Add DexiNed outline if enabled
        if self.config.enable_dexined_outline and self.dexined_generator:
            # Generate edge outline and convert to contours
            outline_image = self.dexined_generator.generate_outline(image, self.config)
            contours = self._extract_contours_from_outline(outline_image)
            
            dexined_config = {
                'color': f'rgb{self.config.dexined_color}',
                'thickness': self.config.dexined_line_thickness
            }
            svg_generator.add_dexined_outline(contours, dexined_config)
            metadata['features'].append('dexined_outline')
        
        # Add metadata
        svg_generator.add_metadata(metadata)
        
        return svg_generator.to_string(pretty=True)
    
    def _extract_contours_from_outline(self, outline_image: np.ndarray) -> List[np.ndarray]:
        """Extract contours from DexiNed outline image"""
        # Convert to grayscale if needed
        if len(outline_image.shape) == 3:
            gray = cv2.cvtColor(outline_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = outline_image
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        min_contour_area = 50
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        
        return filtered_contours
    
    def _blend_images(self, base_image: np.ndarray, overlay_image: np.ndarray) -> np.ndarray:
        """Blend two images together (legacy method, use _add_lines_to_canvas instead)"""
        return self._add_lines_to_canvas(base_image, overlay_image)
    
    def _save_image(self, image: np.ndarray, output_path: str):
        """Save image to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            # Convert RGBA to BGRA for OpenCV
            bgra_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
            success = cv2.imwrite(output_path, bgra_image)
        else:  # RGB
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(output_path, bgr_image)
        
        if success:
            print(f"Saved wireframe to: {output_path}")
        else:
            print(f"Failed to save wireframe to: {output_path}")

def create_preset_configs() -> Dict[str, WireframeConfig]:
    """Create preset configurations for different user types"""
    presets = {}
    
    # Beginner: All features enabled
    presets['beginner'] = WireframeConfig(
        enable_construction_lines=True,
        enable_mesh=True,
        enable_dexined_outline=True,
        output_format="rgba"
    )
    
    # Intermediate: Construction + Mesh
    presets['intermediate'] = WireframeConfig(
        enable_construction_lines=True,
        enable_mesh=True,
        enable_dexined_outline=False,
        output_format="rgba"
    )
    
    # Advanced: Construction only
    presets['advanced'] = WireframeConfig(
        enable_construction_lines=True,
        enable_mesh=False,
        enable_dexined_outline=False,
        output_format="rgba"
    )
    
    # Outline only: DexiNed only
    presets['outline_only'] = WireframeConfig(
        enable_construction_lines=False,
        enable_mesh=False,
        enable_dexined_outline=True,
        output_format="rgba"
    )
    
    # Mesh only: Face mesh only
    presets['mesh_only'] = WireframeConfig(
        enable_construction_lines=False,
        enable_mesh=True,
        enable_dexined_outline=False,
        output_format="rgba"
    )
    
    return presets

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Wireframe Portrait Processor')
    
    # Input/Output
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output image path')
    
    # Feature toggles
    parser.add_argument('--construction-lines', action='store_true', 
                       help='Enable construction lines')
    parser.add_argument('--mesh', action='store_true',
                       help='Enable face mesh')
    parser.add_argument('--dexined', action='store_true',
                       help='Enable DexiNed outline')
    
    # Presets
    parser.add_argument('--preset', choices=['beginner', 'intermediate', 'advanced', 
                                           'outline_only', 'mesh_only'],
                       help='Use preset configuration')
    
    # Advanced options
    parser.add_argument('--dexined-model', 
                       default='../DexiNed/checkpoints/BIPED2CLASSIC/10_model.pth',
                       help='Path to DexiNed model')
    parser.add_argument('--output-format', choices=['rgb', 'rgba', 'svg'],
                       default='rgba', help='Output format')
    parser.add_argument('--background-removal', 
                       choices=['lines_only', 'face_mask', 'color_diff', 'color_filter'],
                       default='lines_only', help='Background removal method')
    
    # SVG options
    parser.add_argument('--svg', action='store_true',
                       help='Enable SVG export (in addition to raster output)')
    parser.add_argument('--svg-output', help='SVG output file path')
    
    args = parser.parse_args()
    
    # Create configuration
    if args.preset:
        presets = create_preset_configs()
        config = presets[args.preset]
        # Override with command line options
        config.output_format = args.output_format  # Override preset output format
        config.enable_svg_export = args.svg or args.output_format == 'svg'
        config.svg_output_path = args.svg_output or ""
    else:
        config = WireframeConfig(
            enable_construction_lines=args.construction_lines,
            enable_mesh=args.mesh,
            enable_dexined_outline=args.dexined,
            output_format=args.output_format,
            background_removal_method=args.background_removal,
            enable_svg_export=args.svg or args.output_format == 'svg',
            svg_output_path=args.svg_output or ""
        )
    
    # Set DexiNed model path - use absolute path
    if args.dexined_model.startswith('../'):
        # Convert relative path to absolute
        config.dexined_model_path = os.path.abspath(args.dexined_model)
    else:
        config.dexined_model_path = args.dexined_model
    
    # Process image
    processor = WireframePortraitProcessor(config)
    results = processor.process_image(args.input, args.output)
    
    if results:
        print("Wireframe processing completed successfully!")
        print(f"Features used:")
        print(f"  Construction Lines: {'✓' if config.enable_construction_lines else '✗'}")
        print(f"  Face Mesh: {'✓' if config.enable_mesh else '✗'}")
        print(f"  DexiNed Outline: {'✓' if config.enable_dexined_outline else '✗'}")
    else:
        print("Wireframe processing failed!")

if __name__ == '__main__':
    main()