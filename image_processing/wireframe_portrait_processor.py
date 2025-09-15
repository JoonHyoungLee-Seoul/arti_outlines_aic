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
import mediapipe.tasks as mp_tasks
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
    POSE_LANDMARKS = "pose_landmarks"

@dataclass
class WireframeConfig:
    """Configuration for wireframe generation"""
    # Feature toggles
    enable_construction_lines: bool = True
    enable_mesh: bool = False
    enable_dexined_outline: bool = False
    enable_pose_landmarks: bool = False
    
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
    
    # Pose landmarks settings
    pose_model_path: str = ""  # Path to pose_landmarker.task file
    pose_detection_confidence: float = 0.5
    pose_line_thickness: int = 2
    pose_point_radius: int = 4
    pose_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'body_connections': (255, 165, 0),   # Orange - body skeleton
        'landmark_points': (255, 0, 0),      # Red - landmark points
    })
    
    # Output settings
    output_format: str = "rgba"  # "rgba", "rgb", "lines_only", "svg"
    background_removal_method: str = "lines_only"  # "lines_only", "face_mask", "color_diff", "color_filter"
    save_intermediate_steps: bool = False
    
    # SVG settings
    enable_svg_export: bool = False
    svg_output_path: str = ""

    # Background merge settings
    enable_background_merge: bool = False
    background_directory: str = ""
    foreground_directory: str = ""  # Optional foreground directory for creative control
    foreground_transparency: int = 100  # 0-100 scale (0=transparent, 100=opaque)
    background_transparency: int = 50  # 0-100 scale (0=transparent, 100=opaque)

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
            # Landmarks are in [0,1] range relative to image size
            return (int(landmark.x * width), int(landmark.y * height))
        
        def draw_line_between_points(point_indices, line_color):
            """Draw line connecting multiple points"""
            points = []
            for idx in point_indices:
                point = get_pixel_coords(idx)
                if point:
                    points.append(point)

            # Connect the landmark points sequentially.
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

            # Draw tesselation: full triangular mesh across the face
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

            # Draw contours: emphasis around outer facial features
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

            # Draw irises to show eye direction
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
            # If the neural model isn't available fall back to a basic Canny
            # edge detector so the pipeline still produces an outline.
            return self._fallback_edge_detection(image, config)
        
        try:
            # Preprocess image for DexiNed
            img_tensor = self._preprocess_image(image)
            
            if torch is not None:
                with torch.no_grad():
                    predictions = self.model(img_tensor)
                    edge_map = predictions[-1].cpu().numpy()[0, 0]  # Use final prediction map
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
        # Resize to model input size (352x352 for DexiNed)
        img_resized = cv2.resize(image, (352, 352))
        
        # Convert to BGR and normalize
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        img_float = img_bgr.astype(np.float32)
        
        # Apply mean subtraction (DexiNed uses ImageNet means)
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
        # Resize edge map back to the original image resolution
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
        edges = cv2.Canny(gray, 50, 150)  # Simple Canny edge detector
        
        # Convert to RGB with WHITE background
        edge_rgb = np.ones_like(image) * 255  # White background
        edge_rgb[edges > 0] = config.dexined_color
        
        return edge_rgb
    

class PoseLandmarkerGenerator:
    """Generates pose landmarks using MediaPipe Pose Landmarker"""
    
    def __init__(self, model_path: str = ""):
        self.detector = None
        self.model_path = model_path
        
        # Excluded landmarks (face/hands details)
        self.excluded_landmarks = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22}
        
        # Pose connections for drawing skeleton (filtered to exclude face/hands)
        self.pose_connections = [
            # Torso connections
            (11, 12),  # left shoulder to right shoulder
            (11, 23),  # left shoulder to left hip
            (12, 24),  # right shoulder to right hip
            (23, 24),  # left hip to right hip
            
            # Left arm connections
            (11, 13),  # left shoulder to left elbow
            (13, 15),  # left elbow to left wrist
            
            # Right arm connections  
            (12, 14),  # right shoulder to right elbow
            (14, 16),  # right elbow to right wrist
            
            # Left leg connections
            (23, 25),  # left hip to left knee
            (25, 27),  # left knee to left ankle
            (27, 29),  # left ankle to left heel
            (27, 31),  # left ankle to left foot index
            
            # Right leg connections
            (24, 26),  # right hip to right knee
            (26, 28),  # right knee to right ankle
            (28, 30),  # right ankle to right heel
            (28, 32),  # right ankle to right foot index
        ]
        
        if model_path and os.path.exists(model_path):
            self._load_model()
    
    def _load_model(self):
        """Load MediaPipe Pose Landmarker model"""
        try:
            base_options = mp_tasks.BaseOptions(model_asset_path=self.model_path)
            options = mp_tasks.vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                running_mode=mp_tasks.vision.RunningMode.IMAGE
            )
            self.detector = mp_tasks.vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Error loading Pose Landmarker model: {e}")
            self.detector = None
    
    def detect_pose_landmarks(self, image: np.ndarray, config: WireframeConfig) -> Optional[List]:
        """
        Detect pose landmarks from image
        
        Args:
            image: Input RGB image
            config: Wireframe configuration
            
        Returns:
            List of pose landmarks or None if detection fails
        """
        if self.detector is None:
            return None
            
        try:
            # Convert numpy array to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            
            # Detect pose landmarks
            detection_result = self.detector.detect(mp_image)
            
            if detection_result.pose_landmarks:
                return detection_result.pose_landmarks[0]  # Return first detected pose
            else:
                return None
                
        except Exception as e:
            print(f"Error in pose landmark detection: {e}")
            return None
    
    def draw_pose_landmarks(self, image: np.ndarray, landmarks: List, config: WireframeConfig) -> np.ndarray:
        """
        Draw pose landmarks and connections on image
        
        Args:
            image: Input image
            landmarks: Pose landmarks
            config: Wireframe configuration
            
        Returns:
            Annotated image with pose landmarks
        """
        annotated = image.copy()
        height, width = image.shape[:2]
        
        # Draw connections (skeleton)
        for connection in self.pose_connections:
            start_idx, end_idx = connection
            
            # Skip if landmarks are excluded
            if start_idx in self.excluded_landmarks or end_idx in self.excluded_landmarks:
                continue
                
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_landmark = landmarks[start_idx]
                end_landmark = landmarks[end_idx]
                
                # Convert normalized coordinates to pixel coordinates
                start_point = (
                    int(start_landmark.x * width),
                    int(start_landmark.y * height)
                )
                end_point = (
                    int(end_landmark.x * width),
                    int(end_landmark.y * height)
                )
                
                # Draw connection line
                cv2.line(annotated, start_point, end_point, 
                        config.pose_colors['body_connections'], 
                        config.pose_line_thickness)
        
        # Draw landmark points
        for idx, landmark in enumerate(landmarks):
            # Skip excluded landmarks
            if idx in self.excluded_landmarks:
                continue
                
            # Convert normalized coordinates to pixel coordinates
            point = (
                int(landmark.x * width),
                int(landmark.y * height)
            )
            
            # Draw landmark point
            cv2.circle(annotated, point, config.pose_point_radius, 
                      config.pose_colors['landmark_points'], -1)
        
        return annotated


class BackgroundMerger:
    """Merges foreground wireframe with background images at adjustable transparency"""

    def __init__(self, config: WireframeConfig):
        self.config = config

    def find_matching_background(self, input_image_path: str) -> Optional[str]:
        """
        Find matching background image for the given input image

        Args:
            input_image_path: Path to the input foreground image

        Returns:
            Path to matching background image or None if not found
        """
        if not self.config.background_directory or not os.path.exists(self.config.background_directory):
            return None

        # Extract base filename without extension
        input_filename = os.path.splitext(os.path.basename(input_image_path))[0]

        # Remove common suffixes like '_fg' to find base name
        base_name = input_filename.replace('_fg', '')

        # Look for matching background file
        for bg_file in os.listdir(self.config.background_directory):
            if bg_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                bg_basename = os.path.splitext(bg_file)[0]

                # Check for exact match or background pattern
                if (base_name in bg_basename or
                    bg_basename.replace('_bg', '') == base_name or
                    bg_basename == base_name):
                    return os.path.join(self.config.background_directory, bg_file)

        # If no specific match, return first available background
        bg_files = [f for f in os.listdir(self.config.background_directory)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if bg_files:
            print(f"Warning: No matching background found for {input_filename}, using {bg_files[0]}")
            return os.path.join(self.config.background_directory, bg_files[0])

        return None

    def find_matching_foreground(self, input_image_path: str) -> Optional[str]:
        """
        Find matching foreground image when using separate foreground directory

        Args:
            input_image_path: Path to the input image

        Returns:
            Path to matching foreground image or None if not found
        """
        if not self.config.foreground_directory or not os.path.exists(self.config.foreground_directory):
            return None

        # Extract base filename without extension
        input_filename = os.path.splitext(os.path.basename(input_image_path))[0]

        # Remove common suffixes to find base name
        base_name = input_filename.replace('_fg', '').replace('_bg', '')

        # Look for matching foreground file
        for fg_file in os.listdir(self.config.foreground_directory):
            if fg_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                fg_basename = os.path.splitext(fg_file)[0]

                # Check for exact match or foreground pattern
                if (base_name in fg_basename or
                    fg_basename.replace('_fg', '') == base_name or
                    fg_basename == base_name):
                    return os.path.join(self.config.foreground_directory, fg_file)

        # If no specific match, return first available foreground
        fg_files = [f for f in os.listdir(self.config.foreground_directory)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if fg_files:
            print(f"Warning: No matching foreground found for {input_filename}, using {fg_files[0]}")
            return os.path.join(self.config.foreground_directory, fg_files[0])

        return None

    def merge_with_background(self, foreground: np.ndarray, background_path: str, foreground_path: Optional[str] = None) -> np.ndarray:
        """
        Merge foreground wireframe with background image using independent transparency controls

        Args:
            foreground: Generated wireframe image (RGB or RGBA)
            background_path: Path to background image
            foreground_path: Optional path to separate foreground image for creative control

        Returns:
            Merged image with independent transparency controls
        """
        if not background_path or not os.path.exists(background_path):
            print(f"Background image not found: {background_path}")
            return foreground

        try:
            # Load background image
            background = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)
            if background is None:
                print(f"Could not load background image: {background_path}")
                return foreground

            # Convert background to RGB
            if len(background.shape) == 3:
                if background.shape[2] == 4:  # BGRA
                    background_rgb = cv2.cvtColor(background, cv2.COLOR_BGRA2RGB)
                elif background.shape[2] == 3:  # BGR
                    background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
                else:
                    background_rgb = background
            else:
                # Grayscale
                background_rgb = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)

            # Use foreground alpha mask to remove person from background
            # Since bg.png still contains the person, we need to remove it using fg alpha mask
            bg_alpha_mask = None  # Will be set after foreground is loaded

            # Load separate foreground image for background blending
            # The wireframe will be overlaid on top of the merged result
            original_foreground = None
            if foreground_path and os.path.exists(foreground_path):
                fg_loaded = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
                if fg_loaded is not None:
                    # Convert to RGB format
                    if len(fg_loaded.shape) == 3:
                        if fg_loaded.shape[2] == 4:  # BGRA
                            original_foreground = cv2.cvtColor(fg_loaded, cv2.COLOR_BGRA2RGB)
                        elif fg_loaded.shape[2] == 3:  # BGR
                            original_foreground = cv2.cvtColor(fg_loaded, cv2.COLOR_BGR2RGB)
                        else:
                            original_foreground = fg_loaded
                    else:
                        # Grayscale
                        original_foreground = cv2.cvtColor(fg_loaded, cv2.COLOR_GRAY2RGB)
                else:
                    print(f"Could not load foreground file: {foreground_path}")
            
            # Note: Top layer wireframes will be applied after background merge in main process

            # Get dimensions and resize both images to match
            fg_height, fg_width = foreground.shape[:2]
            background_resized = cv2.resize(background_rgb, (fg_width, fg_height))

            # Use original foreground image for blending if available, otherwise use wireframe
            blend_foreground = original_foreground if original_foreground is not None else foreground
            
            # Handle foreground format and extract proper alpha mask
            if len(blend_foreground.shape) == 3 and blend_foreground.shape[2] == 4:
                # RGBA foreground - use existing alpha channel
                foreground_rgb = blend_foreground[:, :, :3]
                fg_alpha_channel = blend_foreground[:, :, 3] / 255.0
            else:
                # RGB foreground - create alpha mask for non-white areas
                foreground_rgb = blend_foreground
                # Create alpha mask: transparent for white areas, opaque for colored areas
                white_threshold = 250
                white_mask = np.all(foreground_rgb >= white_threshold, axis=2)
                fg_alpha_channel = np.ones((fg_height, fg_width), dtype=np.float32)
                fg_alpha_channel[white_mask] = 0.0

            # Create background alpha mask using inverted foreground alpha
            # Where foreground has person (alpha=1) → background should be transparent (alpha=0)
            # Where foreground is transparent (alpha=0) → background should be visible (alpha=1)
            bg_alpha_mask = 1.0 - fg_alpha_channel  # Invert the foreground mask

            # Apply transparency settings (0-100 scale)
            fg_alpha = self.config.foreground_transparency / 100.0  # 0=transparent, 1=opaque
            bg_alpha = self.config.background_transparency / 100.0   # 0=transparent, 1=opaque

            # Create result image
            result = np.zeros_like(foreground_rgb, dtype=np.float32)

            for c in range(3):  # RGB channels
                # Calculate contributions with proper transparency handling
                fg_contribution = foreground_rgb[:, :, c].astype(np.float32) * fg_alpha_channel * fg_alpha
                # Apply background alpha mask to prevent person areas from showing in background
                bg_contribution = background_resized[:, :, c].astype(np.float32) * bg_alpha * bg_alpha_mask

                # Alpha blending for independent transparency control
                effective_fg_alpha = fg_alpha_channel * fg_alpha

                # Start with background contribution
                result[:, :, c] = bg_contribution
                
                # Add foreground contribution where person exists
                person_areas = fg_alpha_channel > 0
                result[:, :, c][person_areas] = (
                    fg_contribution[person_areas] + 
                    bg_contribution[person_areas] * (1 - effective_fg_alpha[person_areas])
                )
                
                # For areas where foreground transparency is 0%, fill with background color
                # For creative control: fg_alpha=0 means transparent person area
                if fg_alpha == 0:
                    # When foreground is completely transparent, show background in person areas
                    result[:, :, c][person_areas] = bg_contribution[person_areas]

            # Clip to valid range and convert back to uint8
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            # Top layer wireframes will be applied in main process after this function returns

            print(f"Images merged - Foreground: {self.config.foreground_transparency}%, Background: {self.config.background_transparency}%")
            return result

        except Exception as e:
            print(f"Error merging images: {e}")
            return foreground


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
        self.pose_landmarker_generator = None
        self.background_merger = None

        if config.enable_dexined_outline and config.dexined_model_path:
            self.dexined_generator = DexiNedGenerator(config.dexined_model_path)

        if config.enable_pose_landmarks and config.pose_model_path:
            self.pose_landmarker_generator = PoseLandmarkerGenerator(config.pose_model_path)

        if config.enable_background_merge:
            self.background_merger = BackgroundMerger(config)
        
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
        
        # Start with blank canvas (white background) - the original photo is not
        # part of the final wireframe output.
        height, width = image.shape[:2]
        current_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Generate each wireframe layer separately to ensure proper stacking
        # Layer 1: Face Mesh (drawn directly on white canvas)
        face_mesh_layer = None
        if self.config.enable_mesh:
            print("Generating face mesh layer...")
            face_mesh_layer = self.mesh_generator.draw_face_mesh(
                current_image.copy(), detection_result, self.config
            )
            results['mesh'] = face_mesh_layer.copy()
            print("Face mesh layer generated")
        
        # Layer 2: Construction Lines (separate layer)
        construction_lines_layer = None
        if self.config.enable_construction_lines:
            print("Generating construction lines layer...")
            construction_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
            construction_lines_layer = self.construction_generator.draw_construction_lines(
                construction_canvas, landmarks, self.config
            )
            results['construction_lines'] = construction_lines_layer.copy()
            print("Construction lines layer generated")
        
        # Layer 3: Pose Landmarks (separate layer)
        pose_landmarks_layer = None
        if self.config.enable_pose_landmarks and self.pose_landmarker_generator:
            print("Generating pose landmarks layer...")
            pose_landmarks = self.pose_landmarker_generator.detect_pose_landmarks(image, self.config)
            if pose_landmarks:
                pose_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
                pose_landmarks_layer = self.pose_landmarker_generator.draw_pose_landmarks(
                    pose_canvas, pose_landmarks, self.config
                )
                results['pose_landmarks'] = pose_landmarks_layer.copy()
                print("Pose landmarks layer generated")
            else:
                print("No pose landmarks detected")
        
        # Layer 4: DexiNed Outline (if enabled)
        dexined_layer = None
        if self.config.enable_dexined_outline and self.dexined_generator:
            print("Generating DexiNed outline layer...")
            dexined_layer = self.dexined_generator.generate_outline(image, self.config)
            results['dexined_outline'] = dexined_layer.copy()
            print("DexiNed outline layer generated")

        # FINAL LAYER COMPOSITION (Bottom → Top)
        print("Starting final layer composition...")
        
        if self.config.enable_background_merge and self.background_merger:
            # Step 1 & 2: Background + Foreground merge
            background_path = self.background_merger.find_matching_background(image_path)
            foreground_path = self.background_merger.find_matching_foreground(image_path)

            if background_path:
                print("Compositing background and foreground layers...")
                # Use white canvas as base for background merge
                base_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
                current_image = self.background_merger.merge_with_background(
                    base_canvas, background_path, foreground_path
                )
                results['background_merged'] = current_image.copy()
                print("Background/foreground layers composited")
            else:
                print("Warning: Background merge enabled but no matching background found")
        else:
            # Start with white canvas if no background merge
            current_image = np.ones((height, width, 3), dtype=np.uint8) * 255
            
        # Step 3: Overlay Face Mesh
        if face_mesh_layer is not None:
            print("Overlaying face mesh layer...")
            mesh_mask = np.all(face_mesh_layer < 250, axis=2)
            mesh_pixel_count = np.sum(mesh_mask)
            if mesh_pixel_count > 0:
                current_image[mesh_mask] = face_mesh_layer[mesh_mask]
                print(f"Face mesh overlaid: {mesh_pixel_count} pixels")
            
        # Step 4: Overlay Construction Lines (Top layer)
        if construction_lines_layer is not None:
            print("Overlaying construction lines layer...")
            # More lenient mask detection - not pure white (255,255,255)
            construction_mask = ~np.all(construction_lines_layer == 255, axis=2)
            construction_pixel_count = np.sum(construction_mask)
            if construction_pixel_count > 0:
                # Make construction lines darker for better visibility
                darkened_construction = np.clip(construction_lines_layer[construction_mask] * 0.8, 0, 180)
                current_image[construction_mask] = darkened_construction
                print(f"Construction lines overlaid: {construction_pixel_count} pixels")
            else:
                print("WARNING: No construction line pixels found")
                
        # Step 5: Overlay Pose Landmarks (Top layer)  
        if pose_landmarks_layer is not None:
            print("Overlaying pose landmarks layer...")
            # More lenient mask detection - not pure white (255,255,255)
            pose_mask = ~np.all(pose_landmarks_layer == 255, axis=2)
            pose_pixel_count = np.sum(pose_mask)
            if pose_pixel_count > 0:
                # Make pose landmarks darker for better visibility
                darkened_pose = np.clip(pose_landmarks_layer[pose_mask] * 0.8, 0, 180)
                current_image[pose_mask] = darkened_pose
                print(f"Pose landmarks overlaid: {pose_pixel_count} pixels")
            else:
                print("WARNING: No pose landmark pixels found")
                
        # Step 6: Overlay DexiNed Outline (if enabled)
        if dexined_layer is not None:
            print("Overlaying DexiNed outline layer...")
            outline_mask = np.all(dexined_layer < 250, axis=2)
            outline_pixel_count = np.sum(outline_mask)
            if outline_pixel_count > 0:
                current_image[outline_mask] = dexined_layer[outline_mask]
                print(f"DexiNed outline overlaid: {outline_pixel_count} pixels")
        
        print("Final layer composition completed")

        # Apply background removal if needed to produce RGBA output
        if self.config.output_format == "rgba":
            rgba_image = BackgroundRemover.create_wireframe_rgba(
                current_image, landmarks, self.config.background_removal_method
            )
            results['final_rgba'] = rgba_image
            final_result = rgba_image
        else:
            results['final_rgb'] = current_image
            final_result = current_image
        
        # Generate SVG if requested
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
        
        # Read image with alpha channel support so transparent PNGs are handled
        # correctly.
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
        
        # Handle different channel counts
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # BGRA
                # Convert transparent images to RGB by compositing over white
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
            # Convert to MediaPipe format and run the face landmarker
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
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
            # Color image - treat any non-white pixel as a line
            line_mask = ~np.all(line_image == [255, 255, 255], axis=2)
        else:
            # Grayscale - find any non-white pixels
            line_mask = line_image < 255

        # Copy the detected line pixels onto the canvas
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
        
        # Add pose landmarks if enabled
        if self.config.enable_pose_landmarks and self.pose_landmarker_generator:
            pose_landmarks = self.pose_landmarker_generator.detect_pose_landmarks(image, self.config)
            if pose_landmarks:
                # Convert pose landmarks to format compatible with SVG generator
                pose_landmark_coords = []
                for landmark in pose_landmarks:
                    pose_landmark_coords.append([landmark.x, landmark.y, landmark.z])
                pose_landmark_array = np.array(pose_landmark_coords)
                
                pose_config = {
                    'line_color': f'rgb{self.config.pose_colors["body_connections"]}',
                    'point_color': f'rgb{self.config.pose_colors["landmark_points"]}',
                    'line_thickness': self.config.pose_line_thickness,
                    'point_radius': self.config.pose_point_radius,
                    'connections': self.pose_landmarker_generator.pose_connections,
                    'excluded_landmarks': self.pose_landmarker_generator.excluded_landmarks
                }
                svg_generator.add_pose_landmarks(pose_landmark_array, pose_config)
                metadata['features'].append('pose_landmarks')
        
        # Add metadata
        svg_generator.add_metadata(metadata)
        
        return svg_generator.to_string(pretty=True)
    
    def _extract_contours_from_outline(self, outline_image: np.ndarray) -> List[np.ndarray]:
        """Extract contours from DexiNed outline image with quality optimization"""
        # Convert to grayscale if needed
        if len(outline_image.shape) == 3:
            gray = cv2.cvtColor(outline_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = outline_image
        
        # Ensure proper binary image for contour detection
        # DexiNed outputs float values, convert to proper binary
        if gray.dtype == np.float32 or gray.dtype == np.float64:
            gray = (gray * 255).astype(np.uint8)
        
        # Apply light morphological operations to clean up the image
        kernel = np.ones((1,1), np.uint8)  # Smaller kernel to preserve details
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Apply threshold to ensure binary image
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)  # Even lower threshold for more edge details
        
        # Find contours - use RETR_LIST to get all contours, not just external
        # Use CHAIN_APPROX_NONE for more precise contours initially
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # Filter and process contours
        processed_contours = []
        min_contour_length = 15  # Reduced minimum perimeter to capture more details
        epsilon_factor = 0.002  # Reduced approximation for better detail preservation
        
        for contour in contours:
            # Filter by perimeter length for better edge quality
            if cv2.arcLength(contour, True) > min_contour_length:
                # Approximate contour to reduce noise while preserving important features
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                processed_contours.append(approx_contour)
        
        # Optional debug info
        # print(f"Debug: Found {len(contours)} total contours, {len(processed_contours)} after filtering and processing")
        
        return processed_contours
    
    def _blend_images(self, base_image: np.ndarray, overlay_image: np.ndarray) -> np.ndarray:
        """Blend two images together (legacy method, use _add_lines_to_canvas instead)"""
        return self._add_lines_to_canvas(base_image, overlay_image)
    
    def _save_image(self, image: np.ndarray, output_path: str):
        """Save image to file"""
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if dirname is not empty
            os.makedirs(output_dir, exist_ok=True)
        
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
    
    # Default DexiNed model path
    default_dexined_path = os.path.join(
        os.path.dirname(__file__), "..", "DexiNed", "checkpoints", "BIPED", "10", "10_model.pth"
    )
    
    # Default pose landmarker model path  
    default_pose_path = os.path.join(
        os.path.dirname(__file__), "..", "mediapipe_practice", "pose_landmarker.task"
    )
    default_pose_path = os.path.abspath(default_pose_path)
    
    # Beginner: All features enabled with detailed guidelines
    presets['beginner'] = WireframeConfig(
        enable_construction_lines=True,
        enable_mesh=True,
        enable_dexined_outline=True,
        enable_pose_landmarks=True,
        dexined_model_path=default_dexined_path,
        pose_model_path=default_pose_path,
        output_format="rgba",
    )
    
    # Intermediate: Construction + Mesh
    presets['intermediate'] = WireframeConfig(
        enable_construction_lines=True,
        enable_mesh=True,
        enable_dexined_outline=False,
        enable_pose_landmarks=True,
        pose_model_path=default_pose_path,
        output_format="rgba",
    )
    
    # Advanced: Construction only
    presets['advanced'] = WireframeConfig(
        enable_construction_lines=True,
        enable_mesh=False,
        enable_dexined_outline=False,
        enable_pose_landmarks=True,
        pose_model_path=default_pose_path,
        output_format="rgba",
    )
    
    # Outline only: DexiNed outline
    presets['outline_only'] = WireframeConfig(
        enable_construction_lines=False,
        enable_mesh=False,
        enable_dexined_outline=True,
        enable_pose_landmarks=True,
        dexined_model_path=default_dexined_path,
        pose_model_path=default_pose_path,
        output_format="rgba",
    )
    
    # Mesh only: Face mesh only with moderate enhancement
    presets['mesh_only'] = WireframeConfig(
        enable_construction_lines=False,
        enable_mesh=True,
        enable_dexined_outline=False,
        enable_pose_landmarks=True,
        pose_model_path=default_pose_path,
        output_format="rgba",
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
    parser.add_argument('--pose-landmarks', action='store_true',
                       help='Enable pose landmarks detection')
    
    # Presets
    parser.add_argument('--preset', choices=['beginner', 'intermediate', 'advanced', 
                                           'outline_only', 'mesh_only'],
                       help='Use preset configuration')
    
    # Advanced options
    parser.add_argument('--dexined-model', 
                       default='../DexiNed/checkpoints/BIPED2CLASSIC/10_model.pth',
                       help='Path to DexiNed model')
    parser.add_argument('--pose-model',
                       default='../mediapipe_practice/pose_landmarker.task',
                       help='Path to pose landmarker model')
    parser.add_argument('--output-format', choices=['rgb', 'rgba', 'svg'],
                       default='rgba', help='Output format')
    parser.add_argument('--background-removal', 
                       choices=['lines_only', 'face_mask', 'color_diff', 'color_filter'],
                       default='lines_only', help='Background removal method')
    
    # SVG options
    parser.add_argument('--svg', action='store_true',
                       help='Enable SVG export (in addition to raster output)')
    parser.add_argument('--svg-output', help='SVG output file path')

    # Background merge options
    parser.add_argument('--background-merge', action='store_true',
                       help='Enable background merge with independent transparency controls')
    parser.add_argument('--background-dir',
                       default='/home/joonhyoung-lee/바탕화면/arti_outlines/image_processing/out_sample/clipped_images_bg',
                       help='Directory containing background images')
    parser.add_argument('--foreground-dir',
                       default='/home/joonhyoung-lee/바탕화면/arti_outlines/image_processing/out_sample/clipped_images_fg',
                       help='Directory containing foreground images for creative control')
    parser.add_argument('--foreground-transparency', type=int, default=100,
                       help='Foreground transparency level (0-100, where 0=transparent, 100=opaque)')
    parser.add_argument('--background-transparency', type=int, default=50,
                       help='Background transparency level (0-100, where 0=transparent, 100=opaque)')

    # Legacy compatibility (deprecated)
    parser.add_argument('--background-opacity', type=int,
                       help='Deprecated: use --background-transparency instead')

    args = parser.parse_args()
    
    # Create configuration
    if args.preset:
        presets = create_preset_configs()
        config = presets[args.preset]
        # Override with command line options
        config.output_format = args.output_format  # Override preset output format
        config.enable_svg_export = args.svg or args.output_format == 'svg'
        config.svg_output_path = args.svg_output or ""
        # Override background merge settings
        config.enable_background_merge = args.background_merge
        config.background_directory = args.background_dir
        config.foreground_directory = args.foreground_dir
        config.foreground_transparency = args.foreground_transparency
        config.background_transparency = args.background_transparency

        # Handle legacy compatibility
        if args.background_opacity is not None:
            print("Warning: --background-opacity is deprecated, use --background-transparency instead")
            config.background_transparency = args.background_opacity
    else:
        # Handle legacy compatibility for background opacity
        bg_transparency = args.background_transparency
        if args.background_opacity is not None:
            print("Warning: --background-opacity is deprecated, use --background-transparency instead")
            bg_transparency = args.background_opacity

        config = WireframeConfig(
            enable_construction_lines=args.construction_lines,
            enable_mesh=args.mesh,
            enable_dexined_outline=args.dexined,
            enable_pose_landmarks=args.pose_landmarks,
            output_format=args.output_format,
            background_removal_method=args.background_removal,
            enable_svg_export=args.svg or args.output_format == 'svg',
            svg_output_path=args.svg_output or "",
            enable_background_merge=args.background_merge,
            background_directory=args.background_dir,
            foreground_directory=args.foreground_dir,
            foreground_transparency=args.foreground_transparency,
            background_transparency=bg_transparency
        )
    
    # Set DexiNed model path - use absolute path
    if args.dexined_model.startswith('../'):
        # Convert relative path to absolute
        config.dexined_model_path = os.path.abspath(args.dexined_model)
    else:
        config.dexined_model_path = args.dexined_model
    
    # Set pose model path - use absolute path
    if args.pose_model.startswith('../'):
        # Convert relative path to absolute, relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config.pose_model_path = os.path.abspath(os.path.join(script_dir, args.pose_model))
    else:
        config.pose_model_path = args.pose_model
    
    # Process image
    processor = WireframePortraitProcessor(config)
    results = processor.process_image(args.input, args.output)
    
    if results:
        print("Wireframe processing completed successfully!")
        print(f"Features used:")
        print(f"  Construction Lines: {'✓' if config.enable_construction_lines else '✗'}")
        print(f"  Face Mesh: {'✓' if config.enable_mesh else '✗'}")
        print(f"  DexiNed Outline: {'✓' if config.enable_dexined_outline else '✗'}")
        print(f"  Pose Landmarks: {'✓' if config.enable_pose_landmarks else '✗'}")
        print(f"  Background Merge: {'✓' if config.enable_background_merge else '✗'}")
    else:
        print("Wireframe processing failed!")

if __name__ == '__main__':
    main()