#!/usr/bin/env python3
"""
Face Detection and Guide Line Generation Module
Uses Mediapipe FaceDetection to extract facial features and generate construction guides.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional, List
import logging

class FaceGuideGenerator:
    """Generate construction guide lines using face detection"""
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """Initialize face detection model"""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for close-range, 1 for full-range detection
            min_detection_confidence=min_detection_confidence
        )
        
        # Face landmarks model for more precise eye detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence
        )
        
        self.logger = logging.getLogger(__name__)
    
    def detect_face_features(self, image: np.ndarray) -> Dict:
        """
        Detect face and extract key features for guide line generation
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary containing face bbox, eye positions, and calculated guide lines
        """
        if len(image.shape) == 4:  # RGBA
            rgb_image = cv2.cvtColor(image[:,:,:3], cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = rgb_image.shape[:2]
        
        # Basic face detection for bbox
        face_results = self.face_detection.process(rgb_image)
        
        # Face mesh for precise landmarks
        mesh_results = self.face_mesh.process(rgb_image)
        
        if not face_results.detections:
            self.logger.warning("No face detected in image")
            return self._generate_default_guides(w, h)
        
        # Get the largest/most confident face
        main_face = max(face_results.detections, key=lambda x: x.score[0])
        bbox = main_face.location_data.relative_bounding_box
        
        # Convert relative coordinates to absolute
        face_bbox = {
            'x': int(bbox.xmin * w),
            'y': int(bbox.ymin * h), 
            'width': int(bbox.width * w),
            'height': int(bbox.height * h)
        }
        
        # Extract eye positions from face mesh if available
        eye_positions = self._extract_eye_positions(mesh_results, w, h)
        
        # Generate guide lines
        guide_lines = self._calculate_guide_lines(face_bbox, eye_positions, w, h)
        
        return {
            'face_bbox': face_bbox,
            'eye_positions': eye_positions,
            'guide_lines': guide_lines,
            'face_detected': True
        }
    
    def _extract_eye_positions(self, mesh_results, w: int, h: int) -> Dict:
        """Extract eye center positions from face mesh landmarks"""
        if not mesh_results.multi_face_landmarks:
            return {'left_eye': None, 'right_eye': None}
        
        landmarks = mesh_results.multi_face_landmarks[0]
        
        # Key eye landmarks (mediapipe face mesh indices)
        # Left eye (from viewer's perspective, person's right eye)
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Right eye (from viewer's perspective, person's left eye) 
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        def get_eye_center(indices: List[int]) -> Tuple[int, int]:
            eye_points = [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in indices]
            center_x = sum(p[0] for p in eye_points) / len(eye_points)
            center_y = sum(p[1] for p in eye_points) / len(eye_points)
            return (int(center_x), int(center_y))
        
        try:
            left_eye = get_eye_center(left_eye_indices)
            right_eye = get_eye_center(right_eye_indices)
            
            return {
                'left_eye': left_eye,
                'right_eye': right_eye
            }
        except Exception as e:
            self.logger.warning(f"Could not extract eye positions: {e}")
            return {'left_eye': None, 'right_eye': None}
    
    def _calculate_guide_lines(self, face_bbox: Dict, eye_positions: Dict, w: int, h: int) -> Dict:
        """Calculate vertical center line and eye level line"""
        
        # Vertical center line (from face bbox center)
        face_center_x = face_bbox['x'] + face_bbox['width'] // 2
        vertical_line = {
            'x1': face_center_x,
            'y1': 0,
            'x2': face_center_x, 
            'y2': h
        }
        
        # Eye level line
        if eye_positions['left_eye'] and eye_positions['right_eye']:
            # Use actual eye positions for more accurate line
            left_eye_y = eye_positions['left_eye'][1]
            right_eye_y = eye_positions['right_eye'][1]
            eye_level_y = (left_eye_y + right_eye_y) // 2
        else:
            # Fallback: estimate eye level at ~40% down from top of face bbox
            eye_level_y = face_bbox['y'] + int(face_bbox['height'] * 0.4)
        
        eye_line = {
            'x1': 0,
            'y1': eye_level_y,
            'x2': w,
            'y2': eye_level_y
        }
        
        return {
            'vertical_line': vertical_line,
            'eye_line': eye_line
        }
    
    def _generate_default_guides(self, w: int, h: int) -> Dict:
        """Generate default guide lines when no face is detected"""
        vertical_line = {
            'x1': w // 2,
            'y1': 0,
            'x2': w // 2,
            'y2': h
        }
        
        eye_line = {
            'x1': 0,
            'y1': h // 3,  # Rough estimate at 1/3 from top
            'x2': w,
            'y2': h // 3
        }
        
        return {
            'face_bbox': None,
            'eye_positions': {'left_eye': None, 'right_eye': None},
            'guide_lines': {
                'vertical_line': vertical_line,
                'eye_line': eye_line
            },
            'face_detected': False
        }
    
    def render_construction_guide(self, image: np.ndarray, guide_data: Dict, 
                                line_color: Tuple[int, int, int] = (180, 180, 180),
                                line_thickness: int = 1) -> np.ndarray:
        """
        Render construction guide lines on the image
        
        Args:
            image: Base outline image
            guide_data: Guide line data from detect_face_features()
            line_color: RGB color for guide lines (light gray by default)
            line_thickness: Line thickness in pixels
            
        Returns:
            Image with construction guide lines overlaid
        """
        result = image.copy()
        guide_lines = guide_data['guide_lines']
        
        # Draw vertical center line
        if 'vertical_line' in guide_lines:
            vl = guide_lines['vertical_line']
            cv2.line(result, (vl['x1'], vl['y1']), (vl['x2'], vl['y2']), 
                    line_color, line_thickness, cv2.LINE_AA)
        
        # Draw eye level line
        if 'eye_line' in guide_lines:
            el = guide_lines['eye_line']
            cv2.line(result, (el['x1'], el['y1']), (el['x2'], el['y2']), 
                    line_color, line_thickness, cv2.LINE_AA)
        
        # Optional: draw eye points for debugging
        if guide_data.get('face_detected', False):
            eyes = guide_data['eye_positions']
            if eyes['left_eye']:
                cv2.circle(result, eyes['left_eye'], 2, line_color, -1)
            if eyes['right_eye']:
                cv2.circle(result, eyes['right_eye'], 2, line_color, -1)
        
        return result
    
    def close(self):
        """Clean up mediapipe resources"""
        self.face_detection.close()
        self.face_mesh.close()

# Convenience functions for standalone usage
def generate_guide_lines(image: np.ndarray) -> Dict:
    """Standalone function to generate guide lines from an image"""
    generator = FaceGuideGenerator()
    try:
        return generator.detect_face_features(image)
    finally:
        generator.close()

def render_construction_guides(outline_image: np.ndarray, guide_data: Dict) -> np.ndarray:
    """Standalone function to render construction guides"""
    generator = FaceGuideGenerator()
    try:
        return generator.render_construction_guide(outline_image, guide_data)
    finally:
        generator.close()