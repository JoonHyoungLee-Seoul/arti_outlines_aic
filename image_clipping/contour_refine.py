#!/usr/bin/env python3
"""
Contour Refinement Module for Portrait Outline Generation
Extracts, simplifies, and smooths contours for clean outline generation.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

class ContourRefiner:
    """Refine and process contours for clean outline generation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_contours(self, edges: np.ndarray, 
                        mode: int = cv2.RETR_EXTERNAL,
                        method: int = cv2.CHAIN_APPROX_SIMPLE,
                        min_contour_area: int = 100) -> List[np.ndarray]:
        """
        Extract contours from edge image
        
        Args:
            edges: Binary edge image
            mode: Contour retrieval mode
            method: Contour approximation method
            min_contour_area: Minimum contour area to keep
            
        Returns:
            List of contour arrays
        """
        # Find contours
        contours, hierarchy = cv2.findContours(edges, mode, method)
        
        # Filter contours by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_contour_area:
                filtered_contours.append(contour)
        
        # Sort by area (largest first)
        filtered_contours.sort(key=cv2.contourArea, reverse=True)
        
        self.logger.debug(f"Extracted {len(filtered_contours)} contours (min_area: {min_contour_area})")
        return filtered_contours
    
    def douglas_peucker_simplify(self, contour: np.ndarray, epsilon_factor: float = 0.005) -> np.ndarray:
        """
        Simplify contour using Douglas-Peucker algorithm
        
        Args:
            contour: Input contour points
            epsilon_factor: Approximation accuracy factor (relative to contour perimeter)
            
        Returns:
            Simplified contour
        """
        # Calculate epsilon based on contour perimeter
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = epsilon_factor * perimeter
        
        # Apply Douglas-Peucker approximation
        simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
        
        self.logger.debug(f"Simplified contour: {len(contour)} -> {len(simplified)} points")
        return simplified
    
    def chaikin_smooth(self, contour: np.ndarray, iterations: int = 2, 
                      ratio: float = 0.25) -> np.ndarray:
        """
        Apply Chaikin curve smoothing to contour
        
        Args:
            contour: Input contour points
            iterations: Number of smoothing iterations
            ratio: Smoothing ratio (0.25 = quarter-point subdivision)
            
        Returns:
            Smoothed contour
        """
        if len(contour) < 3:
            return contour
        
        # Convert contour to simple point array
        if contour.ndim == 3:
            points = contour[:, 0, :]  # Remove middle dimension
        else:
            points = contour
        
        smoothed = points.copy()
        
        for iteration in range(iterations):
            new_points = []
            n = len(smoothed)
            
            for i in range(n):
                # Current and next point (wrap around for closed curve)
                p1 = smoothed[i]
                p2 = smoothed[(i + 1) % n]
                
                # Calculate quarter points
                q1 = p1 + ratio * (p2 - p1)
                q2 = p1 + (1 - ratio) * (p2 - p1)
                
                new_points.append(q1)
                new_points.append(q2)
            
            smoothed = np.array(new_points)
        
        # Convert back to contour format
        if contour.ndim == 3:
            smoothed = smoothed.reshape(-1, 1, 2)
        
        self.logger.debug(f"Chaikin smoothing: {len(contour)} -> {len(smoothed)} points ({iterations} iterations)")
        return smoothed.astype(np.int32)
    
    def adaptive_smooth(self, contour: np.ndarray, smoothing_strength: str = 'medium') -> np.ndarray:
        """
        Apply adaptive smoothing based on contour characteristics
        
        Args:
            contour: Input contour
            smoothing_strength: 'light', 'medium', 'heavy'
            
        Returns:
            Adaptively smoothed contour
        """
        # Parameters based on smoothing strength
        params = {
            'light': {'iterations': 1, 'ratio': 0.2, 'epsilon_factor': 0.008},
            'medium': {'iterations': 2, 'ratio': 0.25, 'epsilon_factor': 0.005},
            'heavy': {'iterations': 3, 'ratio': 0.3, 'epsilon_factor': 0.003}
        }
        
        config = params.get(smoothing_strength, params['medium'])
        
        # First simplify, then smooth
        simplified = self.douglas_peucker_simplify(contour, config['epsilon_factor'])
        smoothed = self.chaikin_smooth(simplified, config['iterations'], config['ratio'])
        
        return smoothed
    
    def filter_contours_by_properties(self, contours: List[np.ndarray],
                                     min_area: int = 500,
                                     max_area: Optional[int] = None,
                                     min_perimeter: int = 100,
                                     aspect_ratio_range: Tuple[float, float] = (0.1, 10.0)) -> List[np.ndarray]:
        """
        Filter contours based on geometric properties
        
        Args:
            contours: List of contours
            min_area: Minimum contour area
            max_area: Maximum contour area (None = no limit)
            min_perimeter: Minimum contour perimeter
            aspect_ratio_range: (min_ratio, max_ratio) for bounding rectangle
            
        Returns:
            Filtered list of contours
        """
        filtered = []
        
        for contour in contours:
            # Area filter
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            if max_area and area > max_area:
                continue
            
            # Perimeter filter
            perimeter = cv2.arcLength(contour, closed=True)
            if perimeter < min_perimeter:
                continue
            
            # Aspect ratio filter
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width > 0 and height > 0:
                aspect_ratio = max(width/height, height/width)
                if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
                    continue
            
            filtered.append(contour)
        
        self.logger.debug(f"Filtered contours: {len(contours)} -> {len(filtered)}")
        return filtered
    
    def process_contours(self, edges: np.ndarray, 
                        simplify: bool = True,
                        smooth: bool = True,
                        smoothing_strength: str = 'medium',
                        filter_properties: bool = True) -> List[np.ndarray]:
        """
        Complete contour processing pipeline
        
        Args:
            edges: Binary edge image
            simplify: Whether to apply Douglas-Peucker simplification
            smooth: Whether to apply Chaikin smoothing
            smoothing_strength: Smoothing intensity
            filter_properties: Whether to filter by geometric properties
            
        Returns:
            List of processed contours
        """
        # Extract contours
        contours = self.extract_contours(edges)
        
        if not contours:
            self.logger.warning("No contours found in edge image")
            return []
        
        # Filter by properties
        if filter_properties:
            contours = self.filter_contours_by_properties(contours)
        
        # Process each contour
        processed_contours = []
        for contour in contours:
            processed = contour.copy()
            
            if simplify and smooth:
                # Combined adaptive smoothing (includes both steps)
                processed = self.adaptive_smooth(processed, smoothing_strength)
            elif simplify:
                # Only simplification
                processed = self.douglas_peucker_simplify(processed)
            elif smooth:
                # Only smoothing
                processed = self.chaikin_smooth(processed)
            
            processed_contours.append(processed)
        
        return processed_contours
    
    def render_contours(self, contours: List[np.ndarray], 
                       image_size: Tuple[int, int],
                       line_thickness: int = 2,
                       line_color: Tuple[int, int, int] = (0, 0, 0),
                       background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        Render contours to create clean outline image
        
        Args:
            contours: List of processed contours
            image_size: (height, width) of output image
            line_thickness: Thickness of contour lines
            line_color: RGB color for contour lines
            background_color: RGB color for background
            
        Returns:
            Rendered outline image
        """
        h, w = image_size
        outline_img = np.full((h, w, 3), background_color, dtype=np.uint8)
        
        # Draw contours
        for i, contour in enumerate(contours):
            cv2.drawContours(outline_img, [contour], -1, line_color, 
                           thickness=line_thickness, lineType=cv2.LINE_AA)
        
        return outline_img
    
    def get_contour_statistics(self, contours: List[np.ndarray]) -> dict:
        """Get statistical information about contours"""
        if not contours:
            return {"count": 0}
        
        areas = [cv2.contourArea(c) for c in contours]
        perimeters = [cv2.arcLength(c, closed=True) for c in contours]
        
        return {
            "count": len(contours),
            "total_area": sum(areas),
            "mean_area": np.mean(areas),
            "median_area": np.median(areas),
            "mean_perimeter": np.mean(perimeters),
            "largest_contour_area": max(areas) if areas else 0
        }

# Convenience functions
def process_contours(edges: np.ndarray, **kwargs) -> List[np.ndarray]:
    """Standalone function for contour processing"""
    refiner = ContourRefiner()
    return refiner.process_contours(edges, **kwargs)

def render_outline(contours: List[np.ndarray], image_size: Tuple[int, int], **kwargs) -> np.ndarray:
    """Standalone function for outline rendering"""
    refiner = ContourRefiner()
    return refiner.render_contours(contours, image_size, **kwargs)