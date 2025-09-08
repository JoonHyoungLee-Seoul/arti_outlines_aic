#!/usr/bin/env python3
"""
Edge Detection Module for Portrait Outline Generation
Implements various edge detection methods optimized for portrait outline extraction.
"""

import cv2
import numpy as np
import logging

class EdgeDetector:
    """Advanced edge detection for portrait outline generation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def preprocess_for_edges(self, image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """
        Preprocess image for optimal edge detection
        
        Args:
            image: Input image (BGR or BGRA)
            method: Preprocessing method ('bilateral', 'guided', 'gaussian')
            
        Returns:
            Preprocessed image ready for edge detection
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # BGRA
                # Use alpha channel for masking
                alpha = image[:, :, 3]
                gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
                # Apply alpha mask (set background to white)
                gray = np.where(alpha > 0, gray, 255)
            else:  # BGR
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply edge-friendly smoothing
        if method == 'bilateral':
            # Bilateral filter preserves edges while smoothing
            processed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        elif method == 'guided':
            # Guided filter (approximated with bilateral)
            processed = cv2.bilateralFilter(gray, d=15, sigmaColor=80, sigmaSpace=80)
        elif method == 'gaussian':
            # Simple Gaussian blur
            processed = cv2.GaussianBlur(gray, (5, 5), 1.0)
        else:
            processed = gray
        
        return processed
    
    def auto_canny(self, image: np.ndarray, sigma: float = 0.33, 
                   lower_percentile: float = 50, upper_percentile: float = 99) -> np.ndarray:
        """
        Automatic Canny edge detection with adaptive thresholding
        
        Args:
            image: Grayscale input image
            sigma: Controls the ratio of low to high thresholds
            lower_percentile: Percentile for automatic lower threshold
            upper_percentile: Percentile for automatic upper threshold
            
        Returns:
            Binary edge image
        """
        # Calculate automatic thresholds based on image statistics
        median_intensity = np.percentile(image, 50)
        lower_threshold = max(0, int((1.0 - sigma) * median_intensity))
        upper_threshold = min(255, int((1.0 + sigma) * median_intensity))
        
        # Alternative: Use intensity percentiles
        alt_lower = np.percentile(image, lower_percentile) * 0.5
        alt_upper = np.percentile(image, upper_percentile) * 0.7
        
        # Use more conservative thresholds
        final_lower = max(lower_threshold, alt_lower)
        final_upper = min(upper_threshold, alt_upper)
        
        # Apply Canny edge detection
        edges = cv2.Canny(image, final_lower, final_upper, 
                         apertureSize=3, L2gradient=True)
        
        self.logger.debug(f"Auto-Canny thresholds: {final_lower:.1f}, {final_upper:.1f}")
        return edges
    
    def multiscale_canny(self, image: np.ndarray, scales: list = [1.0, 0.8, 1.2]) -> np.ndarray:
        """
        Multi-scale Canny edge detection for robust edge extraction
        
        Args:
            image: Grayscale input image
            scales: List of scale factors for multi-scale processing
            
        Returns:
            Combined edge image from multiple scales
        """
        h, w = image.shape
        combined_edges = np.zeros((h, w), dtype=np.uint8)
        
        for scale in scales:
            if scale != 1.0:
                # Resize image
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Detect edges
                edges = self.auto_canny(scaled_img)
                
                # Resize back to original size
                edges = cv2.resize(edges, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                edges = self.auto_canny(image)
            
            # Combine edges (OR operation)
            combined_edges = cv2.bitwise_or(combined_edges, edges)
        
        return combined_edges
    
    def structured_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Structured edge detection using multiple gradient operators
        
        Args:
            image: Grayscale input image
            
        Returns:
            Combined structured edges
        """
        # Sobel gradients
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Scharr gradients (more accurate)
        scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        scharr_magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
        
        # Laplacian
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
        laplacian = np.abs(laplacian)
        
        # Combine different edge operators
        combined = 0.4 * sobel_magnitude + 0.4 * scharr_magnitude + 0.2 * laplacian
        
        # Normalize and threshold
        combined = ((combined - combined.min()) / (combined.max() - combined.min()) * 255).astype(np.uint8)
        
        # Apply adaptive threshold using OpenCV
        threshold_value, _ = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_edges = combined > threshold_value
        
        return (binary_edges * 255).astype(np.uint8)
    
    def detect_edges(self, image: np.ndarray, method: str = 'auto_canny', 
                    preprocess_method: str = 'bilateral', **kwargs) -> np.ndarray:
        """
        Main edge detection function with multiple methods
        
        Args:
            image: Input image (BGR or BGRA)
            method: Edge detection method ('auto_canny', 'multiscale_canny', 'structured')
            preprocess_method: Preprocessing method
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Binary edge image
        """
        # Preprocess image
        processed = self.preprocess_for_edges(image, preprocess_method)
        
        # Apply selected edge detection method
        if method == 'auto_canny':
            edges = self.auto_canny(processed, **kwargs)
        elif method == 'multiscale_canny':
            scales = kwargs.get('scales', [1.0, 0.8, 1.2])
            edges = self.multiscale_canny(processed, scales=scales)
        elif method == 'structured':
            edges = self.structured_edge_detection(processed)
        else:
            # Fallback to basic Canny
            edges = cv2.Canny(processed, 50, 150)
        
        return edges
    
    def refine_edges(self, edges: np.ndarray, 
                    remove_small_components: bool = True,
                    min_component_size: int = 100,
                    apply_morphology: bool = True) -> np.ndarray:
        """
        Refine detected edges by removing noise and small components
        
        Args:
            edges: Binary edge image
            remove_small_components: Whether to remove small connected components
            min_component_size: Minimum size for connected components
            apply_morphology: Whether to apply morphological operations
            
        Returns:
            Refined edge image
        """
        refined = edges.copy()
        
        if apply_morphology:
            # Close gaps in edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
            
            # Remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
        
        if remove_small_components:
            # Remove small connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                refined, connectivity=8, dtype=cv2.CV_32S
            )
            
            # Create mask for components larger than threshold
            large_component_mask = np.zeros_like(refined)
            for i in range(1, num_labels):  # Skip background (label 0)
                if stats[i, cv2.CC_STAT_AREA] >= min_component_size:
                    large_component_mask[labels == i] = 255
            
            refined = large_component_mask
        
        return refined

# Convenience functions
def detect_edges(image: np.ndarray, method: str = 'auto_canny', **kwargs) -> np.ndarray:
    """Standalone function for edge detection"""
    detector = EdgeDetector()
    return detector.detect_edges(image, method=method, **kwargs)

def preprocess_image(image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
    """Standalone function for image preprocessing"""
    detector = EdgeDetector()
    return detector.preprocess_for_edges(image, method=method)