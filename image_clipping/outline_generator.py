#!/usr/bin/env python3
"""
Portrait Outline Generator
Generates clean outlines and construction guides from segmented foreground images.

Input: ./out/clipped_images_fg/*.png (foreground portraits)
Output: 
  - ./out/outlines/*_outline.png (clean outline drawings)
  - ./out/construction/*_construction.png (with guide lines)
  - ./out/vectors/*_outline.svg (vector format, optional)
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import logging

# Import our custom modules
from face_guide import FaceGuideGenerator
from edge_detection import EdgeDetector
from contour_refine import ContourRefiner

# === Default Configuration ===
DEFAULT_FG_DIR = "./out/clipped_images_fg"
DEFAULT_OUTLINE_DIR = "./out/outlines"
DEFAULT_CONSTRUCTION_DIR = "./out/construction"
DEFAULT_VECTOR_DIR = "./out/vectors"

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outline_generator.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_foreground_files(fg_dir: str) -> list:
    """Get all foreground PNG files from input directory"""
    fg_path = Path(fg_dir)
    if not fg_path.exists():
        raise FileNotFoundError(f"Foreground directory not found: {fg_dir}")
    
    files = list(fg_path.glob("*_fg.png"))
    return sorted([str(f) for f in files])

def process_single_image(input_path: str, outline_dir: str, construction_dir: str, vector_dir: str = None):
    """Process a single foreground image to generate outline and construction guide"""
    try:
        # Load foreground image
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        # Extract base filename
        base_name = Path(input_path).stem.replace("_fg", "")
        
        # Initialize processors
        face_guide_gen = FaceGuideGenerator()
        edge_detector = EdgeDetector()
        contour_refiner = ContourRefiner()
        
        try:
            # Detect face features and generate guide lines
            guide_data = face_guide_gen.detect_face_features(img)
            
            # Detect edges
            edges = edge_detector.detect_edges(img, method='auto_canny')
            
            # Process contours
            contours = contour_refiner.process_contours(edges, smoothing_strength='medium')
            
            # Generate clean outline image
            h, w = img.shape[:2]
            outline_img = contour_refiner.render_contours(contours, (h, w), line_thickness=2)
            
            # Generate construction guide with face guides
            construction_img = face_guide_gen.render_construction_guide(outline_img, guide_data)
            
            # Save outputs
            cv2.imwrite(os.path.join(outline_dir, f"{base_name}_outline.png"), outline_img)
            cv2.imwrite(os.path.join(construction_dir, f"{base_name}_construction.png"), construction_img)
            
            if vector_dir:
                # Generate simple SVG (basic implementation)
                generate_svg_outline(contours, guide_data, os.path.join(vector_dir, f"{base_name}_outline.svg"), (w, h))
            
            return True
            
        finally:
            # Clean up resources
            face_guide_gen.close()
        
    except Exception as e:
        logging.error(f"Error processing {input_path}: {str(e)}")
        return False

# This function is now handled by EdgeDetector.preprocess_for_edges()

# This function is now handled by FaceGuideGenerator.detect_face_features()

# This function is now handled by EdgeDetector.detect_edges()

# This function is now handled by ContourRefiner.process_contours()

# This function is now handled by ContourRefiner.render_contours()

# This function is now handled by FaceGuideGenerator.render_construction_guide()

def generate_svg_outline(contours, guide_data, output_path: str, image_size):
    """Generate basic SVG vector format"""
    w, h = image_size
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">
'''
    
    # Add contour paths
    for i, contour in enumerate(contours):
        if len(contour) > 2:
            path_data = f"M {contour[0][0][0]},{contour[0][0][1]}"
            for point in contour[1:]:
                path_data += f" L {point[0][0]},{point[0][1]}"
            path_data += " Z"
            
            svg_content += f'  <path d="{path_data}" fill="none" stroke="black" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>\n'
    
    # Add guide lines
    if guide_data.get('guide_lines'):
        guides = guide_data['guide_lines']
        if 'vertical_line' in guides:
            vl = guides['vertical_line']
            svg_content += f'  <line x1="{vl["x1"]}" y1="{vl["y1"]}" x2="{vl["x2"]}" y2="{vl["y2"]}" stroke="#AAAAAA" stroke-width="1" stroke-dasharray="5,5"/>\n'
        
        if 'eye_line' in guides:
            el = guides['eye_line']
            svg_content += f'  <line x1="{el["x1"]}" y1="{el["y1"]}" x2="{el["x2"]}" y2="{el["y2"]}" stroke="#AAAAAA" stroke-width="1" stroke-dasharray="5,5"/>\n'
    
    svg_content += '</svg>'
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
    except Exception as e:
        logging.warning(f"Could not save SVG: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Portrait Outline Generator")
    parser.add_argument("-i", "--input", default=DEFAULT_FG_DIR,
                        help=f"Input foreground images directory (default: {DEFAULT_FG_DIR})")
    parser.add_argument("--outline-dir", default=DEFAULT_OUTLINE_DIR,
                        help=f"Output directory for outlines (default: {DEFAULT_OUTLINE_DIR})")
    parser.add_argument("--construction-dir", default=DEFAULT_CONSTRUCTION_DIR,
                        help=f"Output directory for construction guides (default: {DEFAULT_CONSTRUCTION_DIR})")
    parser.add_argument("--vector-dir", default=None,
                        help=f"Output directory for SVG vectors (default: disabled)")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Maximum number of parallel workers (default: 4)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Portrait Outline Generator")
    
    # Create output directories
    for dir_path in [args.outline_dir, args.construction_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    if args.vector_dir:
        os.makedirs(args.vector_dir, exist_ok=True)
    
    # Get input files
    try:
        input_files = get_foreground_files(args.input)
        logger.info(f"Found {len(input_files)} foreground images to process")
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    # Process images
    success_count = 0
    with tqdm(input_files, desc="Processing images") as pbar:
        for input_path in pbar:
            if process_single_image(input_path, args.outline_dir, args.construction_dir, args.vector_dir):
                success_count += 1
            pbar.set_postfix({"Success": f"{success_count}/{len(input_files)}"})
    
    logger.info(f"Processing complete: {success_count}/{len(input_files)} images successful")
    return 0

if __name__ == "__main__":
    sys.exit(main())