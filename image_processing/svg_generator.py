"""
SVG Generator for Wireframe Portrait Processing
Converts wireframe elements to scalable vector graphics for frontend/backend integration.
"""

import numpy as np
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom


class SVGGenerator:
    """Generates SVG output for wireframe portrait elements."""
    
    def __init__(self, width: int, height: int, background_color: str = "white"):
        """
        Initialize SVG generator.
        
        Args:
            width: Canvas width
            height: Canvas height
            background_color: Background color for the SVG
        """
        self.width = width
        self.height = height
        self.background_color = background_color
        # Build the <svg> root element once and reuse it for subsequent calls.
        self.svg_root = self._create_svg_root()
    
    def _create_svg_root(self) -> ET.Element:
        """Create the root SVG element."""
        svg = ET.Element('svg')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('width', str(self.width))
        svg.set('height', str(self.height))
        svg.set('viewBox', f'0 0 {self.width} {self.height}')

        # Draw a background rectangle so exported files preview correctly in
        # browsers that default to transparent SVG canvases.
        if self.background_color:
            background = ET.SubElement(svg, 'rect')
            background.set('width', '100%')
            background.set('height', '100%')
            background.set('fill', self.background_color)

        return svg
    
    def add_construction_lines(self, landmarks: np.ndarray, config: dict):
        """
        Add construction lines to SVG based on actual face landmarks.
        
        Args:
            landmarks: MediaPipe face landmarks (normalized coordinates)
            config: Configuration dictionary with line properties
        """
        group = ET.SubElement(self.svg_root, 'g')
        group.set('id', 'construction-lines')  # group for easy styling
        
        color = config.get('color', '#FF0000')
        thickness = config.get('thickness', 2)
        
        # MediaPipe landmarks are normalized [0,1]. Convert them to absolute
        # pixel coordinates for the SVG canvas.
        pixel_landmarks = [(int(lm[0] * self.width), int(lm[1] * self.height)) for lm in landmarks]
        
        def get_pixel_coords(landmark_idx):
            """Get pixel coordinates for a landmark index"""
            if landmark_idx >= len(pixel_landmarks):
                return None
            return pixel_landmarks[landmark_idx]
        
        def add_line_through_landmarks(landmark_indices, line_id, line_color=None):
            """Add line connecting multiple landmarks"""
            if line_color is None:
                line_color = color
                
            points = []
            for idx in landmark_indices:
                point = get_pixel_coords(idx)
                if point:
                    points.append(point)

            # Draw line segments between each successive pair of landmarks.
            for i in range(len(points) - 1):
                line_id_segment = f"{line_id}-{i}"
                self._add_line(
                    group,
                    points[i][0], points[i][1],
                    points[i+1][0], points[i+1][1],
                    line_color, thickness, line_id_segment
                )
        
        # Use the same landmark connections as the original wireframe processor
        
        # 1. Vertical Center Line [10→168→4→152]
        add_line_through_landmarks([10, 168, 4, 152], 'center-line')
        
        # 2. Eyebrow Line [63→293] 
        add_line_through_landmarks([63, 293], 'eyebrow-line')
        
        # 3. Eye Lines [33→263], [133→362], [145→159], [374→386]
        add_line_through_landmarks([33, 263], 'eye-line-outer')
        add_line_through_landmarks([133, 362], 'eye-line-inner')
        add_line_through_landmarks([145, 159], 'eye-line-left')
        add_line_through_landmarks([374, 386], 'eye-line-right')
        
        # 4. Nose Line [48→278]
        add_line_through_landmarks([48, 278], 'nose-line')
        
        # 5. Mouth Line [61→291]
        add_line_through_landmarks([61, 291], 'mouth-line')
    
    def add_face_mesh(self, landmarks: np.ndarray, connections: List[Tuple[int, int]], config: dict):
        """
        Add face mesh to SVG.
        
        Args:
            landmarks: MediaPipe face landmarks (normalized coordinates)
            connections: Face mesh connections
            config: Configuration dictionary with mesh properties
        """
        group = ET.SubElement(self.svg_root, 'g')
        group.set('id', 'face-mesh')
        
        color = config.get('color', '#00FF00')
        thickness = config.get('thickness', 1)
        
        # Convert normalized coordinates to pixel coordinates
        pixel_landmarks = [(int(lm[0] * self.width), int(lm[1] * self.height)) for lm in landmarks]
        
        # Add mesh connections
        for connection in connections:
            if connection[0] < len(pixel_landmarks) and connection[1] < len(pixel_landmarks):
                start_point = pixel_landmarks[connection[0]]
                end_point = pixel_landmarks[connection[1]]
                self._add_line(group, start_point[0], start_point[1], 
                             end_point[0], end_point[1], color, thickness, 
                             f'mesh-{connection[0]}-{connection[1]}')
    
    def add_edge_outline(self, edge_points: List[Tuple[int, int]], config: dict):
        """
        Add edge outline to SVG.
        
        Args:
            edge_points: List of edge points (pixel coordinates)
            config: Configuration dictionary with outline properties
        """
        group = ET.SubElement(self.svg_root, 'g')
        group.set('id', 'edge-outline')
        
        color = config.get('color', '#0000FF')
        thickness = config.get('thickness', 1)
        
        if len(edge_points) < 2:
            return  # nothing to draw
        
        # Create path element for smooth curves
        path = ET.SubElement(group, 'path')
        path.set('id', 'edge-path')
        path.set('stroke', color)
        path.set('stroke-width', str(thickness))
        path.set('fill', 'none')
        
        # Build path data
        path_data = f'M {edge_points[0][0]} {edge_points[0][1]}'
        
        for i in range(1, len(edge_points)):
            path_data += f' L {edge_points[i][0]} {edge_points[i][1]}'
        
        path.set('d', path_data)
    
    def add_dexined_outline(self, contours: List[np.ndarray], config: dict):
        """
        Add DexiNed processed outline to SVG with improved path generation.
        
        Args:
            contours: OpenCV contours from DexiNed processing
            config: Configuration dictionary with outline properties
        """
        group = ET.SubElement(self.svg_root, 'g')
        group.set('id', 'dexined-outline')
        
        color = config.get('color', '#000000')  # Black for better contrast
        thickness = config.get('thickness', 1.5)  # Slightly thicker for better visibility
        
        for i, contour in enumerate(contours):
            if len(contour) < 4:  # Need at least 4 points for meaningful contour
                continue
                
            path = ET.SubElement(group, 'path')
            path.set('id', f'contour-{i}')
            path.set('stroke', color)
            path.set('stroke-width', str(thickness))
            path.set('fill', 'none')
            path.set('stroke-linecap', 'round')  # Rounded line caps for smoother appearance
            path.set('stroke-linejoin', 'round')  # Rounded line joins
            
            # Build path data from contour points with curve optimization
            points = contour.reshape(-1, 2)
            
            if len(points) < 4:
                # Simple line for very short contours
                path_data = f'M {points[0][0]} {points[0][1]}'
                for point in points[1:]:
                    path_data += f' L {point[0]} {point[1]}'
            else:
                # Use cubic Bezier curves for smoother paths
                path_data = f'M {points[0][0]} {points[0][1]}'
                
                # Generate smooth curve through points
                for j in range(1, len(points)):
                    if j == len(points) - 1:
                        # Last point - simple line to close
                        path_data += f' L {points[j][0]} {points[j][1]}'
                    else:
                        # Use quadratic curves for smoother paths
                        path_data += f' L {points[j][0]} {points[j][1]}'
            
            # Close the path if it forms a meaningful closed shape
            contour_area = len(points)
            is_closed_contour = (abs(points[0][0] - points[-1][0]) < 5 and 
                               abs(points[0][1] - points[-1][1]) < 5)
            
            if contour_area > 6 and is_closed_contour:
                path_data += ' Z'
            
            path.set('d', path_data)
    
    def _add_line(self, parent: ET.Element, x1: int, y1: int, x2: int, y2: int,
                  color: str, thickness: int, line_id: str):
        """Add a line element to the parent group."""
        line = ET.SubElement(parent, 'line')
        line.set('id', line_id)
        line.set('x1', str(x1))
        line.set('y1', str(y1))
        line.set('x2', str(x2))
        line.set('y2', str(y2))
        line.set('stroke', color)
        line.set('stroke-width', str(thickness))
    
    def add_pose_landmarks(self, pose_landmarks: np.ndarray, config: dict):
        """
        Add pose landmarks and body skeleton to SVG.
        
        Args:
            pose_landmarks: Pose landmark coordinates array (shape: [33, 3])
            config: Configuration dictionary containing:
                   - line_color: Color for body connections
                   - point_color: Color for landmark points  
                   - line_thickness: Thickness of connection lines
                   - point_radius: Radius of landmark points
                   - connections: List of connection tuples
                   - excluded_landmarks: Set of landmark indices to exclude
        """
        if pose_landmarks.size == 0:
            return
            
        # Extract configuration
        line_color = config.get('line_color', 'rgb(255, 165, 0)')  # Orange
        point_color = config.get('point_color', 'rgb(255, 0, 0)')  # Red
        line_thickness = config.get('line_thickness', 2)
        point_radius = config.get('point_radius', 4)
        connections = config.get('connections', [])
        excluded_landmarks = config.get('excluded_landmarks', set())
        
        # Create group for pose landmarks
        pose_group = ET.SubElement(self.svg_root, 'g')
        pose_group.set('id', 'pose-landmarks')
        pose_group.set('class', 'wireframe-pose')
        
        # Add connection lines (body skeleton)
        connections_group = ET.SubElement(pose_group, 'g')
        connections_group.set('id', 'pose-connections')
        
        for start_idx, end_idx in connections:
            # Skip if landmarks are excluded or out of bounds
            if (start_idx in excluded_landmarks or 
                end_idx in excluded_landmarks or
                start_idx >= len(pose_landmarks) or 
                end_idx >= len(pose_landmarks)):
                continue
                
            start_landmark = pose_landmarks[start_idx]
            end_landmark = pose_landmarks[end_idx]
            
            # Convert normalized coordinates to pixel coordinates
            x1 = start_landmark[0] * self.width
            y1 = start_landmark[1] * self.height
            x2 = end_landmark[0] * self.width
            y2 = end_landmark[1] * self.height
            
            # Create connection line
            line = ET.SubElement(connections_group, 'line')
            line.set('x1', str(x1))
            line.set('y1', str(y1))
            line.set('x2', str(x2))
            line.set('y2', str(y2))
            line.set('stroke', line_color)
            line.set('stroke-width', str(line_thickness))
            line.set('stroke-linecap', 'round')
            line.set('stroke-linejoin', 'round')
        
        # Add landmark points
        points_group = ET.SubElement(pose_group, 'g')
        points_group.set('id', 'pose-points')
        
        for idx, landmark in enumerate(pose_landmarks):
            # Skip excluded landmarks
            if idx in excluded_landmarks:
                continue
                
            # Convert normalized coordinates to pixel coordinates
            x = landmark[0] * self.width
            y = landmark[1] * self.height
            
            # Create landmark point
            circle = ET.SubElement(points_group, 'circle')
            circle.set('cx', str(x))
            circle.set('cy', str(y))
            circle.set('r', str(point_radius))
            circle.set('fill', point_color)
            circle.set('stroke', 'none')
    
    def add_metadata(self, metadata: dict):
        """Add metadata to SVG."""
        desc = ET.SubElement(self.svg_root, 'desc')
        desc.text = f"Wireframe Portrait - Generated with settings: {metadata}"

        # Store additional machine-readable metadata as <meta> tags.
        metadata_group = ET.SubElement(self.svg_root, 'metadata')
        for key, value in metadata.items():
            meta = ET.SubElement(metadata_group, 'meta')
            meta.set('name', key)
            meta.set('content', str(value))
    
    def to_string(self, pretty: bool = True) -> str:
        """
        Convert SVG to string representation.
        
        Args:
            pretty: Whether to format with nice indentation
            
        Returns:
            SVG as string
        """
        if pretty:
            rough_string = ET.tostring(self.svg_root, 'unicode')
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="  ")
        else:
            return ET.tostring(self.svg_root, 'unicode')
    
    def save(self, filepath: str):
        """Save SVG to file."""
        svg_string = self.to_string(pretty=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(svg_string)
    
    def get_viewbox_for_zoom(self, zoom_factor: float, center_x: float, center_y: float) -> str:
        """
        Generate viewBox for zoomed view.
        
        Args:
            zoom_factor: Zoom level (1.0 = normal, 2.0 = 2x zoom)
            center_x: X coordinate of zoom center (0-1 normalized)
            center_y: Y coordinate of zoom center (0-1 normalized)
            
        Returns:
            ViewBox string for zoomed view
        """
        # Calculate zoomed dimensions
        zoomed_width = self.width / zoom_factor
        zoomed_height = self.height / zoom_factor

        # Calculate top-left corner so that the viewBox is centered on the
        # requested point.
        x = (center_x * self.width) - (zoomed_width / 2)
        y = (center_y * self.height) - (zoomed_height / 2)

        # Clamp to bounds to avoid panning outside the canvas.
        x = max(0, min(x, self.width - zoomed_width))
        y = max(0, min(y, self.height - zoomed_height))

        return f"{x} {y} {zoomed_width} {zoomed_height}"
    
    def create_zoomed_version(self, zoom_factor: float, center_x: float, center_y: float) -> str:
        """
        Create a zoomed version of the SVG.
        
        Args:
            zoom_factor: Zoom level
            center_x: X coordinate of zoom center (0-1 normalized)
            center_y: Y coordinate of zoom center (0-1 normalized)
            
        Returns:
            Zoomed SVG as string
        """
        # Create a copy of the SVG root so we don't mutate the original
        zoomed_svg = ET.fromstring(ET.tostring(self.svg_root))

        # Update viewBox for zoom
        new_viewbox = self.get_viewbox_for_zoom(zoom_factor, center_x, center_y)
        zoomed_svg.set('viewBox', new_viewbox)

        return ET.tostring(zoomed_svg, 'unicode')


class SVGWireframeConfig:
    """Configuration for SVG wireframe generation."""
    
    def __init__(self):
        # Each dictionary below controls styling for a distinct SVG layer.
        # Construction lines configuration
        self.construction_lines = {
            'enabled': True,
            'color': '#FF0000',
            'thickness': 2,
            'opacity': 0.8
        }
        
        # Face mesh configuration
        self.face_mesh = {
            'enabled': True,
            'color': '#00FF00',
            'thickness': 1,
            'opacity': 0.6
        }
        
        # Edge outline configuration  
        self.edge_outline = {
            'enabled': True,
            'color': '#0000FF',
            'thickness': 1,
            'opacity': 0.8
        }
        
        # DexiNed outline configuration
        self.dexined_outline = {
            'enabled': True,
            'color': '#800080',
            'thickness': 2,
            'opacity': 0.9
        }
        
        # Canvas configuration
        self.canvas = {
            'background_color': 'white',
            'width': 1920,
            'height': 1080
        }
        
        # Metadata stored in the SVG <metadata> block
        self.metadata = {
            'generator': 'Wireframe Portrait Processor',
            'version': '1.0',
            'created_at': '',
            'zoom_capable': True
        }
    
    @classmethod
    def beginner_preset(cls):
        """Preset for beginner users - construction lines only."""
        config = cls()
        config.face_mesh['enabled'] = False
        config.edge_outline['enabled'] = False
        config.dexined_outline['enabled'] = False
        config.construction_lines['thickness'] = 3  # Thicker lines
        return config
    
    @classmethod
    def intermediate_preset(cls):
        """Preset for intermediate users - construction lines + face mesh."""
        config = cls()
        config.edge_outline['enabled'] = False
        config.dexined_outline['enabled'] = False
        return config
    
    @classmethod
    def advanced_preset(cls):
        """Preset for advanced users - all features enabled."""
        config = cls()
        return config
    
    @classmethod
    def print_quality_preset(cls):
        """Preset for print quality output."""
        config = cls()
        config.canvas['width'] = 3439  # A4 at 300 DPI
        config.canvas['height'] = 2480
        config.construction_lines['thickness'] = 4
        config.face_mesh['thickness'] = 2
        config.dexined_outline['thickness'] = 3
        return config