# Study of Wireframe Processing Modules

## `wireframe_portrait_processor.py`

### Configuration and Feature Toggles
```python
@dataclass
class WireframeConfig:
    # Feature toggles
    enable_construction_lines: bool = True
    enable_mesh: bool = False
    enable_dexined_outline: bool = False
    ...
    enable_svg_export: bool = False
    svg_output_path: str = ""
```
The `WireframeConfig` dataclass centralizes feature flags and visual settings for construction lines, meshes, DexiNed outlines, and output format options【F:image_processing/wireframe_portrait_processor.py†L48-L88】.

### Construction Lines
```python
class ConstructionLinesGenerator:
    @staticmethod
    def draw_construction_lines(image: np.ndarray, landmarks: List, config: WireframeConfig) -> np.ndarray:
        ...
        draw_line_between_points([10, 168, 4, 152], colors['vertical_center'])
        draw_line_between_points([63, 293], colors['eyebrow_line'])
        draw_line_between_points([33, 263], colors['eye_lines'])
        ...
```
`ConstructionLinesGenerator` maps MediaPipe landmarks to classical drawing guides such as the vertical center and eye lines, rendered in configurable colors and thicknesses【F:image_processing/wireframe_portrait_processor.py†L89-L152】.

### Mesh Overlay
```python
class MeshGenerator:
    def draw_face_mesh(self, image: np.ndarray, detection_result, config: WireframeConfig) -> np.ndarray:
        ...
        if colors['tesselation']:
            self.mp_drawing.draw_landmarks(..., connections=self.mp_face_mesh.FACEMESH_TESSELATION, ...)
        if colors['contours']:
            self.mp_drawing.draw_landmarks(..., connections=self.mp_face_mesh.FACEMESH_CONTOURS, ...)
        if colors['irises']:
            self.mp_drawing.draw_landmarks(..., connections=self.mp_face_mesh.FACEMESH_IRISES, ...)
```
`MeshGenerator` uses MediaPipe's built‑in connection sets to overlay tesselation, facial contours, and irises with individually styled colors and line weights【F:image_processing/wireframe_portrait_processor.py†L156-L229】.

### DexiNed Edge Detection
```python
class DexiNedGenerator:
    def generate_outline(self, image: np.ndarray, config: WireframeConfig) -> np.ndarray:
        ...
        img_tensor = self._preprocess_image(image)
        predictions = self.model(img_tensor)
        edge_map = predictions[-1].cpu().numpy()[0, 0]
        ...
    def _preprocess_image(self, image: np.ndarray):
        img_resized = cv2.resize(image, (352, 352))
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        ...
```
When the DexiNed model is available, its predictions produce an edge map that is thresholded and recolored; otherwise, the generator falls back to Canny edges on a white canvas【F:image_processing/wireframe_portrait_processor.py†L231-L349】.

### SVG Export and Presets
```python
def _generate_svg(self, image: np.ndarray, landmarks: List, detection_result) -> str:
    svg_generator = SVGGenerator(width, height, "white")
    ...
    if self.config.enable_construction_lines:
        svg_generator.add_construction_lines(landmark_array, construction_config)
    if self.config.enable_mesh and detection_result.face_landmarks:
        svg_generator.add_face_mesh(landmark_array, connections, mesh_config)
    if self.config.enable_dexined_outline and self.dexined_generator:
        svg_generator.add_dexined_outline(contours, dexined_config)
    ...
```
The processor converts landmarks and optional edges into vector form by delegating to `SVGGenerator`, attaching metadata about enabled features and resolution【F:image_processing/wireframe_portrait_processor.py†L680-L749】.

Preset configurations expose common combinations such as beginner or outline‑only modes for both the CLI and library usage【F:image_processing/wireframe_portrait_processor.py†L789-L833】.
The command‑line interface wires these presets and toggles into an executable tool for batch processing【F:image_processing/wireframe_portrait_processor.py†L835-L913】.

## `svg_generator.py`

### Building the Canvas
```python
def _create_svg_root(self) -> ET.Element:
    svg = ET.Element('svg')
    svg.set('xmlns', 'http://www.w3.org/2000/svg')
    svg.set('width', str(self.width))
    svg.set('height', str(self.height))
    svg.set('viewBox', f'0 0 {self.width} {self.height}')
    if self.background_color:
        background = ET.SubElement(svg, 'rect')
        background.set('width', '100%')
        background.set('height', '100%')
        background.set('fill', self.background_color)
    return svg
```
An `SVGGenerator` begins by creating a root `<svg>` element sized to the target canvas and optionally filled with a background rectangle【F:image_processing/svg_generator.py†L29-L44】.

### Construction Lines and Mesh
```python
def add_construction_lines(self, landmarks: np.ndarray, config: dict):
    ...
    add_line_through_landmarks([10, 168, 4, 152], 'center-line')
    add_line_through_landmarks([63, 293], 'eyebrow-line')
    ...
def add_face_mesh(self, landmarks: np.ndarray, connections: List[Tuple[int, int]], config: dict):
    ...
    self._add_line(group, start_point[0], start_point[1], end_point[0], end_point[1], color, thickness, f'mesh-{connection[0]}-{connection[1]}')
```
Landmark indices are translated into pixel coordinates to draw straight construction guides and mesh connections using SVG `<line>` elements【F:image_processing/svg_generator.py†L46-L106】【F:image_processing/svg_generator.py†L107-L132】.

### Freehand and DexiNed Outlines
```python
def add_edge_outline(self, edge_points: List[Tuple[int, int]], config: dict):
    path = ET.SubElement(group, 'path')
    path.set('d', path_data)
    ...
def add_dexined_outline(self, contours: List[np.ndarray], config: dict):
    for i, contour in enumerate(contours):
        path = ET.SubElement(group, 'path')
        ...
        if len(points) > 10:
            path_data += ' Z'
```
Freehand edge points or DexiNed contours are concatenated into SVG `<path>` data for smooth scalable outlines【F:image_processing/svg_generator.py†L134-L201】.

### Metadata and Config Presets
```python
def add_metadata(self, metadata: dict):
    desc = ET.SubElement(self.svg_root, 'desc')
    desc.text = f"Wireframe Portrait - Generated with settings: {metadata}"
    ...
def beginner_preset(cls):
    config = cls()
    config.face_mesh['enabled'] = False
    config.edge_outline['enabled'] = False
    config.dexined_outline['enabled'] = False
    config.construction_lines['thickness'] = 3
    return config
```
The generator can embed arbitrary metadata tags and provides preset style configurations for beginner, intermediate, advanced, and print‑quality exports【F:image_processing/svg_generator.py†L203-L240】【F:image_processing/svg_generator.py†L299-L383】.

## `high_resolution_wireframe_processor.py`

### HighResolutionConfig
```python
@dataclass
class HighResolutionConfig(WireframeConfig):
    target_resolution: Tuple[int, int] = (3840, 2160)
    max_resolution: Tuple[int, int] = (7680, 4320)
    enable_super_resolution: bool = True
    vector_construction_lines: bool = True
    line_thickness_scaling: float = 1.0
    mesh_density_scaling: float = 1.0
    ...
```
`HighResolutionConfig` extends the base configuration with target output sizes, optional super‑resolution, vector rendering, and scaling factors for lines and mesh density【F:image_processing/high_resolution_wireframe_processor.py†L37-L58】.

### Adaptive Line and Mesh Scaling
```python
class HighResolutionConstructionLinesGenerator(ConstructionLinesGenerator):
    @staticmethod
    def draw_construction_lines(image, landmarks, config):
        base_resolution = 1080
        resolution_factor = max(height, width) / base_resolution
        thickness = max(1, int(config.construction_line_thickness * resolution_factor * config.line_thickness_scaling))
        ...
class HighResolutionMeshGenerator(MeshGenerator):
    def draw_face_mesh(self, image, detection_result, config):
        resolution_factor = max(height, width) / base_resolution
        mesh_thickness = max(1, int(config.mesh_thickness * resolution_factor * config.mesh_density_scaling))
        ...
```
Both high‑resolution generators compute line weights relative to the image’s size to maintain consistent visual proportions at 4K and beyond【F:image_processing/high_resolution_wireframe_processor.py†L63-L124】【F:image_processing/high_resolution_wireframe_processor.py†L126-L199】.

### Super‑Resolution Edge Detection
```python
class HighResolutionDexiNedGenerator(DexiNedGenerator):
    def generate_outline(self, image, config):
        scales = [0.5, 1.0, 1.5] if config.enable_super_resolution else [1.0]
        for scale in scales:
            scaled_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
            img_tensor = self._preprocess_image(scaled_image)
            predictions = self.model(img_tensor)
            edge_map = predictions[-1].cpu().numpy()[0, 0]
            edge_image = self._postprocess_edges_hires(edge_map, scaled_image.shape, config, scale)
            ...
```
Multi‑scale inference blends edge maps at different resolutions and optionally enhances them to preserve sharp outlines in large prints【F:image_processing/high_resolution_wireframe_processor.py†L201-L267】.

### Processor Scaling Factors
```python
def _calculate_scaling_factors(self):
    target_width, target_height = self.config.target_resolution
    base_resolution = 1920 * 1080
    target_total = target_width * target_height
    resolution_factor = math.sqrt(target_total / base_resolution)
    self.config.line_thickness_scaling = min(3.0, max(0.5, resolution_factor))
    self.config.mesh_density_scaling = min(2.0, max(0.5, resolution_factor * 0.8))
```
Before processing, the high‑resolution processor derives multipliers that adjust line thickness and mesh density according to the requested canvas size【F:image_processing/high_resolution_wireframe_processor.py†L355-L365】.

### High‑Resolution Presets
```python
def create_high_resolution_presets() -> Dict[str, HighResolutionConfig]:
    resolutions = {
        'HD': (1920, 1080),
        '4K': (3840, 2160),
        '8K': (7680, 4320),
        'Print300DPI_A4': (3508, 2480),
        'Print300DPI_A3': (4961, 3508),
    }
    for res_name, resolution in resolutions.items():
        for preset_name, base_config in base_presets.items():
            hr_config = HighResolutionConfig(
                enable_construction_lines=base_config.enable_construction_lines,
                ...
                target_resolution=resolution,
                enable_super_resolution=True,
                vector_construction_lines=True,
                enable_edge_enhancement=True,
                tile_processing=resolution[0] * resolution[1] > 3840 * 2160,
                output_format="rgba",
                background_removal_method="lines_only",
            )
            hr_presets[config_name] = hr_config
```
Preset builder multiplies base configurations by multiple resolution options, automatically enabling super‑resolution and other high‑quality defaults for large canvases【F:image_processing/high_resolution_wireframe_processor.py†L504-L541】.

---
These code excerpts illustrate how the modules collaborate to convert photos into scalable wireframe portraits, from base processing through SVG export and high‑resolution refinement.
