#!/usr/bin/env python3
"""
Working wireframe demo server on port 8081
Serves the wireframe demo with proper CORS and individual layer control
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def main():
    # Change to the image_processing directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    PORT = 8081
    
    print(f"""
🎨 Working Wireframe Portrait Demo Server
========================================

Starting server on port {PORT}...

Demo URL: http://localhost:{PORT}/wireframe_demo_working.html

Features:
  ✓ Independent wireframe layer toggles
  ✓ Real-time transparency controls
  ✓ Separated SVG layer architecture
  ✓ 6 sample portraits from AIC collection

Available portraits:
""")
    
    # List available sample images
    fg_dir = Path("out_sample/clipped_images_fg")
    if fg_dir.exists():
        for img_file in sorted(fg_dir.glob("*_fg.png")):
            img_id = img_file.stem.replace("_fg", "")
            print(f"  - Portrait {img_id}")
    
    print(f"""
Layer files location: demo_layers/
  - {img_id}_construction_lines.svg (red guidelines)
  - {img_id}_face_mesh.svg (gray wireframes)  
  - {img_id}_pose_landmarks.svg (body skeleton)

Press Ctrl+C to stop the server
""")
    
    try:
        with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        sys.exit(0)
    except OSError as e:
        print(f"Error: {e}")
        print(f"Port {PORT} might already be in use. Try a different port.")
        sys.exit(1)

if __name__ == "__main__":
    main()