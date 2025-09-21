#!/usr/bin/env python3
"""
Simple HTTP server for the wireframe demo
Serves the demo and handles CORS for local file access
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
    
    PORT = 8080
    
    print(f"""
🎨 Wireframe Portrait Demo Server
================================

Starting server on port {PORT}...

Demo URL: http://localhost:{PORT}/wireframe_demo.html

Available sample images:
""")
    
    # List available sample images
    fg_dir = Path("out_sample/clipped_images_fg")
    if fg_dir.exists():
        for img_file in sorted(fg_dir.glob("*_fg.png")):
            img_id = img_file.stem.replace("_fg", "")
            print(f"  - Portrait {img_id}")
    
    print(f"""
Controls available in demo:
  ✓ Construction Lines toggle
  ✓ Face Mesh toggle  
  ✓ DexiNed Outlines toggle
  ✓ Pose Landmarks toggle
  ✓ Foreground transparency (0-100%)
  ✓ Background transparency (0-100%)

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