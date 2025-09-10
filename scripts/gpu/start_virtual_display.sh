#!/bin/bash

# Start virtual display for full GPU acceleration
echo "=== Starting Virtual Display with GPU Support ==="

# Kill existing Xvfb processes
echo "Cleaning up existing Xvfb processes..."
pkill -f Xvfb
sleep 1

# Start Xvfb with full GPU extensions
echo "Starting Xvfb with GPU acceleration support..."
Xvfb :99 \
    -screen 0 1920x1080x24 \
    -ac \
    +extension GLX \
    +extension RENDER \
    +extension RANDR \
    +extension COMPOSITE \
    -dpi 96 \
    -noreset \
    -nolisten tcp &

XVFB_PID=$!

# Wait for Xvfb to fully start
echo "Waiting for Xvfb to initialize..."
sleep 3

# Set environment variables
export DISPLAY=:99.0
export LIBGL_ALWAYS_INDIRECT=0
export MESA_GL_VERSION_OVERRIDE=4.5

echo "Virtual display started successfully!"
echo "DISPLAY: $DISPLAY"
echo "Xvfb PID: $XVFB_PID"

# Test OpenGL support
echo "Testing OpenGL capabilities..."
if command -v glxinfo &> /dev/null; then
    echo "--- OpenGL Information ---"
    glxinfo | grep -E "(OpenGL vendor|OpenGL renderer|OpenGL version|direct rendering)"
    echo "--- Available GLX Extensions ---"
    glxinfo | grep "GLX extensions:" -A 5
else
    echo "glxinfo not available - install mesa-utils"
fi

# Save configuration
echo $XVFB_PID > /tmp/xvfb.pid
echo ":99.0" > /tmp/xvfb_display

echo "=== Virtual Display Setup Complete ==="
echo "In your terminal, run:"
echo "  export DISPLAY=:99.0"
echo "  export LIBGL_ALWAYS_INDIRECT=0"