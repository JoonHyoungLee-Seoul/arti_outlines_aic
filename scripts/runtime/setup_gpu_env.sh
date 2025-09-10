#!/bin/bash

echo "=== Setting up complete GPU acceleration environment ==="

# Display settings
export DISPLAY=:99.0
export LIBGL_ALWAYS_INDIRECT=0
export MESA_GL_VERSION_OVERRIDE=4.5

# ROCm/AMD GPU settings
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_VISIBLE_DEVICES=0
export HSA_ENABLE_INTERRUPT=0
export HSA_ENABLE_SDMA=0

# OpenCL settings
export OPENCL_VENDOR_PATH=/opt/rocm/etc/OpenCL/vendors

# MediaPipe GPU settings
export MEDIAPIPE_DISABLE_GPU=0

# Additional GPU acceleration
export __GL_SYNC_TO_VBLANK=0
export __GL_YIELD="NOTHING"

echo "Environment variables set:"
echo "  DISPLAY=$DISPLAY"
echo "  ROCM_PATH=$ROCM_PATH"
echo "  HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"

# Check GPU status
echo "=== GPU Status ==="
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname --showtemp --showpower
else
    echo "ROCm SMI not available"
fi

# Check OpenCL
echo "=== OpenCL Platforms ==="
if command -v clinfo &> /dev/null; then
    clinfo --list
else
    echo "clinfo not available - install clinfo for OpenCL diagnostics"
fi

echo "=== GPU environment setup complete ==="
echo "Ready for MediaPipe with full GPU acceleration!"