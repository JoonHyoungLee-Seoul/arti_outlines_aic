#!/bin/bash

echo "=== Installing packages for complete GPU acceleration ==="

# Update package list
echo "Updating package list..."
sudo apt update

# Install Xvfb (X Virtual Framebuffer)
echo "Installing Xvfb and X11 utilities..."
sudo apt install -y xvfb x11-utils

# Install Mesa OpenGL libraries
echo "Installing Mesa OpenGL libraries..."
sudo apt install -y mesa-utils libegl1-mesa-dev libgl1-mesa-glx libgles2-mesa-dev

# Install additional GPU libraries
echo "Installing additional GPU libraries..."
sudo apt install -y libvulkan1 mesa-vulkan-drivers

# For ROCm support (AMD GPU)
echo "Installing ROCm OpenCL support..."
sudo apt install -y rocm-opencl rocm-opencl-dev

# Install additional dependencies
echo "Installing additional dependencies..."
sudo apt install -y libx11-dev libxext-dev libxrender-dev libxtst6

echo "=== Installation complete! ==="
echo "Please run: source setup_gpu_env.sh"