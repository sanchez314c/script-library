#!/bin/bash

# This script automates the process of downloading, modifying, and compiling Ollama
# for a Debian 12 system with an AMD RX580 GPU (gfx803) and AMD 5800X CPU.
# It assumes ROCm 6.3 is already installed.

# Exit on any error
set -e

# Step 1: Install prerequisites
echo "Installing prerequisites..."
sudo apt update
sudo apt install -y git golang-go cmake

# Step 2: Verify ROCm installation and set up symlink
echo "Verifying ROCm installation..."
if ! rocminfo --full | grep -q "gfx803"; then
  echo "ROCm is not properly installed or the GPU is not detected. Please ensure ROCm 6.3 is configured correctly."
  exit 1
else
  echo "ROCm is installed and GPU (gfx803) is detected."
fi

echo "Creating symlink for ROCm 6.3..."
sudo ln -sf /opt/rocm-6.3.0 /opt/rocm

# Step 3: Set HSA_OVERRIDE_GFX_VERSION environment variable
echo "Setting HSA_OVERRIDE_GFX_VERSION for gfx803..."
export HSA_OVERRIDE_GFX_VERSION=8.3.0
echo 'export HSA_OVERRIDE_GFX_VERSION=8.3.0' >> ~/.bashrc

# Step 4: Clone the Ollama repository (specific version v0.5.12)
echo "Cloning Ollama v0.5.12 repository..."
git clone --branch v0.5.12 https://github.com/ollama/ollama.git
cd ollama

# Step 5: Modify source code for gfx803 support
echo "Modifying discover/gpu.go to support gfx803..."
sed -i 's/var RocmComputeMajorMin = "9"/var RocmComputeMajorMin = "8"/' discover/gpu.go

echo "Adding gfx803 to CMakePresets.json..."
sed -i 's/"gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-" /"gfx803;gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-" /g' CMakePresets.json

echo "Adding gfx803 to CMakeLists.txt..."
sed -i 's/"list(FILTER AMDGPU_TARGETS INCLUDE REGEX \"^gfx(900|94[012]|101[02]|1030|110[012])$\")"/"list(FILTER AMDGPU_TARGETS INCLUDE REGEX \"^gfx(803|900|94[012]|101[02]|1030|110[012])$\")"/g' CMakeLists.txt

# Step 6: Build Ollama backend for gfx803
echo "Building Ollama backend for gfx803..."
mkdir -p build
cmake -B build -DAMDGPU_TARGETS=gfx803
cmake --build build --config Release -j 8

# Step 7: Build Ollama frontend
echo "Building Ollama frontend..."
go generate ./...
go build .

# Step 8: Start Ollama backend in the background
echo "Starting Ollama backend..."
./ollama serve &

# Step 9: Prompt to test the frontend
echo "Ollama backend is running in the background."
echo "To test the frontend, run: './ollama run llama3.1:8b' in another terminal."
echo "Check the output for GPU usage (e.g., 'library=rocm' and 'compute=gfx803')."

echo "Ollama setup for gfx803 is complete!"