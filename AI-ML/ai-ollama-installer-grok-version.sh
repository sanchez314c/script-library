#!/bin/bash

# Exit on any error
set -e

# Install necessary dependencies
echo "Updating package lists and installing dependencies..."
sudo apt update
sudo apt install -y git cmake build-essential golang libclang-dev

# Clean up any previous Ollama clone attempts
if [ -d "ollama" ]; then
    echo "Removing existing ollama directory..."
    rm -rf ollama
fi

# Clone Ollama repository
echo "Cloning Ollama repository..."
git clone https://github.com/ollama/ollama.git
cd ollama || { echo "Failed to cd to ollama"; exit 1; }

# Checkout the desired Ollama version
echo "Checking out Ollama v0.1.28..."
git checkout v0.1.28 || { echo "Failed to checkout v0.1.28"; exit 1; }

# Modify discover/gpu.go to enable gfx803 support
echo "Modifying discover/gpu.go for gfx803 support..."
sed -i 's/var RocmComputeMajorMin = "9"/var RocmComputeMajorMin = "8"/' llm/discover/gpu.go || { echo "gpu.go not found or sed failed"; exit 1; }

# Modify CMakePresets.json and CMakeLists.txt to add gfx803 to AMDGPUTARGETS
echo "Updating CMake files for gfx803..."
sed -i 's/"gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-"/"gfx803;gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-"/g' llm/CMakePresets.json || { echo "CMakePresets.json not found or sed failed"; exit 1; }
sed -i 's/"list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(900|94[012]|101[02]|1030|110[012])$"/"list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(803|900|94[012]|101[02]|1030|110[012])$"/g' llm/CMakeLists.txt || { echo "CMakeLists.txt not found or sed failed"; exit 1; }

# Check if ROCm is installed, install if missing
if ! command -v rocminfo &> /dev/null; then
    echo "Installing ROCm 6.3.2..."
    wget -q -O - https://repo.radeon.com/rocm/apt/6.3.2/rocm.gpg.key | sudo apt-key add -
    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.3.2/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
    sudo apt update
    sudo apt install -y rocm-libs rocm-dev rocm-cmake
    echo "Sourcing ROCm environment..."
    echo "source /opt/rocm/bin/rocm-setup-env.sh" >> ~/.bashrc
    source /opt/rocm/bin/rocm-setup-env.sh
else
    echo "ROCm already detected, skipping installation."
fi

# Compile rocBLAS for gfx803
echo "Cloning and building rocBLAS for gfx803..."
cd "$HOME/Desktop/Claude_Working" || { echo "Failed to cd back to Claude_Working"; exit 1; }
if [ -d "rocBLAS" ]; then
    rm -rf rocBLAS
fi
git clone --recursive https://github.com/ROCmSoftwarePlatform/rocBLAS.git
cd rocBLAS || { echo "Failed to cd to rocBLAS"; exit 1; }
# Use a tag compatible with ROCm 6.3.2
git checkout rocm-6.3.2 || { echo "Failed to checkout rocm-6.3.2"; exit 1; }
mkdir build && cd build || { echo "Failed to create or cd to build"; exit 1; }
cmake .. -DAMDGPU_TARGETS=gfx803 || { echo "rocBLAS cmake failed"; exit 1; }
make -j$(nproc) || { echo "rocBLAS make failed"; exit 1; }
sudo make install || { echo "rocBLAS install failed"; exit 1; }
cd "$HOME/Desktop/Claude_Working/ollama" || { echo "Failed to cd back to ollama"; exit 1; }

# Compile Ollama backend for gfx803
echo "Building Ollama backend for gfx803..."
mkdir build && cd build || { echo "Failed to create or cd to build"; exit 1; }
cmake .. -DAMDGPU_TARGETS=gfx803 || { echo "Ollama cmake failed"; exit 1; }
make -j$(nproc) || { echo "Ollama make failed"; exit 1; }
sudo make install || { echo "Ollama install failed"; exit 1; }
cd .. || { echo "Failed to cd back"; exit 1; }

# Compile Ollama frontend
echo "Building Ollama frontend..."
go generate ./... || { echo "go generate failed"; exit 1; }
go build . || { echo "go build failed"; exit 1; }

# Run Ollama server in the background
echo "Starting Ollama server..."
./ollama serve &

# Wait briefly to ensure server starts
sleep 5

# Test Ollama with a model
echo "Testing Ollama with llama3.1:8b..."
./ollama run llama3.1:8b || { echo "Failed to run model; ensure server is running and model is downloaded"; exit 1; }

echo "Setup complete! Ollama should be running with gfx803 support."

# Notes:
# - Ensure /dev/kfd and /dev/dri permissions are set (e.g., sudo chmod 666 /dev/kfd /dev/dri/*)
# - If model download fails, run './ollama pull llama3.1:8b' separately
# - Monitor RAM/VRAM usage during compilation and runtime