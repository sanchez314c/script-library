#!/bin/bash

# Ollama ROCm Source Build Script for RX 580 (HIP-Only) with i9-10900X Optimizations
# Version: 1.5.0 - Updated by Cortana for Jason
# Date: February 27, 2025
# Built against Ollama v0.5.12
# Target: AMD RX 580 (gfx803) + Intel i9-10900X (Cascade Lake-X)

set -x  # Echo all commands
set -e  # Exit on any error

echo "Starting Ollama ROCm Source Build for RX 580 (HIPBLAS) with i9-10900X optimizations..."

OLLAMA_DIR="/home/heathen-admin/ollama-rocm"
BINDIR="/home/heathen-admin/bin"
ROCM_PATH="/opt/rocm-6.3.3"
LOG_FILE="/home/heathen-admin/Desktop/ollama-build.log"

# Set all environment variables globally
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export PYTORCH_ROCM_ARCH=gfx803
export ROCM_ARCH=gfx803
export TORCH_BLAS_PREFER_HIPBLASLT=0
export ROC_ENABLE_PRE_VEGA=1
export USE_CUDA=0
export USE_ROCM=1
export USE_NINJA=1
export FORCE_CUDA=1
# Added CPU-specific flags
export CFLAGS="-march=cascadelake -O3"
export CXXFLAGS="-march=cascadelake -O3"

# Sanity check: ensure home dir exists
if [ ! -d "/home/heathen-admin" ]; then
    echo "Error: User home directory /home/heathen-admin does not exist"
    exit 1
fi
echo "DEBUG: Home dir /home/heathen-admin exists"

# Ensure log file exists and is writable
echo "DEBUG: Attempting to create log file at $LOG_FILE"
touch "$LOG_FILE" || { echo "Error: Cannot create log file at $LOG_FILE"; exit 1; }
chmod 664 "$LOG_FILE" || { echo "Error: Cannot set permissions on $LOG_FILE"; exit 1; }
echo "DEBUG: Logging initialized to $LOG_FILE"
# Redirect all output to LOG_FILE and terminal
exec > >(tee -a "$LOG_FILE") 2>&1

setup_environment() {
    echo "Setting up environment..."
    
    # Install GCC and build essentials first
    echo "Installing GCC and build dependencies..."
    apt update || { echo "Error: Apt update failed - run with sudo if needed"; exit 1; }
    apt install -y gcc g++ build-essential || { echo "Error: GCC installation failed - run with sudo if needed"; exit 1; }
    
    # Install other dependencies
    apt install -y git cmake golang rocm-dev rocm-libs rocm-hip-sdk librocblas-dev ninja-build doxygen || { echo "Error: Build deps install failed - run with sudo if needed"; exit 1; }
    
    [ -L "/opt/rocm" ] || echo "Warning: /opt/rocm symlink not present - may need sudo ln -sf /opt/rocm-6.3.3 /opt/rocm"
    
    echo "Environment variables set:"
    echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
    echo "PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"
    echo "ROCM_ARCH=$ROCM_ARCH"
    echo "TORCH_BLAS_PREFER_HIPBLASLT=$TORCH_BLAS_PREFER_HIPBLASLT"
    echo "ROC_ENABLE_PRE_VEGA=$ROC_ENABLE_PRE_VEGA"
    echo "USE_CUDA=$USE_CUDA"
    echo "USE_ROCM=$USE_ROCM"
    echo "USE_NINJA=$USE_NINJA"
    echo "FORCE_CUDA=$FORCE_CUDA"
    echo "CFLAGS=$CFLAGS"
    echo "CXXFLAGS=$CXXFLAGS"
    
    gcc --version
    g++ --version
    go version
    /opt/rocm-6.3.3/bin/amdclang++ --version
    echo "Success: Environment configured"
}

rebuild_rocblas() {
    echo "Rebuilding rocBLAS for gfx803..."
    cd /tmp
    rm -rf rocBLAS || { echo "Error: Failed to remove existing rocBLAS dir"; exit 1; }
    git clone --depth 1 https://github.com/ROCmSoftwarePlatform/rocBLAS.git || { echo "Error: rocBLAS clone failed"; exit 1; }
    cd rocBLAS
    sed -i 's/set(ROCM_DEFAULT_GFX_LIST "gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201")/set(ROCM_DEFAULT_GFX_LIST "gfx803;gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201")/' CMakeLists.txt || { echo "Error: rocBLAS CMake patch failed"; exit 1; }
    sed -i 's/list(APPEND supported_targets "gfx900" "gfx906:xnack-" "gfx908:xnack-" "gfx90a" "gfx942" "gfx1010" "gfx1012" "gfx1030" "gfx1100" "gfx1101" "gfx1102" "gfx1151" "gfx1200" "gfx1201")/list(APPEND supported_targets "gfx803" "gfx900" "gfx906:xnack-" "gfx908:xnack-" "gfx90a" "gfx942" "gfx1010" "gfx1012" "gfx1030" "gfx1100" "gfx1101" "gfx1102" "gfx1151" "gfx1200" "gfx1201")/' CMakeLists.txt || { echo "Error: rocBLAS supported_targets patch failed"; exit 1; }
    sed -i 's/set(BUILD_WITH_HIPBLASLT ON)/set(BUILD_WITH_HIPBLASLT OFF)/' CMakeLists.txt || { echo "Error: hipBLASLt disable patch failed"; exit 1; }
    mkdir build
    cd build
    export HSA_OVERRIDE_GFX_VERSION=8.0.3
    export ROCM_PATH=/opt/rocm-6.3.3
    export HIP_PLATFORM=amd
    export CPLUS_INCLUDE_PATH="/opt/rocm-6.3.3/include:$CPLUS_INCLUDE_PATH"
    export C_INCLUDE_PATH="/opt/rocm-6.3.3/include:$C_INCLUDE_PATH"
    export CPLUS_INCLUDE_PATH=$(echo $CPLUS_INCLUDE_PATH | sed "s|/usr/include||g")
    export C_INCLUDE_PATH=$(echo $C_INCLUDE_PATH | sed "s|/usr/include||g")
    cmake .. -DGPU_TARGETS=gfx803 -DCMAKE_INSTALL_PREFIX=/opt/rocm-6.3.3 \
             -DCMAKE_C_COMPILER=/opt/rocm-6.3.3/bin/amdclang \
             -DCMAKE_CXX_COMPILER=/opt/rocm-6.3.3/bin/amdclang++ \
             -G Ninja || { echo "Error: rocBLAS CMake failed"; exit 1; }
    ninja || { echo "Error: rocBLAS ninja build failed"; exit 1; }
    ninja install || { echo "Error: rocBLAS install failed - may need sudo"; exit 1; }
    cd /tmp && rm -rf rocBLAS
    echo "Success: rocBLAS rebuilt for gfx803"
    echo "DEBUG: Checking rocBLAS install - ls /opt/rocm-6.3.3/lib/librocblas*"
    ls -l /opt/rocm-6.3.3/lib/librocblas*
}

setup_ollama() {
    echo "Setting up Ollama v0.5.12..."
    if [ -d "$OLLAMA_DIR" ]; then
        echo "Removing existing Ollama directory for clean build..."
        rm -rf "$OLLAMA_DIR"
    fi
    
    git clone https://github.com/ollama/ollama.git "$OLLAMA_DIR" || { echo "Error: Git clone failed"; exit 1; }
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed"; exit 1; }
    git checkout v0.5.12 || { echo "Error: Checkout to v0.5.12 failed"; exit 1; }
    
    # Remove alderlake CPU target and force skylakex
    sed -i '/ggml-cpu-alderlake/d' CMakeLists.txt
    sed -i 's/set(GGML_CPU_TARGET "alderlake")/set(GGML_CPU_TARGET "skylakex")/' CMakeLists.txt
    
    # Patch for RX 580 support
    sed -i 's/var RocmComputeMajorMin = "9"/var RocmComputeMajorMin = "8"/' discover/gpu.go || { echo "Error: gpu.go patch failed"; exit 1; }
    
    # Update GPU targets
    sed -i 's/"gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-"/"gfx803;gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-"/g' CMakePresets.json
    
    # Update AMDGPU targets
    sed -i 's/"list(FILTER AMDGPU_TARGETS INCLUDE REGEX \"^gfx(900|94[012]|101[02]|1030|110[012])$\")"/"list(FILTER AMDGPU_TARGETS INCLUDE REGEX \"^gfx(803|900|94[012]|101[02]|1030|110[012])$\")"/g' CMakeLists.txt
    
    # Add Cascade Lake-X specific optimizations
    echo 'set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=cascadelake -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl")' >> CMakeLists.txt
    echo 'set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=cascadelake -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl")' >> CMakeLists.txt
    
    echo "Success: Ollama v0.5.12 checked out and patched"
}

build_ollama() {
    echo "Building Ollama with ROCm/HIPBLAS support..."
    cd "$OLLAMA_DIR"
    echo "DEBUG: Current dir - $(/bin/pwd)"
    
    # Backend
    echo "Building backend..."
    cmake -B build -S . \
        -DGPU_TARGETS=gfx803 \
        -DCMAKE_CXX_FLAGS="-march=cascadelake -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl" \
        -DCMAKE_C_FLAGS="-march=cascadelake -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl" \
        -DGGML_CPU_TARGET=skylakex \
        -G Ninja || { echo "Error: CMake failed"; exit 1; }
    
    ninja -C build || { echo "Error: Backend ninja build failed"; exit 1; }
    echo "DEBUG: Backend built - ls build/llama.cpp/build/bin"
    ls -l build/llama.cpp/build/bin
    
    # Frontend
    echo "Building frontend..."
    export CGO_ENABLED=1
    export CGO_CFLAGS="-I$ROCM_PATH/include"
    export CGO_LDFLAGS="-L$ROCM_PATH/lib -lhipblas -lrocblas"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib"
    go generate ./... || { echo "Error: Go generate failed"; exit 1; }
    go build -o ollama || { echo "Error: Go build failed"; exit 1; }
    install -m 755 ollama "$BINDIR/ollama" || { echo "Error: Binary install failed - ensure $BINDIR exists"; exit 1; }
    echo "Verifying binary installation..."
    ls -l "$BINDIR/ollama"
    file "$BINDIR/ollama"
    echo "DEBUG: Binary built - file $BINDIR/ollama"
    file "$BINDIR/ollama"
}

create_service() {
    echo "Creating systemd service..."
    mkdir -p /home/heathen-admin/.config/systemd/user
    tee /home/heathen-admin/.config/systemd/user/ollama-rocm.service > /dev/null << EOF || { echo "Error: User service file creation failed"; exit 1; }
[Unit]
Description=Ollama Service (ROCm HIPBLAS)
After=network-online.target

[Service]
ExecStart=$BINDIR/ollama serve
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=/home/heathen-admin/.ollama/models"
Environment="OLLAMA_HOST=127.0.0.1:11435"
Environment="ROCM_PATH=$ROCM_PATH"
Environment="LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib"
Environment="HSA_OVERRIDE_GFX_VERSION=8.0.3"
Environment="PYTORCH_ROCM_ARCH=gfx803"
Environment="ROCM_ARCH=gfx803"
Environment="TORCH_BLAS_PREFER_HIPBLASLT=0"
Environment="ROC_ENABLE_PRE_VEGA=1"
Environment="USE_CUDA=0"
Environment="USE_ROCM=1"
Environment="OLLAMA_DEBUG=true"

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload || { echo "Error: User daemon reload failed"; exit 1; }
    systemctl --user enable ollama-rocm || { echo "Error: User service enable failed"; exit 1; }
    systemctl --user restart ollama-rocm || { echo "Error: User service restart failed"; exit 1; }
    echo "Success: User service created and started"
}

verify_installation() {
    echo "Verifying installation..."
    sleep 2
    rocminfo | grep -A 5 "Name:.*gfx803" || { echo "Error: rocminfo failed or RX580 not detected"; exit 1; }
    echo "DEBUG: rocminfo passed"
    rocm-smi || { echo "Error: rocm-smi failed"; exit 1; }
    echo "DEBUG: rocm-smi passed"
    systemctl --user status ollama-rocm --no-pager || { echo "Error: User service status check failed"; exit 1; }
    echo "DEBUG: Service running"
    "$BINDIR/ollama" list || { echo "Error: Ollama test failed"; exit 1; }
    echo "DEBUG: Ollama binary works"
    echo "Checking HIP usage in logs..."
    journalctl --user -u ollama-rocm --since "2 minutes ago" -l | grep -i "hip" && echo "Success: HIP detected in logs" || { echo "Error: No HIP usage detected—running on CPU"; exit 1; }
    echo "Testing GPU activity..."
    "$BINDIR/ollama" run llama2:7b "Hello world" & sleep 5; rocm-smi | grep -q "%" || { echo "Error: No GPU activity detected"; exit 1; }
    echo "Success: Verification complete—HIPBLAS confirmed"
}

main() {
    echo "Entering main function..."
    echo "Logging to $LOG_FILE"
    setup_environment
    rebuild_rocblas
    setup_ollama
    build_ollama
    create_service
    verify_installation
    echo "
Ollama ROCm Build Complete!
- Built with HIPBLAS support for RX 580 (gfx803)
- Optimized for Intel i9-10900X (Cascade Lake-X)
- Binary: $BINDIR/ollama
- Service: ollama-rocm (user service, port 11435)

Commands:
- $BINDIR/ollama list : List models
- $BINDIR/ollama run <model> : Run model
- journalctl --user -u ollama-rocm : View logs

Notes:
- Models stored in /home/heathen-admin/.ollama/models
- Runs standalone—no CUDA or Conda dependency
- CPU optimized with AVX-512 instructions
- GPU optimized for RX 580 architecture
    "
}

trap 'echo "Error: Script failed at line $LINENO with exit code $?"' ERR

main