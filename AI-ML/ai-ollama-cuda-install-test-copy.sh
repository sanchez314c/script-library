#!/bin/bash

# Ollama ROCm Source Build Script for RX 580 (HIP-Only)
# Version: 1.4.41 - Updated by Cortana for Jason
# Date: February 26, 2025
# Built against Ollama v0.5.12

set -x  # Echo all commands
set -e  # Exit on any error

echo "Starting Ollama ROCm Source Build for RX 580 (HIPBLAS)..."

OLLAMA_DIR="/home/$USER/ollama-rocm"
BINDIR="/usr/local/bin"
ROCM_PATH="/opt/rocm-6.3.3"
LOG_FILE="/home/$USER/Desktop/ollama-build.log"

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

# Redirect all output to LOG_FILE and terminal
exec > >(tee -a "$LOG_FILE") 2>&1

check_root() {
    echo "Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    echo "Success: Running as root (UID: $(id -u), User: $(whoami))"
}

setup_environment() {
    echo "Setting up environment..."
    apt update || { echo "Error: Apt update failed"; exit 1; }
    apt install -y git cmake build-essential golang rocm-dev rocm-libs rocm-hip-sdk librocblas-dev ninja-build doxygen || { echo "Error: Build deps install failed"; exit 1; }
    ln -sf /opt/rocm-6.3.3 /opt/rocm || { echo "Error: Symlink creation failed"; exit 1; }
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
    # Patch CMakeLists.txt for gfx803
    sed -i 's/set(ROCM_DEFAULT_GFX_LIST "gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201")/set(ROCM_DEFAULT_GFX_LIST "gfx803;gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201")/' CMakeLists.txt || { echo "Error: rocBLAS CMake patch failed"; exit 1; }
    sed -i 's/list(APPEND supported_targets "gfx900" "gfx906:xnack-" "gfx908:xnack-" "gfx90a" "gfx942" "gfx1010" "gfx1012" "gfx1030" "gfx1100" "gfx1101" "gfx1102" "gfx1151" "gfx1200" "gfx1201")/list(APPEND supported_targets "gfx803" "gfx900" "gfx906:xnack-" "gfx908:xnack-" "gfx90a" "gfx942" "gfx1010" "gfx1012" "gfx1030" "gfx1100" "gfx1101" "gfx1102" "gfx1151" "gfx1200" "gfx1201")/' CMakeLists.txt || { echo "Error: rocBLAS supported_targets patch failed"; exit 1; }
    # Disable hipBLASLt to avoid header clashes
    sed -i 's/set(BUILD_WITH_HIPBLASLT ON)/set(BUILD_WITH_HIPBLASLT OFF)/' CMakeLists.txt || { echo "Error: hipBLASLt disable patch failed"; exit 1; }
    mkdir build
    cd build
    export HSA_OVERRIDE_GFX_VERSION=8.0.3
    export ROCM_PATH=/opt/rocm-6.3.3
    export HIP_PLATFORM=amd
    export CPLUS_INCLUDE_PATH="/opt/rocm-6.3.3/include:$CPLUS_INCLUDE_PATH"
    export C_INCLUDE_PATH="/opt/rocm-6.3.3/include:$C_INCLUDE_PATH"
    # Exclude system HIP headers
    export CPLUS_INCLUDE_PATH=$(echo $CPLUS_INCLUDE_PATH | sed "s|/usr/include||g")
    export C_INCLUDE_PATH=$(echo $C_INCLUDE_PATH | sed "s|/usr/include||g")
    cmake .. -DGPU_TARGETS=gfx803 -DCMAKE_INSTALL_PREFIX=/opt/rocm-6.3.3 \
             -DCMAKE_C_COMPILER=/opt/rocm-6.3.3/bin/amdclang \
             -DCMAKE_CXX_COMPILER=/opt/rocm-6.3.3/bin/amdclang++ \
             -G Ninja || { echo "Error: rocBLAS CMake failed"; exit 1; }
    ninja || { echo "Error: rocBLAS ninja build failed"; exit 1; }
    sudo ninja install || { echo "Error: rocBLAS install failed"; exit 1; }
    cd /tmp && rm -rf rocBLAS
    echo "Success: rocBLAS rebuilt for gfx803"
}

setup_ollama() {
    echo "Setting up Ollama v0.5.12..."
    rm -rf "$OLLAMA_DIR" || echo "No old directory to remove"
    git clone https://github.com/ollama/ollama.git "$OLLAMA_DIR" || { echo "Error: Git clone failed"; exit 1; }
    cd "$OLLAMA_DIR"
    git checkout v0.5.12 || { echo "Error: Checkout to v0.5.12 failed"; exit 1; }
    sed -i 's/var RocmComputeMajorMin = "9"/var RocmComputeMajorMin = "8"/' discover/gpu.go || { echo "Error: gpu.go patch failed"; exit 1; }
    sed -i 's/"gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-" /"gfx803;gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-" /g' CMakePresets.json
    sed -i 's/"list(FILTER AMDGPU_TARGETS INCLUDE REGEX \"^gfx(900|94[012]|101[02]|1030|110[012])$\")"/"list(FILTER AMDGPU_TARGETS INCLUDE REGEX \"^gfx(803|900|94[012]|101[02]|1030|110[012])$\")"/g' CMakeLists.txt
    echo "Success: Ollama v0.5.12 checked out and patched"
}

build_ollama() {
    echo "Building Ollama with ROCm/HIPBLAS support..."
    cd "$OLLAMA_DIR"

    # Backend
    echo "Building backend..."
    cmake -B build -DGPU_TARGETS=gfx803 -G Ninja || { echo "Error: CMake failed"; exit 1; }
    ninja -C build || { echo "Error: Backend ninja build failed"; exit 1; }

    # Frontend
    echo "Building frontend..."
    export CGO_ENABLED=1
    export CGO_CFLAGS="-I$ROCM_PATH/include"
    export CGO_LDFLAGS="-L$ROCM_PATH/lib -lhipblas -lrocblas"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib"
    go generate ./... || { echo "Error: Go generate failed"; exit 1; }
    go build -o ollama || { echo "Error: Go build failed"; exit 1; }
    sudo install -m 755 ollama "$BINDIR/ollama" || { echo "Error: Binary install failed"; exit 1; }
    echo "Verifying binary installation..."
    ls -l "$BINDIR/ollama"
    file "$BINDIR/ollama"
    echo "Success: Ollama built and installed"
}

create_service() {
    echo "Creating systemd service..."
    tee /etc/systemd/system/ollama-rocm.service > /dev/null << EOF || { echo "Error: Service file creation failed"; exit 1; }
[Unit]
Description=Ollama Service (ROCm HIPBLAS)
After=network-online.target

[Service]
ExecStart=$BINDIR/ollama serve
User=heathen-admin
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=/usr/share/ollama/.ollama/models"
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
    systemctl daemon-reload || { echo "Error: Daemon reload failed"; exit 1; }
    systemctl enable ollama-rocm || { echo "Error: Service enable failed"; exit 1; }
    systemctl restart ollama-rocm || { echo "Error: Service restart failed"; exit 1; }
    echo "Success: Service created and started"
}

verify_installation() {
    echo "Verifying installation..."
    sleep 2
    rocminfo | grep -A 5 "Name:.*gfx803" || { echo "Error: rocminfo failed or RX580 not detected"; exit 1; }
    rocm-smi || { echo "Error: rocm-smi failed"; exit 1; }
    systemctl status ollama-rocm --no-pager || { echo "Error: Service status check failed"; exit 1; }
    ollama list || { echo "Error: Ollama test failed"; exit 1; }
    echo "Checking HIP usage in logs..."
    journalctl -u ollama-rocm --since "2 minutes ago" -l | grep -i "hip" && echo "Success: HIP detected in logs" || { echo "Error: No HIP usage detected—running on CPU"; exit 1; }
    echo "Testing GPU activity..."
    ollama run llama3.1:8b "Hello world" & sleep 5; rocm-smi | grep -q "%" || { echo "Error: No GPU activity detected"; exit 1; }
    echo "Success: Verification complete—HIPBLAS confirmed"
}

main() {
    echo "Entering main function..."
    echo "Logging to $LOG_FILE"
    check_root
    setup_environment
    rebuild_rocblas
    setup_ollama
    build_ollama
    create_service
    verify_installation
    echo "
Ollama ROCm Build Complete!
- Built with HIPBLAS support for RX 580 (gfx803)
- Binary: $BINDIR/ollama
- Service: ollama-rocm (port 11435)
Commands:
- ollama list : List models
- ollama run <model> : Run model
- journalctl -u ollama-rocm : View logs
Notes:
- Models stored in /usr/share/ollama/.ollama/models
- Runs standalone—no CUDA or Conda dependency
    "
}

trap 'echo "Error: Script failed at line $LINENO with exit code $?"' ERR

main
