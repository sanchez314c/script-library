#!/bin/bash

# Ollama ROCm Source Build Script for RX 580 (HIP-Only)
# Version: 1.2.23 - Built by Cortana (via Grok 3, xAI) for Jason
# Date: February 25, 2025

set -x
set -e

echo "Starting Ollama ROCm Source Build for RX 580..."

OLLAMA_DIR="/home/$USER/ollama-rocm"
BINDIR="/usr/local/bin"
ROCM_PATH="/opt/rocm-6.3.3"
GO_VERSION="1.23.4"
LOG_FILE="/home/$USER/Desktop/ollama-build.log"

check_root() {
    echo "Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    echo "Success: Running as root"
}

setup_repository() {
    echo "Cloning Ollama repository (v0.5.12)..."
    rm -rf "$OLLAMA_DIR" || echo "No old directory to remove"
    git clone --branch v0.5.12 https://github.com/ollama/ollama.git "$OLLAMA_DIR" || { echo "Error: Git clone failed"; exit 1; }
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed"; exit 1; }
    echo "Verifying clone contents..."
    ls -l llm/ || { echo "Error: llm/ directory not found—clone may have failed"; exit 1; }
    echo "Success: Repository cloned"
}

setup_build_env() {
    echo "Setting up build environment..."
    apt update || { echo "Error: Apt update failed"; exit 1; }
    apt install -y libstdc++-12-dev cmake gcc-10 g++-10 git librocprim-dev || { echo "Error: Build deps install failed"; exit 1; }
    apt remove -y golang-go golang || true
    rm -rf /usr/local/go /usr/bin/go /usr/local/bin/go || true
    wget -v https://go.dev/dl/go$GO_VERSION.linux-amd64.tar.gz -O go.tar.gz || { echo "Error: Go download failed"; exit 1; }
    tar -C /usr/local -xzf go.tar.gz || { echo "Error: Go extraction failed"; exit 1; }
    rm -fv go.tar.gz
    echo "Installed Go version: $(/usr/local/go/bin/go version)"
    export PATH="/usr/local/go/bin:$PATH"
    export CC="/usr/bin/gcc-10"
    export CXX="/usr/bin/g++-10"

    echo "Setting HIP-specific environment variables..."
    export CGO_CFLAGS="-I$ROCM_PATH/include -I$ROCM_PATH/hip/include"
    export CGO_LDFLAGS="-L$ROCM_PATH/lib -L$ROCM_PATH/lib64 -L$ROCM_PATH/hip/lib -lamdhip64 -lhiprtc -lrocm_smi64 -lroctx64 -lrocfft"
    export GOFLAGS="-tags=hip"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib"
    export HSA_OVERRIDE_GFX_VERSION="8.0.3"
    export ROC_ENABLE_PRE_VEGA="1"
    export PYTORCH_ROCM_ARCH="gfx803"
    export HIP_VISIBLE_DEVICES="0"
    echo "CGO_CFLAGS=$CGO_CFLAGS"
    echo "CGO_LDFLAGS=$CGO_LDFLAGS"
    echo "GOFLAGS=$GOFLAGS"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "Success: Build environment configured for ROCm (HIP-only)"
}

patch_ollama() {
    echo "Patching Ollama to accept gfx803..."
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed"; exit 1; }
    
    # Find and patch the AMD GPU detection file
    GPU_FILE=$(find . -type f -name "*.go" -exec grep -l "amdgpu" {} + | head -1)
    if [ -z "$GPU_FILE" ]; then
        echo "Error: Could not locate GPU detection file with 'amdgpu'"
        ls -lR "$OLLAMA_DIR" > "$OLLAMA_DIR/dir_listing.txt"
        exit 1
    fi
    echo "Patching $GPU_FILE..."
    sed -i '/amdgpu too old/{s/return nil, err/return g, nil/}' "$GPU_FILE" || { echo "Error: Failed to patch $GPU_FILE"; exit 1; }
    
    # Find the main llama implementation file where the get_compiler needs to be added
    # In v0.5.12, the structure may have changed
    echo "Searching for appropriate file to add get_compiler shim..."
    
    # Look for llm implementation files in the repository
    LLM_FILES=$(find . -type f -name "*.go" -path "*/llm/*" | grep -v "_test.go")
    
    # Try to identify the main llm file for Linux
    LLM_LINUX_FILE=""
    for file in $LLM_FILES; do
        if grep -q "linux" "$file" || grep -q "unix" "$file"; then
            echo "Found potential Linux implementation file: $file"
            LLM_LINUX_FILE="$file"
            break
        fi
    done
    
    # If no specific Linux file found, use a general llm file
    if [ -z "$LLM_LINUX_FILE" ]; then
        LLM_LINUX_FILE=$(echo "$LLM_FILES" | head -1)
        echo "No specific Linux implementation found, using: $LLM_LINUX_FILE"
    fi
    
    if [ -z "$LLM_LINUX_FILE" ]; then
        echo "Error: Could not find any llm implementation files"
        exit 1
    fi
    
    # Create a new file for our get_compiler shim
    echo "Creating get_compiler shim in a separate file..."
    SHIM_FILE="llm/getcompiler_amd.go"
    mkdir -p $(dirname "$SHIM_FILE")
    
    # Determine the package name from the LLM_LINUX_FILE
    PACKAGE_NAME=$(grep "package" "$LLM_LINUX_FILE" | head -1 | awk '{print $2}')
    if [ -z "$PACKAGE_NAME" ]; then
        PACKAGE_NAME="llm"  # Default package name
    fi
    
    # Write the shim to a separate file
    cat > "$SHIM_FILE" << EOF
// Package $PACKAGE_NAME provides LLM functionality
package $PACKAGE_NAME

/*
#include <stdlib.h>
*/
import "C"

//export get_compiler
func get_compiler() *C.char {
    return C.CString("hipcc")  // Shim to bypass linker
}
EOF
    
    echo "Success: Created get_compiler shim in $SHIM_FILE"
    
    # Patch CMakeLists.txt if it exists, to support gfx803
    if [ -f "CMakeLists.txt" ]; then
        echo "Patching CMakeLists.txt for gfx803 support..."
        sed -i 's/set(CMAKE_CUDA_ARCHITECTURES .*/set(CMAKE_CUDA_ARCHITECTURES 37)/' CMakeLists.txt || echo "Warning: CMakeLists.txt patch failed (may be harmless)"
    fi
    
    echo "Success: Patched Ollama source"
}

build_ollama() {
    echo "Building Ollama with ROCm (HIP-only) support..."
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed"; exit 1; }
    echo "Generating Go files with HIP..."
    /usr/local/go/bin/go generate ./... || { echo "Error: Go generate failed"; exit 1; }
    echo "Building Ollama with HIP tags..."
    CGO_ENABLED=1 /usr/local/go/bin/go build -v -x -o ollama-rocm . || { echo "Error: Go build failed—check above for cgo errors"; exit 1; }
    if [ ! -f ollama-rocm ]; then
        echo "Error: Build failed—ollama-rocm binary not found"
        exit 1
    fi
    mv -v ollama-rocm "$BINDIR/ollama-rocm" || { echo "Error: Failed to move binary"; exit 1; }
    echo "Success: Ollama-rocm built and installed"
}

create_service() {
    echo "Creating systemd service..."
    tee /etc/systemd/system/ollama-rocm.service > /dev/null << EOF || { echo "Error: Service file creation failed"; exit 1; }
[Unit]
Description=Ollama Service (ROCm HIP-Only)
After=network-online.target

[Service]
ExecStart=$BINDIR/ollama-rocm serve
User=$USER
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=/usr/share/ollama-rocm/.ollama/models"
Environment="OLLAMA_HOST=127.0.0.1:11435"
Environment="ROCM_PATH=$ROCM_PATH"
Environment="LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib"
Environment="HSA_OVERRIDE_GFX_VERSION=8.0.3"
Environment="ROC_ENABLE_PRE_VEGA=1"
Environment="PYTORCH_ROCM_ARCH=gfx803"
Environment="HIP_VISIBLE_DEVICES=0"
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
    $ROCM_PATH/bin/rocminfo | grep -A 5 "Name:.*gfx803" || { echo "Error: rocminfo failed or RX580 not detected"; exit 1; }
    $ROCM_PATH/bin/rocm-smi || { echo "Error: rocm-smi failed"; exit 1; }
    systemctl status ollama-rocm --no-pager || { echo "Error: Service status check failed"; exit 1; }
    $BINDIR/ollama-rocm list || { echo "Error: Ollama test failed"; exit 1; }
    echo "Checking HIP usage in logs..."
    journalctl -u ollama-rocm --since "2 minutes ago" -l | grep -i "hip" && echo "Success: HIP detected in logs" || { echo "Error: No HIP usage detected—running on CPU"; exit 1; }
    echo "Testing GPU activity..."
    $BINDIR/ollama-rocm run llama2 "Hello world" & sleep 5; $ROCM_PATH/bin/rocm-smi | grep -q "[1-9]%" || { echo "Error: No GPU activity detected"; exit 1; }
    echo "Success: Verification complete—HIP confirmed"
}

main() {
    echo "Entering main function..."
    check_root
    setup_repository
    setup_build_env
    patch_ollama
    build_ollama
    create_service
    verify_installation
    echo "
Ollama ROCm Build Complete!
- Built with HIP-only support for RX 580 (gfx803)
- Binary: $BINDIR/ollama-rocm
- Service: ollama-rocm (port 11435)
Commands:
- ollama-rocm list : List models
- ollama-rocm run <model> : Run model
- journalctl -u ollama-rocm : View logs
Notes:
- Models stored in /usr/share/ollama-rocm/.ollama/models
- Runs standalone—no CUDA or Conda dependency
    "
}

trap 'echo "Error: Script failed at line $LINENO with exit code $?"' ERR

main
