#!/bin/bash

# Ollama ROCm Source Build Script for RX 580 (HIP-Only)
# Version: 1.4.1 - Updated by Cortana for Jason
# Date: February 27, 2025

set -x
set -v
set -e

echo "Starting Ollama ROCm Source Build for RX 580..."

OLLAMA_DIR="/home/$USER/ollama-rocm"
BINDIR="/usr/local/bin"
ROCM_PATH="/opt/rocm-6.3.3"
GO_VERSION="1.22.0"
LOG_FILE="/home/$USER/Desktop/ollama-build.log"

check_root() {
    echo "Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    echo "Success: Running as root"
}

check_dependencies() {
    echo "Checking Go version..."
    GO_INSTALLED_VERSION=$(/usr/local/go/bin/go version | cut -d" " -f3 | sed 's/go//')
    if ! awk -v ver="$GO_INSTALLED_VERSION" 'BEGIN{if(ver<1.22) exit 1; exit 0}'; then
        echo "Error: Go version too old. Need >= 1.22, found ${GO_INSTALLED_VERSION}"
        exit 1
    fi
    echo "Success: Go version verified"
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
    snap remove go || true

    wget -v "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -O go.tar.gz || { echo "Error: Go download failed"; exit 1; }
    tar -C /usr/local -xzf go.tar.gz || { echo "Error: Go extraction failed"; exit 1; }
    rm -fv go.tar.gz
    echo "Installed Go version: $(/usr/local/go/bin/go version)"
    
    export PATH="/usr/local/go/bin:$PATH"
    export CC="/usr/bin/gcc-10"
    export CXX="/usr/bin/g++-10"

    # Check GCC version and adjust AVX flags
    GCC_VERSION=$(/usr/bin/gcc-10 --version | head -n1 | cut -d" " -f4 | cut -d. -f1)
    if [ "$GCC_VERSION" -lt 11 ]; then
        echo "Warning: GCC 10 detected—replacing -mavxvnni with -mavx512vnni for AVX-512 VNNI support"
        sed -i 's/-mavxvnni/-mavx512vnni/g' CMakeLists.txt
    else
        echo "GCC 11 or newer detected—keeping -mavxvnni or -mavx512vnni as is"
    fi

    # Export ROCM_PATH early and create symlink
    export ROCM_PATH
    ln -sf "$ROCM_PATH" /opt/rocm || echo "Symlink already exists"

    echo "Setting HIP and ROCm environment variables..."
    # Basic build flags
    export CGO_CFLAGS="-I$ROCM_PATH/include -I$ROCM_PATH/hip/include"
    export CGO_LDFLAGS="-L$ROCM_PATH/lib -L$ROCM_PATH/lib64 -L$ROCM_PATH/hip/lib -lamdhip64 -lhiprtc -lrocm_smi64 -lroctx64 -lrocfft -lhipblas"
    export GOFLAGS="-tags=hip"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib"
    
    # GFX803 specific settings
    export HSA_OVERRIDE_GFX_VERSION="8.0.3"
    export PYTORCH_ROCM_ARCH="gfx803"
    export ROCM_ARCH="gfx803"
    export ROC_ENABLE_PRE_VEGA="1"
    export HSA_ENABLE_SDMA="0"
    
    # ROCm/CUDA control
    export USE_ROCM="1"
    export USE_CUDA="0"
    
    # hipBLAS configuration
    export ROCBLAS_LIBRARY_PATH="$ROCM_PATH/lib/librocblas.so"
    export HIPBLAS_LIBRARY_PATH="$ROCM_PATH/lib/libhipblas.so"
    export USE_HIPBLAS="1"
    export TORCH_BLAS_PREFER_HIPBLASLT="0"
    
    # AMD specific optimizations
    export AMD_SERIALIZE_KERNEL="3"
    export OLLAMA_LLM_LIBRARY="rocm_v60002"

    echo "Environment variables set:"
    env | grep -E 'ROCM|HIP|HSA|AMD|TORCH|OLLAMA'
    echo "Success: Build environment configured for ROCm (HIP-only)"
}

patch_ollama() {
    echo "Applying GFX803 modifications..."
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
    
    # Add get_compiler shim
    echo "Adding get_compiler shim..."
    cat << 'EOF' >> llm/llama_linux.go

//export get_compiler
func get_compiler() *C.char {
    return C.CString("hipcc")  // Shim to bypass linker
}
EOF
    
    echo "Modifying GPU detection threshold..."
    sed -i 's/var RocmComputeMajorMin = "9"/var RocmComputeMajorMin = "8"/' discover/gpu.go || { echo "Error: Failed to modify GPU detection"; exit 1; }
    
    echo "Adding GFX803 to CMakePresets.json..."
    sed -i 's/"gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-"/"gfx803;gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-"/g' CMakePresets.json || { echo "Error: Failed to modify CMakePresets.json"; exit 1; }
    
    echo "Updating CMakeLists.txt for GFX803..."
    sed -i 's/list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(900|94[012]|101[02]|1030|110[012])$")/list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(803|900|94[012]|101[02]|1030|110[012])$")/g' CMakeLists.txt || { echo "Error: Failed to modify CMakeLists.txt"; exit 1; }
    
    echo "Success: Applied all GFX803 modifications"
}

build_ollama() {
    echo "Building Ollama with ROCm (HIP-only) support..."
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed"; exit 1; }
    
    echo "Running CMake build..."
    cmake -B build -DAMDGPU_TARGETS=gfx803 || { echo "Error: CMake configuration failed"; exit 1; }
    cmake --build build || { echo "Error: CMake build failed"; exit 1; }
    
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
Environment="ROCM_ARCH=gfx803"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="USE_ROCM=1"
Environment="USE_CUDA=0"
Environment="HSA_ENABLE_SDMA=0"
Environment="TORCH_BLAS_PREFER_HIPBLASLT=0"
Environment="USE_HIPBLAS=1"
Environment="AMD_SERIALIZE_KERNEL=3"
Environment="OLLAMA_LLM_LIBRARY=rocm_v60002"
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
    setup_build_env
    check_dependencies
    setup_repository
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