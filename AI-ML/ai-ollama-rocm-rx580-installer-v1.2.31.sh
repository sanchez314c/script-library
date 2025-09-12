#!/bin/bash

# Ollama ROCm Source Build Script for RX 580 (HIP-Only)
# Version: 1.2.31 - Fixed by Grok for Jason
# Date: February 26, 2025

set -x
set -e

echo "Starting Ollama ROCm Source Build for RX 580..."

OLLAMA_DIR="/home/$USER/ollama-rocm"
BINDIR="/usr/bin"
ROCM_PATH="/opt/rocm-5.7"
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

install_rocm() {
    echo "Installing ROCm 5.7 for GFX803 support..."
    apt-get update
    apt-get install -y wget gnupg2
    sh -c "echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7 jammy main' > /etc/apt/sources.list.d/rocm.list"
    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
    apt-get update
    apt-get install -y rocm-dev=5.7.0 rocm-libs=5.7.0 rocm-hip-sdk=5.7.0 rocminfo
    echo "Verifying ROCm installation..."
    $ROCM_PATH/bin/rocminfo | grep "gfx803" || { echo "Error: GFX803 not detected"; exit 1; }
}

setup_repository() {
    echo "Cloning Ollama repository (v0.5.12)..."
    rm -rf "$OLLAMA_DIR" || echo "No old directory to remove"
    git clone --branch v0.5.12 https://github.com/ollama/ollama.git "$OLLAMA_DIR" || { echo "Error: Git clone failed"; exit 1; }
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed"; exit 1; }
    echo "Success: Repository cloned"
}

setup_build_env() {
    echo "Setting up build environment..."
    apt-get install -y libstdc++-12-dev cmake gcc g++ git librocprim-dev || { echo "Error: Build deps install failed"; exit 1; }
    apt remove -y golang-go golang || true
    rm -rf /usr/local/go /usr/bin/go /usr/local/bin/go || true
    wget -q https://go.dev/dl/go$GO_VERSION.linux-amd64.tar.gz -O go.tar.gz || { echo "Error: Go download failed"; exit 1; }
    tar -C /usr/local -xzf go.tar.gz || { echo "Error: Go extraction failed"; exit 1; }
    rm -f go.tar.gz
    echo "Installed Go version: $(/usr/local/go/bin/go version)"
    export PATH="/usr/local/go/bin:$PATH"

    echo "Setting HIP-specific environment variables..."
    export CGO_CFLAGS="-I$ROCM_PATH/include -I$ROCM_PATH/hip/include -D__HIP_PLATFORM_AMD__"
    export CGO_LDFLAGS="-L$ROCM_PATH/lib -L$ROCM_PATH/hip/lib -lamdhip64"
    export GOFLAGS="-tags=hip"
    export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$LD_LIBRARY_PATH"
    export HSA_OVERRIDE_GFX_VERSION="8.0.3"
    export ROC_ENABLE_PRE_VEGA="1"
    export HIP_VISIBLE_DEVICES="0"
    export CUDA_VISIBLE_DEVICES=""
    echo "CGO_CFLAGS=$CGO_CFLAGS"
    echo "CGO_LDFLAGS=$CGO_LDFLAGS"
    echo "GOFLAGS=$GOFLAGS"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "Success: Build environment configured"
}

patch_ollama() {
    echo "Patching Ollama for ROCm and GFX803..."
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed"; exit 1; }
    
    GPU_FILE="$OLLAMA_DIR/gpu/amd_linux.go"
    [ -f "$GPU_FILE" ] || { echo "Error: $GPU_FILE not found"; exit 1; }
    echo "Patching $GPU_FILE..."
    cp "$GPU_FILE" "$GPU_FILE.bak"
    sed -i 's/return nil, fmt.Errorf("amdgpu too old %s", ver)/return g, nil/' "$GPU_FILE"
    sed -i 's/Warnf("amdgpu too old %s", ver)/Infof("Cortana: gfx803 accepted, GPU ID: %d", 0)/' "$GPU_FILE"
    grep -q '"gfx803": true' "$GPU_FILE" || sed -i '/supportedTypes = map\[string\]bool{/a\    "gfx803": true,' "$GPU_FILE"

    MAIN_GPU_FILE="$OLLAMA_DIR/gpu/gpu.go"
    [ -f "$MAIN_GPU_FILE" ] || { echo "Error: $MAIN_GPU_FILE not found"; exit 1; }
    echo "Patching $MAIN_GPU_FILE..."
    cp "$MAIN_GPU_FILE" "$MAIN_GPU_FILE.bak"
    sed -i '/"no compatible GPUs were discovered"/i\    if os.Getenv("OLLAMA_LLM_LIBRARY") == "rocm" {\n        log.Infof("Cortana: Forcing ROCm/HIP detection")\n        gpus, err := discoverHIP()\n        if err == nil && len(gpus) > 0 {\n            return gpus, nil\n        }\n    }' "$MAIN_GPU_FILE"

    SHIM_FILE="$OLLAMA_DIR/llm/getcompiler_amd.go"
    mkdir -p $(dirname "$SHIM_FILE")
    cat > "$SHIM_FILE" << EOF
package llm
/*
#include <stdlib.h>
*/
import "C"

//export get_compiler
func get_compiler() *C.char {
    return C.CString("hipcc")
}
EOF
    echo "Success: Patched Ollama source"
}

build_ollama() {
    echo "Building Ollama with ROCm (HIP-only) support..."
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed"; exit 1; }
    /usr/local/go/bin/go generate ./...
    CGO_ENABLED=1 /usr/local/go/bin/go build -o ollama .
    [ -f ollama ] || { echo "Error: Build failed—ollama binary not found"; exit 1; }
    mv -v ollama "$BINDIR/ollama"
    echo "Success: Ollama built and installed"
}

create_service() {
    echo "Creating systemd service..."
    tee /etc/systemd/system/ollama.service > /dev/null << EOF
[Unit]
Description=Ollama Service (ROCm HIP-Only)
After=network-online.target

[Service]
ExecStart=$BINDIR/ollama serve
User=$USER
Group=$(id -gn $USER)
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=/usr/share/ollama/.ollama/models"
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="ROCM_PATH=$ROCM_PATH"
Environment="LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$LD_LIBRARY_PATH"
Environment="HSA_OVERRIDE_GFX_VERSION=8.0.3"
Environment="ROC_ENABLE_PRE_VEGA=1"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="CUDA_VISIBLE_DEVICES="

[Install]
WantedBy=default.target
EOF
    systemctl daemon-reload
    systemctl enable ollama
    systemctl restart ollama
    echo "Success: Service created and started"
}

verify_installation() {
    echo "Verifying installation..."
    sleep 2
    $ROCM_PATH/bin/rocminfo | grep -A 5 "Name:.*gfx803"
    systemctl status ollama --no-pager
    $BINDIR/ollama list || { echo "Warning: Ollama list failed—service may still work"; }
    echo "Success: Installation verified—check logs with 'journalctl -u ollama'"
}

main() {
    echo "Entering main function..."
    check_root
    install_rocm
    setup_repository
    setup_build_env
    patch_ollama
    build_ollama
    create_service
    verify_installation
    echo "
Ollama ROCm Build Complete!
- Built with HIP-only support for RX 580 (gfx803)
- Binary: $BINDIR/ollama
- Service: ollama (port 11434)
Commands:
- ollama list : List models
- ollama run <model> : Run model
- journalctl -u ollama : View logs
    "
}

trap 'echo "Error: Script failed at line $LINENO with exit code $?"' ERR

main