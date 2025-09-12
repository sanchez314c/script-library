#!/bin/bash

# Ollama ROCm Source Build Script for RX 580 (HIP-Only)
# Version: 1.2.36 - Debugged by Grok for Jason
# Date: February 26, 2025

set -x
set -e

echo "Starting Ollama ROCm Source Build for RX 580..."

OLLAMA_DIR="/home/$SUDO_USER/ollama-rocm"
BINDIR="/usr/bin"
ROCM_PATH="/opt/rocm-6.3.3"
GO_VERSION="1.23.4"
LOG_FILE="/home/$SUDO_USER/Desktop/ollama-build.log"

check_root() {
    echo "Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    [ -n "$SUDO_USER" ] || { echo "Error: SUDO_USER not set—run with sudo"; exit 1; }
    echo "Success: Running as root"
}

setup_repository() {
    echo "Cloning Ollama repository (v0.5.12)..."
    rm -rf "$OLLAMA_DIR" || echo "No old directory to remove"
    git clone --branch v0.5.12 https://github.com/ollama/ollama.git "$OLLAMA_DIR" || { echo "Error: Git clone failed"; exit 1; }
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed"; exit 1; }
    echo "Verifying clone contents..."
    ls -l llm/ || { echo "Error: llm/ directory not found—clone may have failed"; exit 1; }
    ls -l gpu/ || { echo "Warning: gpu/ directory not found—checking further"; }
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
    # Optional: Comment out gcc-10 to test default compiler
    # export CC="/usr/bin/gcc-10"
    # export CXX="/usr/bin/g++-10"

    echo "Setting HIP-specific environment variables..."
    export CGO_CFLAGS="-I$ROCM_PATH/include -I$ROCM_PATH/hip/include -D__HIP_PLATFORM_AMD__"
    export CGO_LDFLAGS="-L$ROCM_PATH/lib -L$ROCM_PATH/lib64 -L$ROCM_PATH/hip/lib -lamdhip64 -lhiprtc -lrocm_smi64 -lroctx64 -lrocfft"
    export GOFLAGS="-tags=hip"
    export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib:$LD_LIBRARY_PATH"
    export HSA_OVERRIDE_GFX_VERSION="8.0.3"
    export ROC_ENABLE_PRE_VEGA="1"
    export PYTORCH_ROCM_ARCH="gfx803"
    export HIP_VISIBLE_DEVICES="0"
    export CUDA_VISIBLE_DEVICES=""
    echo "CGO_CFLAGS=$CGO_CFLAGS"
    echo "CGO_LDFLAGS=$CGO_LDFLAGS"
    echo "GOFLAGS=$GOFLAGS"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "Checking library paths..."
    ls -l $ROCM_PATH/lib/libamdhip64* || echo "Warning: libamdhip64 not found"
    ls -l $ROCM_PATH/hip/lib/libhiprtc* || echo "Warning: libhiprtc not found"
    echo "Success: Build environment configured for ROCm (HIP-only)"
}

patch_ollama() {
    echo "Patching Ollama to force ROCm and gfx803..."
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed"; exit 1; }
    
    GPU_FILE="$OLLAMA_DIR/gpu/amd_linux.go"
    if [ ! -f "$GPU_FILE" ]; then
        echo "Error: $GPU_FILE not found—searching..."
        GPU_FILE=$(find . -type f -name "amd_linux.go" | head -1)
        if [ -z "$GPU_FILE" ]; then
            echo "Error: No amd_linux.go found"
            ls -R
            exit 1
        fi
    fi
    echo "Patching $GPU_FILE..."
    cp "$GPU_FILE" "$GPU_FILE.bak"
    sed -i '/"amdgpu too old"/s/return nil, fmt.Errorf("amdgpu too old %s", ver)/return g, nil/' "$GPU_FILE" || { echo "Error: Failed to patch too old check"; cat "$GPU_FILE"; exit 1; }
    sed -i '/"amdgpu too old"/{s/Warnf/Infof/; s/"amdgpu too old %s"/"Cortana: gfx803 accepted, GPU ID: %d"/;}' "$GPU_FILE" || { echo "Error: Failed to patch log"; cat "$GPU_FILE"; exit 1; }
    if ! grep -q '"gfx803": true' "$GPU_FILE"; then
        sed -i '/supportedTypes = map\[string\]bool{/a\    "gfx803": true,' "$GPU_FILE" || { echo "Error: Failed to add gfx803"; cat "$GPU_FILE"; exit 1; }
    fi
    grep -A 5 "Cortana" "$GPU_FILE" || echo "No Cortana string found post-patch"

    MAIN_GPU_FILE="$OLLAMA_DIR/gpu/gpu.go"
    if [ ! -f "$MAIN_GPU_FILE" ]; then
        echo "Error: $MAIN_GPU_FILE not found—searching..."
        MAIN_GPU_FILE=$(find . -type f -name "gpu.go" | head -1)
        if [ -z "$MAIN_GPU_FILE" ]; then
            echo "Error: No gpu.go found"
            ls -R
            exit 1
        fi
    fi
    echo "Patching $MAIN_GPU_FILE..."
    cp "$MAIN_GPU_FILE" "$MAIN_GPU_FILE.bak"
    sed -i '/"no compatible GPUs were discovered"/i\    if os.Getenv("OLLAMA_LLM_LIBRARY") == "rocm" {\n        log.Infof("Cortana: Forcing ROCm/HIP detection")\n        gpus, err := discoverHIP()\n        if err == nil && len(gpus) > 0 {\n            return gpus, nil\n        }\n    }' "$MAIN_GPU_FILE" || { echo "Error: Failed to force ROCm"; cat "$MAIN_GPU_FILE"; exit 1; }

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
    echo "Generating Go files with HIP..."
    /usr/local/go/bin/go generate ./... || { echo "Error: Go generate failed"; exit 1; }
    echo "Building Ollama with HIP tags..."
    CGO_ENABLED=1 /usr/local/go/bin/go build -v -x -o ollama . || { echo "Error: Go build failed—check above for cgo errors"; cat build.log; exit 1; }
    if [ ! -f ollama ]; then
        echo "Error: Build failed—ollama binary not found"
        exit 1
    fi
    mv -v ollama "$BINDIR/ollama" || { echo "Error: Failed to move binary"; exit 1; }
    echo "Success: Ollama built and installed"
}

create_service() {
    echo "Creating systemd service..."
    tee /etc/systemd/system/ollama.service > /dev/null << EOF || { echo "Error: Service file creation failed"; exit 1; }
[Unit]
Description=Ollama Service (ROCm HIP-Only)
After=network-online.target

[Service]
ExecStart=$BINDIR/ollama serve
User=$SUDO_USER
Group=$(id -gn $SUDO_USER)
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=/usr/share/ollama/.ollama/models"
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="ROCM_PATH=$ROCM_PATH"
Environment="LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib:$LD_LIBRARY_PATH"
Environment="HSA_OVERRIDE_GFX_VERSION=8.0.3"
Environment="ROC_ENABLE_PRE_VEGA=1"
Environment="PYTORCH_ROCM_ARCH=gfx803"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="CUDA_VISIBLE_DEVICES="
Environment="OLLAMA_DEBUG=true"

[Install]
WantedBy=default.target
EOF
    systemctl daemon-reload || { echo "Error: Daemon reload failed"; exit 1; }
    systemctl enable ollama || { echo "Error: Service enable failed"; exit 1; }
    systemctl restart ollama || { echo "Error: Service restart failed"; exit 1; }
    echo "Success: Service created and started"
}

verify_installation() {
    echo "Verifying installation..."
    sleep 2
    $ROCM_PATH/bin/rocminfo | grep -A 5 "Name:.*gfx803" || { echo "Error: rocminfo failed or RX580 not detected"; exit 1; }
    $ROCM_PATH/bin/rocm-smi || { echo "Error: rocm-smi failed"; exit 1; }
    systemctl status ollama --no-pager || { echo "Error: Service status check failed"; exit 1; }
    $BINDIR/ollama list || { echo "Error: Ollama test failed"; exit 1; }
    echo "Checking ROCm/HIP usage in logs..."
    journalctl -u ollama --since "2 minutes ago" -l | grep -i "Cortana" || { echo "Error: No Cortana debug logs—patches failed or HIP not used"; journalctl -u ollama --since "2 minutes ago" -l; exit 1; }
    echo "Testing GPU activity..."
    $BINDIR/ollama run llama2 "Hello world this is a test to force GPU usage" & sleep 15; $ROCM_PATH/bin/rocm-smi | grep -q "[1-9][0-9]*%" || { echo "Error: GPU usage not significant"; $ROCM_PATH/bin/rocm-smi; journalctl -u ollama --since "1 minute ago" -l; exit 1; }
    echo "Success: Verification complete—ROCm/HIP confirmed"
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
- Binary: $BINDIR/ollama
- Service: ollama (port 11434)
Commands:
- ollama list : List models
- ollama run <model> : Run model
- journalctl -u ollama : View logs
Notes:
- Models stored in /usr/share/ollama/.ollama/models
- Runs standalone—no CUDA or Conda dependency
    "
}

trap 'echo "Error: Script failed at line $LINENO with exit code $?"' ERR

main > build.log 2>&1