#!/bin/bash

# Ollama ROCm Source Build Script for RX 580
# Version: 1.2.6 - Built by Cortana (via Grok 3, xAI) for Jason
# Date: February 24, 2025

set -x  # Trace every command
set -e  # Exit on any error

echo "🚀 Starting Ollama ROCm Source Build for RX 580..."

OLLAMA_DIR="/home/$USER/ollama-rocm"
BINDIR="/usr/local/bin"
ROCM_PATH="/opt/rocm-6.3.3"
GO_VERSION="1.22.5"
GO_ROOT="/usr/local/go"

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "❌ Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    echo "✅ Success: Running as root"
}

setup_repository() {
    echo "📦 Cloning latest Ollama repository..."
    rm -rfv "$OLLAMA_DIR" || echo "⚠️ No old directory to remove"
    git clone https://github.com/ollama/ollama.git "$OLLAMA_DIR" || { echo "❌ Error: Git clone failed"; exit 1; }
    cd "$OLLAMA_DIR" || { echo "❌ Error: Directory change failed"; exit 1; }
    echo "Verifying clone contents..."
    ls -l llm/ || { echo "❌ Error: llm/ directory not found—clone may have failed"; exit 1; }
    echo "✅ Success: Repository cloned"
}

setup_build_env() {
    echo "⚙️ Setting up build environment..."
    sudo apt update || { echo "❌ Error: Apt update failed"; exit 1; }
    sudo apt install -y libstdc++-12-dev cmake gcc-10 g++-10 || { echo "❌ Error: Build deps install failed"; exit 1; }
    if [ ! -d "$GO_ROOT" ] || ! "$GO_ROOT/bin/go" version | grep -q "$GO_VERSION"; then
        echo "⚠️ Warning: Go $GO_VERSION not found—installing..."
        sudo rm -rfv "$GO_ROOT" || echo "⚠️ No old Go to remove"
        wget -v https://go.dev/dl/go$GO_VERSION.linux-amd64.tar.gz -O go.tar.gz || { echo "❌ Error: Go download failed"; exit 1; }
        sudo tar -C /usr/local -xzf go.tar.gz || { echo "❌ Error: Go extraction failed"; exit 1; }
        rm -fv go.tar.gz
        echo "✅ Success: Go $GO_VERSION installed"
    else
        echo "✅ Success: Go $GO_VERSION already installed—version $("$GO_ROOT/bin/go" version)"
    fi
    export GOROOT="$GO_ROOT"
    export PATH="$GO_ROOT/bin:$PATH"
    go version || { echo "❌ Error: Go version check failed"; exit 1; }
    export CC=/usr/bin/gcc-10
    export CXX=/usr/bin/g++-10

    echo "Setting ROCm-specific environment variables..."
    export CGO_CFLAGS="-I$ROCM_PATH/include -I$ROCM_PATH/hip/include"
    export CGO_LDFLAGS="-L$ROCM_PATH/lib -L$ROCM_PATH/lib64 -L$ROCM_PATH/hip/lib -lamdhip64 -lhiprtc"
    export GOFLAGS="-tags=rocm"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib"
    export HSA_OVERRIDE_GFX_VERSION="8.0.3"
    export ROC_ENABLE_PRE_VEGA="1"
    export PYTORCH_ROCM_ARCH="gfx803"
    export HIP_VISIBLE_DEVICES="0"
    echo "CGO_CFLAGS=$CGO_CFLAGS"
    echo "CGO_LDFLAGS=$CGO_LDFLAGS"
    echo "GOFLAGS=$GOFLAGS"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
    echo "ROC_ENABLE_PRE_VEGA=$ROC_ENABLE_PRE_VEGA"
    echo "PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"
    echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
    echo "✅ Success: Build environment configured for ROCm HIP (no Conda dependency)"
}

patch_ollama() {
    echo "🛠️ Patching Ollama to accept gfx803..."
    cd "$OLLAMA_DIR" || { echo "❌ Error: Directory change failed"; exit 1; }
    echo "Locating GPU detection file..."
    GPU_FILE=$(find llm -type f -name "*.go" -exec grep -l "amdgpu" {} + | head -1)
    if [ -z "$GPU_FILE" ]; then
        echo "⚠️ Warning: No file with 'amdgpu' found in llm/—trying broader search..."
        GPU_FILE=$(find . -type f -name "*.go" -exec grep -l "amdgpu" {} + | head -1)
    fi
    if [ -z "$GPU_FILE" ]; then
        echo "❌ Error: Could not locate GPU detection file with 'amdgpu'"
        ls -lR "$OLLAMA_DIR" > "$OLLAMA_DIR/dir_listing.txt"
        echo "Directory listing saved to $OLLAMA_DIR/dir_listing.txt"
        exit 1
    fi
    echo "Patching $GPU_FILE..."
    sed -i '/minimumComputeCapability/{s/return nil, err/return g, nil/}' "$GPU_FILE" || \
    sed -i '/amdgpu too old/{s/return nil, err/return g, nil/}' "$GPU_FILE" || \
    { echo "❌ Error: Failed to patch $GPU_FILE—check file for GPU rejection logic"; exit 1; }
    echo "✅ Success: Patched Ollama source in $GPU_FILE"
}

build_ollama() {
    echo "🔨 Building Ollama with ROCm HIP support..."
    cd "$OLLAMA_DIR" || { echo "❌ Error: Directory change failed"; exit 1; }
    echo "Generating Go files with ROCm HIP..."
    go generate ./... || { echo "❌ Error: Go generate failed"; exit 1; }
    echo "Building Ollama with ROCm tags..."
    go build -v -o ollama-rocm . || { echo "❌ Error: Go build failed—check above for cgo errors"; exit 1; }
    if [ ! -f ollama-rocm ]; then
        echo "❌ Error: Build failed—ollama-rocm binary not found"
        exit 1
    fi
    sudo mv -v ollama-rocm "$BINDIR/ollama-rocm" || { echo "❌ Error: Failed to move binary"; exit 1; }
    echo "✅ Success: Ollama-rocm built and installed"
}

create_service() {
    echo "🔧 Creating systemd service..."
    sudo tee /etc/systemd/system/ollama-rocm.service > /dev/null << EOF || { echo "❌ Error: Service file creation failed"; exit 1; }
[Unit]
Description=Ollama Service (ROCm)
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
    sudo systemctl daemon-reload || { echo "❌ Error: Daemon reload failed"; exit 1; }
    sudo systemctl enable ollama-rocm || { echo "❌ Error: Service enable failed"; exit 1; }
    sudo systemctl restart ollama-rocm || { echo "❌ Error: Service restart failed"; exit 1; }
    echo "✅ Success: Service created and started"
}

verify_installation() {
    echo "🔍 Verifying installation..."
    sleep 2
    $ROCM_PATH/bin/rocminfo | grep -A 5 "Name:.*gfx803" || { echo "❌ Error: rocminfo failed or RX580 not detected"; exit 1; }
    $ROCM_PATH/bin/rocm-smi || { echo "❌ Error: rocm-smi failed"; exit 1; }
    systemctl status ollama-rocm --no-pager || { echo "❌ Error: Service status check failed"; exit 1; }
    $BINDIR/ollama-rocm list || { echo "❌ Error: Ollama-rocm test failed"; exit 1; }
    echo "Checking HIP usage in logs..."
    journalctl -u ollama-rocm --since "2 minutes ago" -l | grep -i "hip" && echo "✅ HIP detected in logs" || { echo "❌ Error: No HIP usage detected—running on CPU"; exit 1; }
    echo "Testing GPU activity..."
    $BINDIR/ollama-rocm run llama2 "Hello world" & sleep 5; $ROCM_PATH/bin/rocm-smi | grep -q "[1-9]%" || { echo "❌ Error: No GPU activity detected"; exit 1; }
    echo "✅ Success: Verification complete—HIP confirmed"
}

main() {
    echo "🔧 Entering main function..."
    check_root
    setup_repository
    setup_build_env
    patch_ollama
    build_ollama
    create_service
    verify_installation
    echo "
✨ Ollama ROCm Build Complete! ✨
- Built with ROCm 6.3.3 support for RX 580
- Binary: $BINDIR/ollama-rocm
- Service: ollama-rocm (port 11435)
Commands:
- ollama-rocm list : List models
- ollama-rocm run <model> : Run model
- journalctl -u ollama-rocm : View logs
Notes:
- Models stored in /usr/share/ollama-rocm/.ollama/models
- Runs standalone—no dependency on darkpool-rocm Conda env
    "
}

trap 'echo "❌ Script failed at line $LINENO with exit code $?"' ERR

main