#!/bin/bash

# Ollama ROCm Source Build Script for RX 580 (HIP-Only)
# Version: 1.3.1 - Built by Cortana (via Grok 3, xAI) for Jason
# Date: February 25, 2025 - Fixed permissions for external drives

set -x  # Trace every command
set -e  # Exit on any error

echo "🚀 Starting Ollama ROCm Source Build for RX 580..."

OLLAMA_DIR="/home/$USER/ollama-rocm"
BINDIR="/usr/local/bin"
ROCM_PATH="/opt/rocm-6.3.3"
GO_VERSION="1.23.4"
MODEL_STORAGE="/media/heathen-admin/llmRAID/AI/Models"
LOG_DIR="/media/heathen-admin/llmRAID/AI/Logs/Ollama"

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "❌ Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    echo "✅ Success: Running as root"
}

ensure_directories() {
    echo "📁 Creating required directories..."
    # Create directories, but don’t chown if on external drive—use user permissions
    if [ ! -d "$MODEL_STORAGE" ]; then
        mkdir -p "$MODEL_STORAGE" || { echo "❌ Error: Failed to create $MODEL_STORAGE"; exit 1; }
    fi
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR" || { echo "❌ Error: Failed to create $LOG_DIR"; exit 1; }
    fi
    # Check if directories are on external drive and adjust permissions for user
    if mountpoint -q /media/heathen-admin/llmRAID 2>/dev/null; then
        chown -R $USER:$USER "$MODEL_STORAGE" "$LOG_DIR" 2>/dev/null || echo "⚠️ Warning: Couldn’t change ownership—external drive permissions may restrict root changes. Verify manually."
        chmod -R 775 "$MODEL_STORAGE" "$LOG_DIR" 2>/dev/null || echo "⚠️ Warning: Couldn’t set permissions—check drive access."
    else
        chown -R $USER:$USER "$MODEL_STORAGE" "$LOG_DIR" || { echo "❌ Error: Failed to change ownership"; exit 1; }
        chmod -R 775 "$MODEL_STORAGE" "$LOG_DIR" || { echo "❌ Error: Failed to set permissions"; exit 1; }
    fi
    echo "✅ Success: Directories created and permissions adjusted"
}

check_rocm() {
    echo "🔍 Checking ROCm installation..."
    if [ ! -d "$ROCM_PATH" ]; then
        echo "❌ Error: ROCm not found at $ROCM_PATH. Run rocm-installer.sh first."
        exit 1
    fi
    if ! $ROCM_PATH/bin/rocminfo &>/dev/null; then
        echo "❌ Error: rocminfo failed - ROCm may not be installed correctly."
        exit 1
    fi
    echo "✅ Success: ROCm installation verified for RX 580 (gfx803)"
}

setup_repository() {
    echo "📦 Cloning Ollama repository (v0.5.12)..."
    rm -rf "$OLLAMA_DIR" || echo "⚠️ No old directory to remove"
    git clone --branch v0.5.12 https://github.com/ollama/ollama.git "$OLLAMA_DIR" || { echo "❌ Error: Git clone failed"; exit 1; }
    cd "$OLLAMA_DIR" || { echo "❌ Error: Directory change failed"; exit 1; }
    echo "Verifying clone contents..."
    ls -l llm/ || { echo "❌ Error: llm/ directory not found—clone may have failed"; exit 1; }
    echo "✅ Success: Repository cloned"
}

setup_build_env() {
    echo "⚙️ Setting up build environment..."
    apt update || { echo "❌ Error: Apt update failed"; exit 1; }
    apt install -y libstdc++-12-dev cmake gcc-10 g++-10 git librocprim-dev || { echo "❌ Error: Build deps install failed"; exit 1; }
    apt remove -y golang-go golang || true
    rm -rf /usr/local/go /usr/bin/go /usr/local/bin/go || true
    wget -v https://go.dev/dl/go$GO_VERSION.linux-amd64.tar.gz -O go.tar.gz || { echo "❌ Error: Go download failed"; exit 1; }
    tar -C /usr/local -xzf go.tar.gz || { echo "❌ Error: Go extraction failed"; exit 1; }
    rm -fv go.tar.gz
    echo "Installed Go version: $(/usr/local/go/bin/go version)"
    export PATH="/usr/local/go/bin:$PATH"
    export CC="/usr/bin/gcc-10"
    export CXX="/usr/bin/g++-10"

    echo "Setting HIP-specific environment variables for RX 580..."
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
    echo "✅ Success: Build environment configured for ROCm (HIP-only) on RX 580"
}

patch_ollama() {
    echo "🔧 Patching Ollama to accept gfx803 and shim get_compiler..."
    cd "$OLLAMA_DIR" || { echo "❌ Error: Directory change failed"; exit 1; }
    GPU_FILE=$(find . -type f -name "*.go" -exec grep -l "amdgpu" {} + | head -1)
    if [ -z "$GPU_FILE" ]; then
        echo "❌ Error: Could not locate GPU detection file with 'amdgpu'"
        ls -lR "$OLLAMA_DIR" > "$OLLAMA_DIR/dir_listing.txt"
        exit 1
    fi
    echo "Patching $GPU_FILE..."
    sed -i '/amdgpu too old/{s/return nil, err/return g, nil/}' "$GPU_FILE" || { echo "❌ Error: Failed to patch $GPU_FILE"; exit 1; }
    
    # Append shim to llm/llama_linux.go
    echo "Patching llm/llama_linux.go with get_compiler shim..."
    cat << 'EOF' >> llm/llama_linux.go

//export get_compiler
func get_compiler() *C.char {
    return C.CString("hipcc")  // Shim to bypass linker for RX 580 (gfx803)
}
EOF
    echo "✅ Success: Patched Ollama source for RX 580"
}

build_ollama() {
    echo "🔨 Building Ollama with ROCm (HIP-only) support for RX 580..."
    cd "$OLLAMA_DIR" || { echo "❌ Error: Directory change failed"; exit 1; }
    echo "Generating Go files with HIP..."
    CGO_ENABLED=1 /usr/local/go/bin/go generate ./... || { echo "❌ Error: Go generate failed"; exit 1; }
    echo "Building Ollama with HIP tags..."
    CGO_ENABLED=1 /usr/local/go/bin/go build -v -x -o ollama-rocm . || { echo "❌ Error: Go build failed—check above for cgo errors"; exit 1; }
    if [ ! -f ollama-rocm ]; then
        echo "❌ Error: Build failed—ollama-rocm binary not found"
        exit 1
    fi
    mv -v ollama-rocm "$BINDIR/ollama-rocm" || { echo "❌ Error: Failed to move binary"; exit 1; }
    echo "✅ Success: Ollama-rocm built and installed for RX 580"
}

create_service() {
    echo "🔧 Creating systemd service..."
    tee /etc/systemd/system/ollama-rocm.service > /dev/null << EOF || { echo "❌ Error: Service file creation failed"; exit 1; }
[Unit]
Description=Ollama Service (ROCm HIP-Only for RX 580)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=$BINDIR/ollama-rocm serve
User=$USER
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=$MODEL_STORAGE"
Environment="OLLAMA_HOST=127.0.0.1:11435"
Environment="ROCM_PATH=$ROCM_PATH"
Environment="LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib"
Environment="HSA_OVERRIDE_GFX_VERSION=8.0.3"
Environment="ROC_ENABLE_PRE_VEGA=1"
Environment="PYTORCH_ROCM_ARCH=gfx803"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="OLLAMA_DEBUG=true"
StandardOutput=append:$LOG_DIR/ollama-rocm.log
StandardError=append:$LOG_DIR/ollama-rocm-error.log

[Install]
WantedBy=default.target
EOF
    systemctl daemon-reload || { echo "❌ Error: Daemon reload failed"; exit 1; }
    systemctl enable ollama-rocm || { echo "❌ Error: Service enable failed"; exit 1; }
    systemctl restart ollama-rocm || { echo "❌ Error: Service restart failed"; exit 1; }
    echo "✅ Success: Service created and started for RX 580"
}

verify_installation() {
    echo "🔍 Verifying installation..."
    sleep 5
    $ROCM_PATH/bin/rocminfo | grep -A 5 "Name:.*gfx803" || { echo "❌ Error: rocminfo failed or RX 580 not detected"; exit 1; }
    $ROCM_PATH/bin/rocm-smi || { echo "❌ Error: rocm-smi failed"; exit 1; }
    systemctl status ollama-rocm --no-pager || { echo "❌ Error: Service status check failed"; exit 1; }
    $BINDIR/ollama-rocm list || { echo "❌ Error: Ollama test failed"; exit 1; }
    echo "Checking HIP usage in logs..."
    journalctl -u ollama-rocm --since "2 minutes ago" -l | grep -i "hip" && echo "✅ Success: HIP detected in logs" || { echo "⚠️ Warning: No HIP usage detected yet—may appear after first model run"; }
    echo "✅ Success: Verification complete for RX 580"
}

download_starter_model() {
    echo "🧠 Downloading a starter model to verify GPU usage..."
    $BINDIR/ollama-rocm pull tinyllama || echo "⚠️ Warning: Failed to download starter model—will try on first use"
    echo "Testing GPU activity..."
    timeout 30s $BINDIR/ollama-rocm run tinyllama "Hello, world" &
    sleep 5
    $ROCM_PATH/bin/rocm-smi | grep -q "[1-9]%" && echo "✅ Success: GPU activity detected" || echo "⚠️ Warning: No GPU activity detected—check logs for details"
    echo "✅ Success: Starter model test complete for RX 580"
}

main() {
    echo "🔧 Entering main function..."
    check_root
    ensure_directories
    check_rocm
    setup_repository
    setup_build_env
    patch_ollama
    build_ollama
    create_service
    verify_installation
    download_starter_model
    echo "
✨ Ollama ROCm Build Complete! ✨
- Built with HIP-only support for RX 580 (gfx803)
- Binary: $BINDIR/ollama-rocm
- Service: ollama-rocm (port 11435)
- Models stored in: $MODEL_STORAGE
- Logs stored in: $LOG_DIR

Commands:
- ollama-rocm list : List models
- ollama-rocm run <model> : Run model
- journalctl -u ollama-rocm : View logs
    "
}

trap 'echo "❌ Script failed at line $LINENO with exit code $?"' ERR

main