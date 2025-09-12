#!/bin/bash

# Ollama CUDA Source Build Script for K80s (Dual Die)
# Version: 1.3.0 - Built by Cortana (via Claude 3.7) for Jason
# Date: February 25, 2025

set -x  # Trace every command
set -e  # Exit on any error

echo "🚀 Starting Ollama CUDA Source Build for K80s (Dual Die)..."

OLLAMA_DIR="/home/$USER/ollama-cuda"
BINDIR="/usr/local/bin"
CUDA_PATH="/usr/local/cuda-11.4"
GO_VERSION="1.22.5"
GO_ROOT="/usr/local/go"
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
    mkdir -p "$MODEL_STORAGE"
    mkdir -p "$LOG_DIR"
    chown -R $USER:$USER "$MODEL_STORAGE" "$LOG_DIR"
    chmod -R 775 "$MODEL_STORAGE" "$LOG_DIR"
    echo "✅ Success: Directories created"
}

check_cuda() {
    echo "🔍 Checking CUDA 11.4 installation..."
    if [ ! -f "$CUDA_PATH/bin/nvcc" ]; then
        echo "❌ Error: CUDA 11.4 not found at $CUDA_PATH—run cuda-install-k80.sh first."
        exit 1
    fi
    $CUDA_PATH/bin/nvcc --version || { echo "❌ Error: nvcc failed—CUDA install corrupted"; exit 1; }
    nvidia-smi || { echo "❌ Error: nvidia-smi failed—NVIDIA driver may not be installed"; exit 1; }
    echo "Checking for K80 GPUs..."
    if ! nvidia-smi -L | grep -q "Tesla K80"; then
        echo "⚠️ Warning: No Tesla K80 GPUs detected. Script optimized for K80s."
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted by user."
            exit 1
        fi
    fi
    echo "✅ Success: CUDA 11.4 detected"
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
    export PATH="$GO_ROOT/bin:$CUDA_PATH/bin:$PATH"
    go version || { echo "❌ Error: Go version check failed"; exit 1; }
    export CC=/usr/bin/gcc-10
    export CXX=/usr/bin/g++-10

    echo "Setting CUDA-specific environment variables..."
    export CGO_CFLAGS="-I$CUDA_PATH/include"
    export CGO_LDFLAGS="-L$CUDA_PATH/lib64 -lcudart -lcublas -lcublasLt -lcuda"
    export GOFLAGS="-tags=cuda"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_PATH/lib64"
    echo "CGO_CFLAGS=$CGO_CFLAGS"
    echo "CGO_LDFLAGS=$CGO_LDFLAGS"
    echo "GOFLAGS=$GOFLAGS"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "✅ Success: Build environment configured for CUDA"
}

patch_ollama() {
    echo "🛠️ Patching Ollama to accept CC 3.7..."
    cd "$OLLAMA_DIR" || { echo "❌ Error: Directory change failed"; exit 1; }
    echo "Locating CUDA GPU detection file..."
    GPU_FILE=$(find llm -type f -name "*.go" -exec grep -l "minimumComputeCapability" {} + | head -1)
    if [ -z "$GPU_FILE" ]; then
        echo "⚠️ Warning: No file with 'minimumComputeCapability' found in llm/—trying broader search..."
        GPU_FILE=$(find . -type f -name "*.go" -exec grep -l "minimumComputeCapability" {} + | head -1)
    fi
    if [ -z "$GPU_FILE" ]; then
        echo "❌ Error: Could not locate GPU detection file with 'minimumComputeCapability'"
        ls -lR "$OLLAMA_DIR" > "$OLLAMA_DIR/dir_listing.txt"
        echo "Directory listing saved to $OLLAMA_DIR/dir_listing.txt"
        exit 1
    fi
    echo "Patching $GPU_FILE..."
    sed -i '/minimumComputeCapability/{s/return nil, err/return g, nil/}' "$GPU_FILE" || { echo "❌ Error: Failed to patch $GPU_FILE"; exit 1; }
    echo "✅ Success: Patched Ollama source in $GPU_FILE"

    echo "Patching CMakeLists.txt for CC 3.7..."
    sed -i 's/set(CMAKE_CUDA_ARCHITECTURES .*/set(CMAKE_CUDA_ARCHITECTURES 37)/' CMakeLists.txt || { echo "❌ Error: Failed to patch CMakeLists.txt"; exit 1; }
    echo "✅ Success: CMakeLists.txt patched for CC 3.7"
}

build_ollama() {
    echo "🔨 Building Ollama with CUDA support..."
    cd "$OLLAMA_DIR" || { echo "❌ Error: Directory change failed"; exit 1; }
    echo "Generating Go files with CUDA..."
    go generate ./... || { echo "❌ Error: Go generate failed"; exit 1; }
    echo "Building Ollama with CUDA tags..."
    go build -v -o ollama-cuda . || { echo "❌ Error: Go build failed—check above for cgo errors"; exit 1; }
    if [ ! -f ollama-cuda ]; then
        echo "❌ Error: Build failed—ollama-cuda binary not found"
        exit 1
    fi
    sudo mv -v ollama-cuda "$BINDIR/ollama-cuda0" || { echo "❌ Error: Failed to move ollama-cuda0"; exit 1; }
    sudo cp -v "$BINDIR/ollama-cuda0" "$BINDIR/ollama-cuda1" || { echo "❌ Error: Failed to copy ollama-cuda1"; exit 1; }
    echo "✅ Success: Ollama-cuda0 and ollama-cuda1 built and installed"
}

create_services() {
    echo "🔧 Creating systemd service for GPU 0..."
    sudo tee /etc/systemd/system/ollama-cuda0.service > /dev/null << EOF || { echo "❌ Error: Service file creation failed"; exit 1; }
[Unit]
Description=Ollama Service (CUDA - GPU 0)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=$BINDIR/ollama-cuda0 serve
User=$USER
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=$MODEL_STORAGE"
Environment="OLLAMA_HOST=127.0.0.1:11436"
Environment="CUDA_PATH=$CUDA_PATH"
Environment="LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_PATH/lib64"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OLLAMA_DEBUG=true"
StandardOutput=append:$LOG_DIR/ollama-cuda0.log
StandardError=append:$LOG_DIR/ollama-cuda0-error.log

[Install]
WantedBy=default.target
EOF

    echo "🔧 Creating systemd service for GPU 1..."
    sudo tee /etc/systemd/system/ollama-cuda1.service > /dev/null << EOF || { echo "❌ Error: Service file creation failed"; exit 1; }
[Unit]
Description=Ollama Service (CUDA - GPU 1)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=$BINDIR/ollama-cuda1 serve
User=$USER
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=$MODEL_STORAGE"
Environment="OLLAMA_HOST=127.0.0.1:11437"
Environment="CUDA_PATH=$CUDA_PATH"
Environment="LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_PATH/lib64"
Environment="CUDA_VISIBLE_DEVICES=1"
Environment="OLLAMA_DEBUG=true"
StandardOutput=append:$LOG_DIR/ollama-cuda1.log
StandardError=append:$LOG_DIR/ollama-cuda1-error.log

[Install]
WantedBy=default.target
EOF

    sudo systemctl daemon-reload || { echo "❌ Error: Daemon reload failed"; exit 1; }
    sudo systemctl enable ollama-cuda0 ollama-cuda1 || { echo "❌ Error: Service enable failed"; exit 1; }
    su