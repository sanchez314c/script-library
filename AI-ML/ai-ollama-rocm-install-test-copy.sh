#!/bin/bash

# Ollama ROCm Source Build Script for RX 580 (HIP-Only)
# Version: 1.4.27 - Updated by Cortana for Jason
# Date: February 26, 2025
# Built against Ollama commit: 0667bad

set -x
set -e

echo "Starting Ollama ROCm Source Build for RX 580 (CMake/HIPBLAS)..."

OLLAMA_DIR="/home/$USER/ollama-rocm"
BINDIR="/usr/local/bin"
ROCM_PATH="/opt/rocm-6.3.3"
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
    echo "Cloning Ollama repository (latest commit: 0667bad)..."
    rm -rf "$OLLAMA_DIR" || echo "No old directory to remove"
    git clone https://github.com/ollama/ollama.git "$OLLAMA_DIR" || { echo "Error: Git clone failed"; exit 1; }
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed"; exit 1; }
    git checkout 0667bad || { echo "Error: Checkout to commit 0667bad failed"; exit 1; }
    echo "Verifying clone contents..."
    ls -l ml/backend/ggml || { echo "Error: ml/backend/ggml not found—clone may have failed"; exit 1; }
    echo "Success: Repository cloned at commit 0667bad"
}

setup_build_env() {
    echo "Setting up build environment..."
    apt update || { echo "Error: Apt update failed"; exit 1; }
    apt install -y git cmake build-essential rocm-dev rocm-libs rocm-hip-sdk librocblas-dev || { echo "Error: Build deps install failed"; exit 1; }
    echo "Success: Build environment configured for ROCm (CMake/HIPBLAS)"
}

build_ollama() {
    echo "Building Ollama with ROCm/HIPBLAS support (CMake)..."
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed"; exit 1; }
    mkdir -p build
    cd build
    # Force HIPBLAS, disable native CPU opts, point to ROCm
    cmake .. -DGGML_HIPBLAS=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_C_COMPILER=/opt/rocm-6.3.3/bin/hipcc \
             -DCMAKE_CXX_COMPILER=/opt/rocm-6.3.3/bin/hipcc \
             -DROCM_PATH="$ROCM_PATH" || { echo "Error: CMake configuration failed"; exit 1; }
    make -j$(nproc) || { echo "Error: Make build failed"; exit 1; }
    sudo make install || { echo "Error: Make install failed"; exit 1; }
    echo "Success: Ollama built and installed with HIPBLAS"
}

create_service() {
    echo "Creating systemd service..."
    tee /etc/systemd/system/ollama-rocm.service > /dev/null << EOF || { echo "Error: Service file creation failed"; exit 1; }
[Unit]
Description=Ollama Service (ROCm HIPBLAS)
After=network-online.target

[Service]
ExecStart=$BINDIR/ollama serve
User=$USER
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=/usr/share/ollama/.ollama/models"
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
    rocminfo | grep -A 5 "Name:.*gfx803" || { echo "Error: rocminfo failed or RX580 not detected"; exit 1; }
    rocm-smi || { echo "Error: rocm-smi failed"; exit 1; }
    systemctl status ollama-rocm --no-pager || { echo "Error: Service status check failed"; exit 1; }
    ollama list || { echo "Error: Ollama test failed"; exit 1; }
    echo "Checking HIP usage in logs..."
    journalctl -u ollama-rocm --since "2 minutes ago" -l | grep -i "hip" && echo "Success: HIP detected in logs" || { echo "Error: No HIP usage detected—running on CPU"; exit 1; }
    echo "Testing GPU activity..."
    ollama run llama2 "Hello world" & sleep 5; rocm-smi | grep -q "%" || { echo "Error: No GPU activity detected"; exit 1; }
    echo "Success: Verification complete—HIPBLAS confirmed"
}

main() {
    echo "Entering main function..."
    check_root
    setup_repository
    setup_build_env
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
