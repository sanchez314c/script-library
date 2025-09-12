#!/bin/bash

# Ollama ROCm Install Script for RX 580 (HIP-Only)
# Version: 1.2.28 - Built by Cortana for Jason
# Date: February 26, 2025, 3:45 AM EST

set -x
set -e

echo "Starting Ollama ROCm Install for RX 580..."

OLLAMA_DIR="/usr/share/ollama"
BINDIR="/usr/bin"
ROCM_PATH="/opt/rocm-6.3.3"
LOG_FILE="/home/$USER/Desktop/ollama-install.log"

check_root() {
    echo "Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    echo "Success: Running as root"
}

install_ollama() {
    echo "Running official Ollama install script..."
    curl -fsSL https://ollama.com/install.sh | sh || { echo "Error: Official install script failed"; exit 1; }
    if [ ! -f "$BINDIR/ollama" ]; then
        echo "Error: Ollama binary not found after install"
        exit 1
    fi
    echo "Success: Ollama installed via official script"
}

setup_build_env() {
    echo "Setting up ROCm environment..."
    apt update || { echo "Error: Apt update failed"; exit 1; }
    apt install -y libstdc++-12-dev librocprim-dev || { echo "Error: ROCm deps install failed"; exit 1; }
    
    echo "Setting HIP-specific environment variables..."
    export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib:$LD_LIBRARY_PATH"
    export HSA_OVERRIDE_GFX_VERSION="8.0.3"
    export ROC_ENABLE_PRE_VEGA="1"
    export PYTORCH_ROCM_ARCH="gfx803"
    export HIP_VISIBLE_DEVICES="0"
    export CUDA_VISIBLE_DEVICES=""
    export OLLAMA_LLM_LIBRARY="rocm"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "Success: ROCm environment configured"
}

patch_service() {
    echo "Patching Ollama systemd service for ROCm..."
    SERVICE_FILE="/etc/systemd/system/ollama.service"
    if [ ! -f "$SERVICE_FILE" ]; then
        echo "Error: Ollama service file not found—install may have failed"
        exit 1
    fi
    cp "$SERVICE_FILE" "$SERVICE_FILE.bak"
    tee "$SERVICE_FILE" > /dev/null << EOF || { echo "Error: Service file update failed"; exit 1; }
[Unit]
Description=Ollama Service (ROCm HIP-Only)
After=network-online.target

[Service]
ExecStart=$BINDIR/ollama serve
User=$USER
Group=$USER
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=$OLLAMA_DIR/.ollama/models"
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="ROCM_PATH=$ROCM_PATH"
Environment="LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib:$LD_LIBRARY_PATH"
Environment="HSA_OVERRIDE_GFX_VERSION=8.0.3"
Environment="ROC_ENABLE_PRE_VEGA=1"
Environment="PYTORCH_ROCM_ARCH=gfx803"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="CUDA_VISIBLE_DEVICES="
Environment="OLLAMA_DEBUG=true"
Environment="OLLAMA_LLM_LIBRARY=rocm"

[Install]
WantedBy=default.target
EOF
    systemctl daemon-reload || { echo "Error: Daemon reload failed"; exit 1; }
    systemctl restart ollama || { echo "Error: Service restart failed"; exit 1; }
    echo "Success: Service patched and restarted"
}

verify_installation() {
    echo "Verifying installation..."
    sleep 2
    $ROCM_PATH/bin/rocminfo | grep -A 5 "Name:.*gfx803" || { echo "Error: rocminfo failed or RX580 not detected"; exit 1; }
    $ROCM_PATH/bin/rocm-smi || { echo "Error: rocm-smi failed"; exit 1; }
    systemctl status ollama --no-pager || { echo "Error: Service status check failed"; exit 1; }
    $BINDIR/ollama list || { echo "Error: Ollama test failed"; exit 1; }
    echo "Checking ROCm/HIP usage in logs..."
    journalctl -u ollama --since "2 minutes ago" -l | grep -i "rocm" || { echo "Error: No ROCm usage detected—running on CPU"; journalctl -u ollama --since "2 minutes ago" -l; exit 1; }
    echo "Testing GPU activity..."
    $BINDIR/ollama run llama2 "Hello world this is a test to force GPU usage" & sleep 15; $ROCM_PATH/bin/rocm-smi | grep -q "[1-9][0-9]*%" || { echo "Error: GPU usage not significant"; $ROCM_PATH/bin/rocm-smi; journalctl -u ollama --since "1 minute ago" -l; exit 1; }
    echo "Success: Verification complete—ROCm/HIP confirmed"
}

main() {
    echo "Entering main function..."
    check_root
    install_ollama
    setup_build_env
    patch_service
    verify_installation
    echo "
Ollama ROCm Install Complete!
- Built with HIP-only support for RX 580 (gfx803)
- Binary: $BINDIR/ollama
- Service: ollama (port 11434)
Commands:
- ollama list : List models
- ollama run <model> : Run model
- journalctl -u ollama : View logs
Notes:
- Models stored in $OLLAMA_DIR/.ollama/models
- Runs with standard Ollama install, patched for ROCm
    "
}

trap 'echo "Error: Script failed at line $LINENO with exit code $?"' ERR

main