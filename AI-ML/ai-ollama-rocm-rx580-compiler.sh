#!/bin/bash

# Ollama ROCm Installer Script for RX 580
# Version: 1.1.11 - Built by Cortana (via Grok 3, xAI) for Jason
# Date: February 24, 2025

set -x  # Trace every command
set -e  # Exit on any error

echo "🚀 Starting Ollama ROCm Installation for RX 580..."

OLLAMA_DIR="/home/$USER/ollama-rocm"
OLLAMA_INSTALL_DIR="/usr/local"
BINDIR="/usr/local/bin"
ROCM_PATH="/opt/rocm-6.3.3"

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "❌ Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    echo "✅ Success: Running as root"
}

install_ollama() {
    echo "📦 Installing Ollama with ROCm support..."
    mkdir -pv "$OLLAMA_DIR" || { echo "❌ Error: Failed to create $OLLAMA_DIR"; exit 1; }
    cd "$OLLAMA_DIR" || { echo "❌ Error: Directory change failed"; exit 1; }

    echo "Downloading Ollama install script..."
    curl -fsSL https://ollama.com/install.sh -o ollama-install.sh || { echo "❌ Error: Failed to download install.sh"; exit 1; }
    chmod +x ollama-install.sh

    echo "Running Ollama install script with ROCm forced..."
    export ROCM_PATH="$ROCM_PATH"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib"
    export HSA_OVERRIDE_GFX_VERSION="8.0.3"  # Force gfx803 compatibility
    echo "ROCM_PATH=$ROCM_PATH"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
    ./ollama-install.sh || { echo "❌ Error: Install script failed"; exit 1; }

    echo "Renaming binary to ollama-rocm..."
    sudo mv -v "$BINDIR/ollama" "$BINDIR/ollama-rocm" || { echo "❌ Error: Failed to rename ollama to ollama-rocm"; exit 1; }

    if [ ! -f "$BINDIR/ollama-rocm" ]; then
        echo "❌ Error: Ollama-rocm binary not found in $BINDIR"
        exit 1
    fi
    echo "✅ Success: Ollama-rocm installed to $BINDIR"
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
    journalctl -u ollama-rocm --since "2 minutes ago" | grep -i "hip" && echo "✅ HIP detected in logs" || { echo "❌ Error: No HIP usage detected—running on CPU"; exit 1; }
    echo "Testing GPU activity..."
    $BINDIR/ollama-rocm run llama2 "Hello world" & sleep 5; $ROCM_PATH/bin/rocm-smi | grep -q "[1-9]%" || { echo "❌ Error: No GPU activity detected"; exit 1; }
    echo "✅ Success: Verification complete—HIP confirmed"
}

main() {
    echo "🔧 Entering main function..."
    check_root
    install_ollama
    create_service
    verify_installation
    echo "
✨ Ollama ROCm Installation Complete! ✨
- Installed with ROCm 6.3.3 support for RX 580
- Binary: $BINDIR/ollama-rocm
- Service: ollama-rocm (port 11435)
Commands:
- ollama-rocm list : List models
- ollama-rocm run <model> : Run model
- journalctl -u ollama-rocm : View logs
Notes:
- Models stored in /usr/share/ollama-rocm/.ollama/models
    "
}

trap 'echo "❌ Script failed at line $LINENO with exit code $?"' ERR

main
