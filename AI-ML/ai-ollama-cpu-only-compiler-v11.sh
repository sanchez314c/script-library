#!/bin/bash

# Ollama CPU-Only Installer Script
# Version: 1.1.0 - Built by Cortana (via Claude 3.7) for Jason
# Date: February 25, 2025

set -x  # Trace every command
set -e  # Exit on any error

echo "🚀 Starting Ollama CPU-Only Installation..."

OLLAMA_DIR="/home/$USER/ollama-cpu"
OLLAMA_INSTALL_DIR="/usr/local"
BINDIR="/usr/local/bin"
MODEL_STORAGE="/media/heathen-admin/llmRAID/AI/Models"

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "❌ Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    echo "✅ Success: Running as root"
}

ensure_model_storage() {
    echo "📁 Ensuring model storage directory exists..."
    mkdir -p "$MODEL_STORAGE"
    chown -R $USER:$USER "$MODEL_STORAGE"
    chmod -R 775 "$MODEL_STORAGE"
    echo "✅ Success: Model storage directory ready"
}

install_ollama_cpu() {
    echo "📦 Installing CPU-only Ollama..."
    echo "Creating install directory: $OLLAMA_DIR..."
    mkdir -pv "$OLLAMA_DIR" || { echo "❌ Error: Failed to create $OLLAMA_DIR"; exit 1; }
    cd "$OLLAMA_DIR" || { echo "❌ Error: Directory change failed"; exit 1; }

    echo "Downloading Ollama install script..."
    curl -fsSL https://ollama.com/install.sh -o ollama-install.sh || { echo "❌ Error: Failed to download install.sh"; exit 1; }
    chmod +x ollama-install.sh

    echo "Running Ollama install script with CPU-only mode..."
    export OLLAMA_NO_GPU="1"  # Force CPU-only by disabling GPU detection
    echo "OLLAMA_NO_GPU=$OLLAMA_NO_GPU"
    ./ollama-install.sh || { echo "❌ Error: Install script failed"; exit 1; }

    echo "Renaming binary to ollama-cpu..."
    sudo mv -v "$BINDIR/ollama" "$BINDIR/ollama-cpu" || { echo "❌ Error: Failed to rename ollama to ollama-cpu"; exit 1; }

    echo "Verifying Ollama binary..."
    if [ ! -f "$BINDIR/ollama-cpu" ]; then
        echo "❌ Error: Ollama-cpu binary not found in $BINDIR"
        exit 1
    fi
    echo "✅ Success: Ollama-cpu installed to $BINDIR"
}

create_service() {
    echo "🔧 Creating systemd service for CPU-only Ollama..."
    echo "Writing service file to /etc/systemd/system/ollama-cpu.service..."
    sudo tee /etc/systemd/system/ollama-cpu.service > /dev/null << EOF || { echo "❌ Error: Service file creation failed"; exit 1; }
[Unit]
Description=Ollama Service (CPU-Only)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=$BINDIR/ollama-cpu serve
User=$USER
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=$MODEL_STORAGE"
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_NO_GPU=1"

[Install]
WantedBy=default.target
EOF
    echo "Reloading systemd daemon..."
    sudo systemctl daemon-reload || { echo "❌ Error: Daemon reload failed"; exit 1; }
    echo "Enabling ollama-cpu service..."
    sudo systemctl enable ollama-cpu || { echo "❌ Error: Service enable failed"; exit 1; }
    echo "Starting ollama-cpu service..."
    sudo systemctl restart ollama-cpu || { echo "❌ Error: Service restart failed"; exit 1; }
    echo "✅ Success: Service created and started"
}

verify_installation() {
    echo "🔍 Verifying CPU-only installation..."
    echo "Waiting 5 seconds for service to stabilize..."
    sleep 5
    echo "Checking service status..."
    systemctl status ollama-cpu --no-pager || { echo "❌ Error: Service status check failed"; exit 1; }
    echo "Testing Ollama-cpu binary..."
    $BINDIR/ollama-cpu list || { echo "❌ Error: Ollama-cpu test failed"; exit 1; }
    echo "Checking CPU-only mode in logs..."
    journalctl -u ollama-cpu --since "2 minutes ago" | grep -i "cpu\|no compatible GPUs" || echo "⚠️ Warning: No CPU-only confirmation in logs"
    echo "✅ Success: Verification complete"
}

download_starter_model() {
    echo "🧠 Downloading a starter model (tiny)..."
    $BINDIR/ollama-cpu pull tinyllama || echo "⚠️ Warning: Failed to download starter model - will try on first use"
    echo "✅ Success: Starter model downloaded"
}

main() {
    echo "🔧 Entering main function..."
    check_root
    ensure_model_storage
    install_ollama_cpu
    create_service
    verify_installation
    download_starter_model
    echo "
✨ Ollama CPU-Only Installation Complete! ✨
- Installed for CPU-only operation
- Binary: $BINDIR/ollama-cpu
- Service: ollama-cpu (port 11434)
- Models stored in: $MODEL_STORAGE
Commands:
- ollama-cpu list : List models
- ollama-cpu run <model> : Run model
- journalctl -u ollama-cpu : View logs
    "
}

trap 'echo "❌ Script failed at line $LINENO with exit code $?"' ERR

main
