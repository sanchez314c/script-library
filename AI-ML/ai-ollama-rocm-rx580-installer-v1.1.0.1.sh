#!/bin/bash

# Ollama ROCm Compiler Script for RX 580
# Version: 1.1.0 - Built by Cortana (via Grok 3, xAI) for Jason
# Updated for ROCm 6.3 - February 23, 2025
set -e

echo "🚀 Starting Ollama ROCm-Optimized Build Process for RX 580..."

# Global variables
OLLAMA_DIR="/home/$USER/ollama-rocm"
CONDA_ENV="darkpool-rocm"
ROCM_PATH="/opt/rocm-6.3"

check_root() {
    echo "🔍 Checking for root privileges..."
    [ "$(id -u)" != "0" ] && { echo "❌ Requires root. Run with sudo."; exit 1; }
    echo "✅ Running as root"
}

setup_repository() {
    echo "📦 Cloning latest Ollama repository..."
    rm -rf "$OLLAMA_DIR"  # Clean slate
    git clone https://github.com/ollama/ollama.git "$OLLAMA_DIR"
    cd "$OLLAMA_DIR"
    echo "✅ Repository cloned"
}

setup_build_env() {
    echo "⚙️ Setting up build environment..."
    sudo apt update
    sudo apt install -y libhip-dev  # ROCm-specific dep

    # Check Conda and activate darkpool-rocm
    source "$HOME/miniconda3/bin/activate" "$CONDA_ENV" || {
        echo "❌ Conda env '$CONDA_ENV' not found. Run ai-ml-docker-frameworks.sh first."
        exit 1
    }

    # Verify ROCm
    [ -d "$ROCM_PATH" ] || { echo "❌ ROCm not at $ROCM_PATH. Run rocm-installer.sh first."; exit 1; }

    # Use env’s ROCm vars, add build-specific ones
    export CGO_CFLAGS="-I$ROCM_PATH/include -I$ROCM_PATH/hip/include"
    export CGO_LDFLAGS="-L$ROCM_PATH/lib -L$ROCM_PATH/lib64 -lhip_hcc"
    export GOFLAGS="-tags=rocm"
    echo "✅ Build environment configured"
}

build_ollama() {
    echo "🔨 Building Ollama with ROCm support..."
    cd "$OLLAMA_DIR"
    go generate ./...
    go build -tags rocm -o ollama-rocm .
    [ -f ollama-rocm ] || { echo "❌ Build failed: ollama-rocm not found."; exit 1; }
    echo "✅ Ollama built"
}

install_ollama() {
    echo "📥 Installing Ollama ROCm version..."
    sudo cp "$OLLAMA_DIR/ollama-rocm" /usr/local/bin/ollama-rocm
    sudo mkdir -p /usr/share/ollama-rocm/.ollama
    sudo chown -R "$USER:$USER" /usr/share/ollama-rocm
    sudo chmod 755 /usr/share/ollama-rocm
    echo "✅ Ollama installed"
}

create_service() {
    echo "🔧 Creating systemd service..."
    sudo tee /etc/systemd/system/ollama-rocm.service > /dev/null << EOF
[Unit]
Description=Ollama Service (ROCm)
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama-rocm serve
User=$USER
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=/usr/share/ollama-rocm/.ollama/models"
Environment="OLLAMA_HOST=127.0.0.1:11434"

[Install]
WantedBy=default.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable ollama-rocm
    sudo systemctl restart ollama-rocm
    echo "✅ Service started"
}

verify_installation() {
    echo "🔍 Verifying installation..."
    sleep 2
    echo "ROCm GPU Info:"; rocminfo | grep -A 5 "Name:.*gfx803" || echo "❌ rocminfo failed"
    echo "GPU Status:"; rocm-smi || echo "❌ rocm-smi failed"
    echo "Service Status:"; systemctl status ollama-rocm --no-pager || echo "❌ Service failed"
    echo "Ollama Test:"; ollama-rocm list || echo "❌ Ollama test failed"
    echo "✅ Verification complete"
}

main() {
    check_root
    setup_repository
    setup_build_env
    build_ollama
    install_ollama
    create_service
    verify_installation
    echo "
✨ Ollama ROCm Build Complete! ✨
- Built for RX 580 with ROCm 6.3
- Binary: /usr/local/bin/ollama-rocm
- Service: ollama-rocm (port 11434)
Commands:
- ollama-rocm list : List models
- ollama-rocm run <model> : Run model
- journalctl -u ollama-rocm : View logs
Notes:
- Uses $CONDA_ENV env—activate with 'dr' if running manually
"
}

main