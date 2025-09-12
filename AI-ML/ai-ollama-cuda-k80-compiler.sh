#!/bin/bash

# Ollama ROCm Compiler Script for RX 580
# Version: 1.1.1 - Built by Cortana (via Grok 3, xAI) for Jason
# Updated for ROCm 6.3.3 - February 24, 2025

# Enable verbosity and error handling
set -x  # Trace every command
set -e  # Exit on any error

echo "🚀 Starting Ollama ROCm-Optimized Build Process for RX 580..."

# Global variables
OLLAMA_DIR="/home/$USER/ollama-rocm"
CONDA_ENV="darkpool-rocm"
ROCM_PATH="/opt/rocm-6.3.3"

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
    echo "Removing old directory if exists: $OLLAMA_DIR..."
    rm -rfv "$OLLAMA_DIR" || echo "⚠️ No old directory to remove"
    echo "Cloning Ollama repo to $OLLAMA_DIR..."
    git clone https://github.com/ollama/ollama.git "$OLLAMA_DIR" || { echo "❌ Error: Git clone failed"; exit 1; }
    echo "Changing to directory: $OLLAMA_DIR..."
    cd "$OLLAMA_DIR" || { echo "❌ Error: Directory change failed"; exit 1; }
    echo "✅ Success: Repository cloned"
}

setup_build_env() {
    echo "⚙️ Setting up build environment..."
    echo "Updating package lists..."
    sudo apt update || { echo "❌ Error: Apt update failed"; exit 1; }
    echo "Checking for Go installation..."
    if ! command -v go &> /dev/null; then
        echo "⚠️ Warning: Go not found—installing Go 1.22.5..."
        wget -v https://go.dev/dl/go1.22.5.linux-amd64.tar.gz -O go.tar.gz || { echo "❌ Error: Go download failed"; exit 1; }
        sudo tar -C /usr/local -xzf go.tar.gz || { echo "❌ Error: Go extraction failed"; exit 1; }
        rm -fv go.tar.gz
        export PATH="/usr/local/go/bin:$PATH"
        echo "✅ Success: Go 1.22.5 installed—version $(go version)"
    else
        echo "✅ Success: Go already installed—version $(go version)"
    fi

    echo "Activating darkpool-rocm Conda environment..."
    source "/root/miniconda3/bin/activate" "$CONDA_ENV" || {
        echo "❌ Error: Conda env '$CONDA_ENV' not found. Run ai-ml-docker-frameworks.sh first."
        exit 1
    }
    echo "✅ Success: Conda environment activated"

    echo "Verifying ROCm at $ROCM_PATH..."
    if [ ! -d "$ROCM_PATH" ]; then
        echo "❌ Error: ROCm not found at $ROCM_PATH. Run rocm-installer.sh first."
        exit 1
    fi
    echo "✅ Success: ROCm verified"

    echo "Setting ROCm-specific environment variables..."
    export CGO_CFLAGS="-I$ROCM_PATH/include -I$ROCM_PATH/hip/include"
    export CGO_LDFLAGS="-L$ROCM_PATH/lib -L$ROCM_PATH/lib64 -lhip_hcc"
    export GOFLAGS="-tags=rocm"
    export PATH="$PATH:/opt/rocm-6.3.3/bin"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/rocm-6.3.3/lib:/opt/rocm-6.3.3/lib64"
    echo "CGO_CFLAGS=$CGO_CFLAGS"
    echo "CGO_LDFLAGS=$CGO_LDFLAGS"
    echo "GOFLAGS=$GOFLAGS"
    echo "PATH=$PATH"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "✅ Success: Build environment configured"
}

build_ollama() {
    echo "🔨 Building Ollama with ROCm support..."
    cd "$OLLAMA_DIR" || { echo "❌ Error: Directory change failed"; exit 1; }
    echo "Generating Go files..."
    go generate ./... || { echo "❌ Error: Go generate failed"; exit 1; }
    echo "Building Ollama with ROCm tags..."
    go build -tags rocm -o ollama-rocm . || { echo "❌ Error: Go build failed"; exit 1; }
    echo "Verifying build output..."
    if [ ! -f ollama-rocm ]; then
        echo "❌ Error: Build failed—ollama-rocm binary not found"
        exit 1
    fi
    echo "✅ Success: Ollama built with ROCm support"
}

install_ollama() {
    echo "📥 Installing Ollama ROCm version..."
    echo "Copying binary to /usr/local/bin/ollama-rocm..."
    sudo cp -v "$OLLAMA_DIR/ollama-rocm" /usr/local/bin/ollama-rocm || { echo "❌ Error: Binary copy failed"; exit 1; }
    echo "Creating model directory: /usr/share/ollama-rocm/.ollama..."
    sudo mkdir -pv /usr/share/ollama-rocm/.ollama || { echo "❌ Error: Directory creation failed"; exit 1; }
    echo "Setting ownership to $USER:$USER..."
    sudo chown -Rv "$USER:$USER" /usr/share/ollama-rocm || { echo "❌ Error: Chown failed"; exit 1; }
    echo "Setting permissions to 755..."
    sudo chmod -v 755 /usr/share/ollama-rocm || { echo "❌ Error: Chmod failed"; exit 1; }
    echo "✅ Success: Ollama ROCm installed"
}

create_service() {
    echo "🔧 Creating systemd service..."
    echo "Writing service file to /etc/systemd/system/ollama-rocm.service..."
    sudo tee /etc/systemd/system/ollama-rocm.service > /dev/null << EOF || { echo "❌ Error: Service file creation failed"; exit 1; }
[Unit]
Description=Ollama Service (ROCm)
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama-rocm serve
User=$USER
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=/usr/share/ollama-rocm/.ollama/models"
Environment="OLLAMA_HOST=127.0.0.1:11435"

[Install]
WantedBy=default.target
EOF
    echo "Reloading systemd daemon..."
    sudo systemctl daemon-reload || { echo "❌ Error: Daemon reload failed"; exit 1; }
    echo "Enabling ollama-rocm service..."
    sudo systemctl enable ollama-rocm || { echo "❌ Error: Service enable failed"; exit 1; }
    echo "Starting ollama-rocm service..."
    sudo systemctl restart ollama-rocm || { echo "❌ Error: Service restart failed"; exit 1; }
    echo "✅ Success: Service created and started"
}

verify_installation() {
    echo "🔍 Verifying installation..."
    echo "Waiting 2 seconds for service to stabilize..."
    sleep 2
    echo "Checking ROCm GPU info with rocminfo..."
    /opt/rocm-6.3.3/bin/rocminfo | grep -A 5 "Name:.*gfx803" || { echo "❌ Error: rocminfo failed or RX580 not detected"; exit 1; }
    echo "Checking GPU status with rocm-smi..."
    /opt/rocm-6.3.3/bin/rocm-smi || { echo "❌ Error: rocm-smi failed"; exit 1; }
    echo "Checking service status..."
    systemctl status ollama-rocm --no-pager || { echo "❌ Error: Service status check failed"; exit 1; }
    echo "Testing Ollama binary..."
    /usr/local/bin/ollama-rocm list || { echo "❌ Error: Ollama test failed"; exit 1; }
    echo "✅ Success: Verification complete"
}

main() {
    echo "🔧 Entering main function..."
    check_root
    setup_repository
    setup_build_env
    build_ollama
    install_ollama
    create_service
    verify_installation
    echo "
✨ Ollama ROCm Build Complete! ✨
- Built for RX 580 with ROCm 6.3.3
- Binary: /usr/local/bin/ollama-rocm
- Service: ollama-rocm (port 11435)
Commands:
- ollama-rocm list : List models
- ollama-rocm run <model> : Run model
- journalctl -u ollama-rocm : View logs
Notes:
- Uses darkpool-rocm env—activate with 'conda activate darkpool-rocm' if running manually
    "
}

# Trap errors with line numbers
trap 'echo "❌ Script failed at line $LINENO with exit code $?"' ERR

main
