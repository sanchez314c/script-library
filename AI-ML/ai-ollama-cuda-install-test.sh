#!/bin/bash

# Ollama CUDA Compiler Script for Tesla K80 (Dual GPUs)
# Version: 1.1.2 - Built by Cortana (via Grok 3, xAI) for Jason
# Date: February 24, 2025

# Enable verbosity and error handling
set -x  # Trace every command
set -e  # Exit on any error

echo "🚀 Starting Ollama K80-Optimized Build Process for Dual GPUs..."

# Global variables
OLLAMA_DIR="/home/$USER/ollama-k80"
CUDA_PATH="/usr/local/cuda-11.4"

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "❌ Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    echo "✅ Success: Running as root"
}

setup_repository() {
    echo "📦 Cloning Ollama repository..."
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
    echo "Installing GCC 10 and G++ 10 for CUDA 11.4..."
    sudo apt install -y gcc-10 g++-10 || { echo "❌ Error: GCC/G++ install failed"; exit 1; }
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

    echo "Verifying CUDA at $CUDA_PATH..."
    if [ ! -d "$CUDA_PATH" ]; then
        echo "❌ Error: CUDA not found at $CUDA_PATH. Run cuda-install-k80.sh first."
        exit 1
    fi
    echo "✅ Success: CUDA verified"

    echo "Setting build environment variables..."
    export CC=/usr/bin/gcc-10
    export CXX=/usr/bin/g++-10
    export PATH="$CUDA_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
    export CUDAHOSTCXX=/usr/bin/g++-10
    export CGO_CFLAGS="-I$CUDA_PATH/include"
    export CGO_LDFLAGS="-L$CUDA_PATH/lib64 -lcudart"
    export GOFLAGS="-tags=cuda"
    echo "CC=$CC"
    echo "CXX=$CXX"
    echo "PATH=$PATH"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "✅ Success: Build environment configured"
}

build_ollama() {
    echo "🔨 Building Ollama for K80 GPU0 and GPU1..."
    cd "$OLLAMA_DIR" || { echo "❌ Error: Directory change failed"; exit 1; }

    # Build for GPU0
    echo "Building for GPU0 (CUDA_VISIBLE_DEVICES=0)..."
    export CUDA_VISIBLE_DEVICES=0
    echo "Generating Go files for GPU0..."
    go generate ./... || { echo "❌ Error: Go generate failed for GPU0"; exit 1; }
    echo "Building Ollama for GPU0..."
    go build -tags cuda -o ollama-k80-gpu0 . || { echo "❌ Error: Go build failed for GPU0"; exit 1; }
    if [ ! -f ollama-k80-gpu0 ]; then
        echo "❌ Error: GPU0 binary not found"
        exit 1
    fi
    echo "✅ Success: Ollama built for GPU0"

    # Build for GPU1
    echo "Building for GPU1 (CUDA_VISIBLE_DEVICES=1)..."
    export CUDA_VISIBLE_DEVICES=1
    echo "Generating Go files for GPU1..."
    go generate ./... || { echo "❌ Error: Go generate failed for GPU1"; exit 1; }
    echo "Building Ollama for GPU1..."
    go build -tags cuda -o ollama-k80-gpu1 . || { echo "❌ Error: Go build failed for GPU1"; exit 1; }
    if [ ! -f ollama-k80-gpu1 ]; then
        echo "❌ Error: GPU1 binary not found"
        exit 1
    fi
    echo "✅ Success: Ollama built for GPU1"
}

install_ollama() {
    echo "📥 Installing Ollama CUDA versions..."
    echo "Copying GPU0 binary to /usr/local/bin..."
    sudo cp -v "$OLLAMA_DIR/ollama-k80-gpu0" /usr/local/bin/ollama-k80-gpu0 || { echo "❌ Error: Failed to copy GPU0 binary"; exit 1; }
    echo "Copying GPU1 binary to /usr/local/bin..."
    sudo cp -v "$OLLAMA_DIR/ollama-k80-gpu1" /usr/local/bin/ollama-k80-gpu1 || { echo "❌ Error: Failed to copy GPU1 binary"; exit 1; }
    echo "Creating model directories for GPU0 and GPU1..."
    sudo mkdir -pv /usr/share/ollama-k80-gpu0/.ollama /usr/share/ollama-k80-gpu1/.ollama || { echo "❌ Error: Failed to create model directories"; exit 1; }
    echo "Setting ownership for GPU0 and GPU1 dirs..."
    sudo chown -Rv "$USER:$USER" /usr/share/ollama-k80-gpu0 /usr/share/ollama-k80-gpu1 || { echo "❌ Error: Failed to set ownership"; exit 1; }
    echo "Setting permissions for GPU0 and GPU1 dirs..."
    sudo chmod -v 755 /usr/share/ollama-k80-gpu0 /usr/share/ollama-k80-gpu1 || { echo "❌ Error: Failed to set permissions"; exit 1; }
    echo "✅ Success: Ollama installed for both GPUs"
}

create_services() {
    echo "🔧 Creating systemd services for GPU0 and GPU1..."

    # GPU0 Service (port 11436)
    echo "Writing GPU0 service file to /etc/systemd/system/ollama-k80-gpu0.service..."
    sudo tee /etc/systemd/system/ollama-k80-gpu0.service > /dev/null << EOF || { echo "❌ Error: Failed to write GPU0 service file"; exit 1; }
[Unit]
Description=Ollama Service (K80 GPU0)
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama-k80-gpu0 serve
User=$USER
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=/usr/share/ollama-k80-gpu0/.ollama/models"
Environment="OLLAMA_HOST=127.0.0.1:11436"
Environment="CUDA_VISIBLE_DEVICES=0"

[Install]
WantedBy=default.target
EOF

    # GPU1 Service (port 11437)
    echo "Writing GPU1 service file to /etc/systemd/system/ollama-k80-gpu1.service..."
    sudo tee /etc/systemd/system/ollama-k80-gpu1.service > /dev/null << EOF || { echo "❌ Error: Failed to write GPU1 service file"; exit 1; }
[Unit]
Description=Ollama Service (K80 GPU1)
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama-k80-gpu1 serve
User=$USER
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=/usr/share/ollama-k80-gpu1/.ollama/models"
Environment="OLLAMA_HOST=127.0.0.1:11437"
Environment="CUDA_VISIBLE_DEVICES=1"

[Install]
WantedBy=default.target
EOF

    echo "Reloading systemd daemon..."
    sudo systemctl daemon-reload || { echo "❌ Error: Daemon reload failed"; exit 1; }
    echo "Enabling services for GPU0 and GPU1..."
    sudo systemctl enable ollama-k80-gpu0 ollama-k80-gpu1 || { echo "❌ Error: Service enable failed"; exit 1; }
    echo "Starting services for GPU0 and GPU1..."
    sudo systemctl restart ollama-k80-gpu0 ollama-k80-gpu1 || { echo "❌ Error: Service restart failed"; exit 1; }
    echo "✅ Success: Services created and started"
}

setup_docker() {
    echo "🐳 Setting up Docker containers for GPU0 and GPU1..."
    echo "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        echo "❌ Error: Docker not installed. Run ubuntu-essentials-setup.sh first."
        exit 1
    fi
    echo "Stopping any existing containers..."
    sudo docker stop ollama-k80-gpu0 ollama-k80-gpu1 2>/dev/null || echo "⚠️ No containers to stop"
    sudo docker rm ollama-k80-gpu0 ollama-k80-gpu1 2>/dev/null || echo "⚠️ No containers to remove"

    # Docker for GPU0
    echo "Building Docker image for GPU0..."
    sudo docker build -t ollama-k80-gpu0 - << EOF || { echo "❌ Error: Failed to build GPU0 Docker image"; exit 1; }
FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY ./ollama-k80-gpu0 /usr/local/bin/ollama
ENV OLLAMA_HOST=0.0.0.0:11436
ENV CUDA_VISIBLE_DEVICES=0
ENTRYPOINT ["/usr/local/bin/ollama"]
CMD ["serve"]
EOF
    echo "Running Docker container for GPU0..."
    sudo docker run -d --gpus '"device=0"' -v /usr/share/ollama-k80-gpu0/.ollama:/root/.ollama -p 11436:11436 --name ollama-k80-gpu0 ollama-k80-gpu0 || { echo "❌ Error: Failed to run GPU0 container"; exit 1; }

    # Docker for GPU1
    echo "Building Docker image for GPU1..."
    sudo docker build -t ollama-k80-gpu1 - << EOF || { echo "❌ Error: Failed to build GPU1 Docker image"; exit 1; }
FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY ./ollama-k80-gpu1 /usr/local/bin/ollama
ENV OLLAMA_HOST=0.0.0.0:11437
ENV CUDA_VISIBLE_DEVICES=1
ENTRYPOINT ["/usr/local/bin/ollama"]
CMD ["serve"]
EOF
    echo "Running Docker container for GPU1..."
    sudo docker run -d --gpus '"device=1"' -v /usr/share/ollama-k80-gpu1/.ollama:/root/.ollama -p 11437:11437 --name ollama-k80-gpu1 ollama-k80-gpu1 || { echo "❌ Error: Failed to run GPU1 container"; exit 1; }
    echo "✅ Success: Docker containers set up for both GPUs"
}

verify_installation() {
    echo "🔍 Verifying installation..."
    echo "Waiting 2 seconds for services to stabilize..."
    sleep 2
    echo "Checking GPU info with nvidia-smi..."
    nvidia-smi || { echo "❌ Error: nvidia-smi failed"; exit 1; }
    echo "Checking CUDA version with nvcc..."
    nvcc --version || { echo "❌ Error: nvcc failed"; exit 1; }
    echo "Checking GPU0 service status..."
    systemctl status ollama-k80-gpu0 --no-pager || { echo "❌ Error: GPU0 service failed"; exit 1; }
    echo "Testing GPU0 Ollama (port 11436)..."
    OLLAMA_HOST=127.0.0.1:11436 /usr/local/bin/ollama-k80-gpu0 list || { echo "❌ Error: GPU0 test failed"; exit 1; }
    echo "Checking GPU1 service status..."
    systemctl status ollama-k80-gpu1 --no-pager || { echo "❌ Error: GPU1 service failed"; exit 1; }
    echo "Testing GPU1 Ollama (port 11437)..."
    OLLAMA_HOST=127.0.0.1:11437 /usr/local/bin/ollama-k80-gpu1 list || { echo "❌ Error: GPU1 test failed"; exit 1; }
    echo "Checking Docker container for GPU0..."
    sudo docker ps -f name=ollama-k80-gpu0 || { echo "❌ Error: GPU0 Docker container not running"; exit 1; }
    echo "Checking Docker container for GPU1..."
    sudo docker ps -f name=ollama-k80-gpu1 || { echo "❌ Error: GPU1 Docker container not running"; exit 1; }
    echo "✅ Success: Installation verified"
}

main() {
    echo "🔧 Entering main function..."
    check_root
    setup_repository
    setup_build_env
    build_ollama
    install_ollama
    create_services
    setup_docker
    verify_installation
    echo "
✨ Ollama K80 Dual-GPU Build Complete! ✨
- GPU0: ollama-k80-gpu0 (port 11436, CUDA_VISIBLE_DEVICES=0)
- GPU1: ollama-k80-gpu1 (port 11437, CUDA_VISIBLE_DEVICES=1)
- Built with CUDA 11.4 for Tesla K80 dual dies
- Docker containers: ollama-k80-gpu0, ollama-k80-gpu1
Commands:
- ollama-k80-gpu0 list : List models (GPU0 service)
- ollama-k80-gpu1 list : List models (GPU1 service)
- ollama-k80-gpu0 run <model> : Run on GPU0
- ollama-k80-gpu1 run <model> : Run on GPU1
- journalctl -u ollama-k80-gpu0 : GPU0 service logs
- journalctl -u ollama-k80-gpu1 : GPU1 service logs
- docker logs ollama-k80-gpu0 : GPU0 Docker logs
- docker logs ollama-k80-gpu1 : GPU1 Docker logs
Notes:
- Each instance uses one K80 GPU die
- Ports 11436/11437 avoid ROCm (11435)
- Docker uses NVIDIA runtime for CUDA
    "
}

# Trap errors with line numbers
trap 'echo "❌ Script failed at line $LINENO with exit code $?"' ERR

main
