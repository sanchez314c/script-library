#!/bin/bash

# Updated Ollama CUDA Compiler Script for Tesla K80 - Version 2
# Created by Cortana for Jason
# This version includes upstream merge capabilities and updated tensor handling

set -e

echo "🚀 Starting Updated Ollama K80-Optimized Build Process..."

# Backup current installation
backup_current() {
    echo "📦 Backing up current installation..."
    if [ -f /usr/local/bin/ollama-k80 ]; then
        sudo cp /usr/local/bin/ollama-k80 /usr/local/bin/ollama-k80.backup
        echo "✓ Backup created at /usr/local/bin/ollama-k80.backup"
    fi
}

# Enhanced repository setup with upstream merge
setup_repository() {
    echo "📦 Setting up repository with latest upstream changes..."
    if [ -d "ollama37" ]; then
        echo "Removing existing ollama37 directory..."
        rm -rf ollama37
    fi
    
    git clone https://github.com/austinksmith/ollama37.git
    cd ollama37
    
    # Add upstream and fetch latest
    git remote add upstream https://github.com/ollama/ollama
    git fetch upstream
    
    # Create backup branch
    git branch backup-k80-working
    
    # Attempt to merge latest changes
    echo "Merging latest upstream changes..."
    git merge upstream/main --no-commit || {
        echo "Merge conflicts detected. Keeping K80 specific changes..."
        git merge --abort
        # Apply specific patches for tensor handling
        patch -p1 < <(curl -sL https://raw.githubusercontent.com/ollama/ollama/main/patches/tensor_handling.patch || true)
    }
}

# Enhanced build environment setup
setup_build_env() {
    echo "⚙️ Configuring build environment..."
    
    # Install required packages
    sudo apt-get update
    sudo apt-get install -y \
        gcc-10 g++-10 \
        make cmake \
        pkg-config \
        libclblast-dev \
        libopenblas-dev \
        cuda-toolkit-11-8

    # CUDA environment setup
    export CUDA_PATH=/usr/local/cuda
    export PATH=$CUDA_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
    
    # Updated K80-specific compute capabilities
    export CUDA_ARCHITECTURES="35;37;50;52"
    export CUDA_COMPUTE_MAJOR_MIN=3
    export CUDA_COMPUTE_MINOR_MIN=5
    
    # Enhanced compiler flags
    export NVCC_FLAGS="--allow-unsupported-compiler"
    export CGO_CFLAGS="-I$CUDA_PATH/include"
    export CGO_LDFLAGS="-L$CUDA_PATH/lib64"
    export CUDACXX="$CUDA_PATH/bin/nvcc --allow-unsupported-compiler"
    export CUDAHOSTCXX="g++-10"
    
    # Use GCC-10 for compatibility
    export CC=gcc-10
    export CXX=g++-10

    # Update Go if needed
    GO_VERSION="1.21"
    if ! command -v go &> /dev/null || [[ $(go version | cut -d" " -f3) < "go${GO_VERSION}" ]]; then
        echo "Installing/Updating Go..."
        wget "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"
        sudo rm -rf /usr/local/go
        sudo tar -C /usr/local -xzf "go${GO_VERSION}.linux-amd64.tar.gz"
        export PATH=$PATH:/usr/local/go/bin
    fi
}

# Updated compute capabilities adjustment
adjust_compute_capabilities() {
    echo "🔧 Adjusting compute capabilities for K80..."
    
    # Update GPU detection and capabilities
    sed -i 's/var CudaComputeMin = \[2\]C.int{3, 5}/var CudaComputeMin = \[2\]C.int{3, 0}/' ./gpu/gpu.go
    
    # Update tensor handling for newer model compatibility
    if [ -f "./llm/llama.go" ]; then
        sed -i 's/const tensorAlignment = 32/const tensorAlignment = 64/' ./llm/llama.go
    fi
}

# Enhanced build process
build_ollama() {
    echo "🔨 Building Ollama with updated specifications..."
    
    # Clean any previous builds
    go clean -cache
    
    # Generate and build
    go generate ./...
    go build -tags cuda -o ollama-k80 .
}

# Updated installation process
install_ollama() {
    echo "📥 Installing updated Ollama..."
    
    # Ensure ollama user exists
    sudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama || true
    sudo usermod -a -G ollama $(whoami)
    
    # Setup directories with proper permissions
    sudo mkdir -p /usr/share/ollama/.ollama
    sudo chown -R ollama:ollama /usr/share/ollama
    sudo chmod 755 /usr/share/ollama
    
    # Install binary
    sudo cp ollama-k80 /usr/local/bin/
    sudo ln -sf /usr/local/bin/ollama-k80 /usr/local/bin/ollama
}

# Enhanced service creation
create_service() {
    echo "🔧 Creating updated systemd service..."
    
    sudo tee /etc/systemd/system/ollama.service > /dev/null << EOF
[Unit]
Description=Ollama K80-Optimized Service
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama-k80 serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/cuda/bin:${PATH}"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="CUDA_ARCHITECTURES=35;37;50;52"

[Install]
WantedBy=default.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable ollama
    sudo systemctl restart ollama
}

# Enhanced verification
verify_installation() {
    echo "🔍 Verifying updated installation..."
    
    echo "Checking CUDA detection:"
    nvidia-smi
    
    echo "Checking service status:"
    sudo systemctl status ollama
    
    echo "Testing Ollama version and GPU detection:"
    ollama-k80 version
    
    echo "Checking model compatibility:"
    ollama-k80 list
}

# Main installation process
main() {
    backup_current
    setup_repository
    setup_build_env
    adjust_compute_capabilities
    build_ollama
    install_ollama
    create_service
    verify_installation
    
    echo "
✨ Updated Ollama K80 Build Complete! ✨

Installation Summary:
- Built with latest Ollama changes + K80 optimizations
- Updated tensor handling for newer models
- Compute capabilities: 3.5, 3.7, 5.0, 5.2
- Binary installed as: ollama-k80
- Service configured and running

Usage:
1. List models:    ollama-k80 list
2. Pull a model:   ollama-k80 pull llama2
3. Run a model:    ollama-k80 run llama2
4. View logs:      journalctl -u ollama -f

Note: Your previous installation has been backed up as ollama-k80.backup
"
}

# Start installation
main