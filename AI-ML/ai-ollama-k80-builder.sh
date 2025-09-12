#!/bin/bash

# Ollama CUDA Build Script for Tesla K80 (Dual Die)
# Using austinksmith/ollama37 repository with GCC 12
# Date: March 2, 2025

set -e  # Exit on any error

echo "🚀 Starting Ollama K80-Optimized Build Process for Dual GPUs..."

# Global variables
OLLAMA_DIR="/home/$USER/ollama37"
BINDIR="/usr/local/bin"
CUDA_PATH="/usr/local/cuda"
MODEL_STORAGE="/usr/share/ollama-k80/.ollama/models"

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "❌ Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    echo "✅ Running as root"
}

check_dependencies() {
    echo "🔍 Checking and installing dependencies..."
    apt update
    apt install -y gcc-12 g++-12 make cmake pkg-config libclblast-dev libopenblas-dev golang-go build-essential ninja-build
    echo "✅ Dependencies installed"
}

check_cuda() {
    echo "🔍 Checking CUDA installation..."
    if [ ! -f "$CUDA_PATH/bin/nvcc" ]; then
        echo "❌ Error: CUDA not found at $CUDA_PATH"
        exit 1
    fi
    
    # Fix gcc version compatibility if needed
    if grep -q "__GNUC__ > 12" "$CUDA_PATH/include/crt/host_config.h"; then
        echo "🔧 Adjusting GCC version compatibility in CUDA headers..."
        sudo sed -i 's/__GNUC__ > 12/__GNUC__ > 13/' "$CUDA_PATH/include/crt/host_config.h"
    fi
    
    echo "✅ CUDA installation verified"
}

setup_repository() {
    echo "📦 Cloning austinksmith/ollama37 repository..."
    rm -rf "$OLLAMA_DIR" || echo "No existing directory to remove"
    git clone https://github.com/austinksmith/ollama37.git "$OLLAMA_DIR"
    cd "$OLLAMA_DIR"
    echo "✅ Repository cloned"
}

adjust_compute_capabilities() {
    echo "🔧 Adjusting compute capabilities for K80..."
    cd "$OLLAMA_DIR"
    # Update GPU detection to allow K80's compute capability (3.5/3.7)
    sed -i 's/var CudaComputeMin = \[2\]C.int{3, 5}/var CudaComputeMin = \[2\]C.int{3, 0}/' ./gpu/gpu.go
    echo "✅ Compute capabilities adjusted"
}

setup_build_env() {
    echo "⚙️ Setting up build environment..."
    
    # Set compiler to GCC-12
    export CC=/usr/bin/gcc-12
    export CXX=/usr/bin/g++-12
    
    # CUDA environment
    export CUDA_PATH="$CUDA_PATH"
    export PATH="$CUDA_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
    
    # Force NVCC to use GCC-12
    export CUDAHOSTCXX=/usr/bin/g++-12
    
    # K80-specific compute capabilities
    export CUDA_ARCHITECTURES="35;37;50;52"
    export CUDA_COMPUTE_MAJOR_MIN=3
    export CUDA_COMPUTE_MINOR_MIN=5
    
    # Compiler flags for CUDA
    # Set number of cores for NVCC
NUM_CORES=$(nproc)
export NVCC_FLAGS="--allow-unsupported-compiler -j$NUM_CORES"
    export CGO_CFLAGS="-I$CUDA_PATH/include"
    export CGO_LDFLAGS="-L$CUDA_PATH/lib64"
    export CUDACXX="$CUDA_PATH/bin/nvcc --allow-unsupported-compiler"
    
    echo "✅ Build environment configured"
}

build_ollama() {
    echo "🔨 Building Ollama with K80 support..."
    cd "$OLLAMA_DIR"
    
    # Set max number of processes for parallel compilation
    NUM_CORES=$(nproc)
    export GOMAXPROCS=$NUM_CORES
    echo "🔄 Setting up parallel build with $NUM_CORES cores"
    
    # Clean any previous builds
    go clean -cache
    
    # Create build script to preserve environment
    cat > build.sh << EOF
#!/bin/bash
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDA_PATH="$CUDA_PATH"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
export CUDAHOSTCXX=/usr/bin/g++-12
export CUDA_ARCHITECTURES="35;37;50;52"
export CUDA_COMPUTE_MAJOR_MIN=3
export CUDA_COMPUTE_MINOR_MIN=5
export NVCC_FLAGS="--allow-unsupported-compiler"
# Set parallel compilation flags
NUM_CORES=$(nproc)
export CGO_CFLAGS="-I$CUDA_PATH/include -j$NUM_CORES"
export CGO_LDFLAGS="-L$CUDA_PATH/lib64"
export CUDACXX="$CUDA_PATH/bin/nvcc --allow-unsupported-compiler"

echo "Generating files..."
go generate ./...

echo "Building Ollama using all CPU cores..."
# Get number of CPU cores
CORES=$(nproc)
echo "Using $CORES CPU cores for compilation"
go build -p $CORES -o ollama-k80-gpu0
cp ollama-k80-gpu0 ollama-k80-gpu1
EOF
    
    chmod +x build.sh
    ./build.sh
    
    # Check if build was successful
    if [ ! -f "ollama-k80-gpu0" ] || [ ! -f "ollama-k80-gpu1" ]; then
        echo "❌ Build failed. Check for errors above."
        exit 1
    fi
    
    echo "✅ Ollama built successfully for both GPU dies"
}

install_ollama() {
    echo "📥 Installing Ollama..."
    
    # Create ollama user if it doesn't exist
    id -u ollama &>/dev/null || sudo useradd -r -s /bin/false -m -d /usr/share/ollama ollama
    sudo usermod -a -G ollama $(whoami)
    
    # Create model directories
    sudo mkdir -p "$MODEL_STORAGE"
    sudo chown -R ollama:ollama /usr/share/ollama-k80
    sudo chmod 755 /usr/share/ollama-k80
    
    # Install binaries
    sudo cp "$OLLAMA_DIR/ollama-k80-gpu0" "$BINDIR/ollama-k80-gpu0" 
    sudo cp "$OLLAMA_DIR/ollama-k80-gpu1" "$BINDIR/ollama-k80-gpu1"
    sudo ln -sf "$BINDIR/ollama-k80-gpu0" "$BINDIR/ollama"
    
    echo "✅ Ollama installed"
}

create_services() {
    echo "🔧 Creating systemd services for dual GPUs..."
    
    # Service for GPU 0
    sudo tee /etc/systemd/system/ollama-k80-gpu0.service > /dev/null << EOF
[Unit]
Description=Ollama Service (K80 GPU 0)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=$BINDIR/ollama-k80-gpu0 serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=$MODEL_STORAGE"
Environment="OLLAMA_HOST=127.0.0.1:11436"
Environment="PATH=$CUDA_PATH/bin:${PATH}"
Environment="LD_LIBRARY_PATH=$CUDA_PATH/lib64:${LD_LIBRARY_PATH}"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="CUDA_ARCHITECTURES=35;37;50;52"

[Install]
WantedBy=default.target
EOF

    # Service for GPU 1
    sudo tee /etc/systemd/system/ollama-k80-gpu1.service > /dev/null << EOF
[Unit]
Description=Ollama Service (K80 GPU 1)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=$BINDIR/ollama-k80-gpu1 serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=$MODEL_STORAGE"
Environment="OLLAMA_HOST=127.0.0.1:11437"
Environment="PATH=$CUDA_PATH/bin:${PATH}"
Environment="LD_LIBRARY_PATH=$CUDA_PATH/lib64:${LD_LIBRARY_PATH}"
Environment="CUDA_VISIBLE_DEVICES=1"
Environment="CUDA_ARCHITECTURES=35;37;50;52"

[Install]
WantedBy=default.target
EOF

    # Reload systemd and enable services
    sudo systemctl daemon-reload
    sudo systemctl enable ollama-k80-gpu0 ollama-k80-gpu1
    sudo systemctl restart ollama-k80-gpu0 ollama-k80-gpu1
    
    echo "✅ Services created and started"
}

verify_installation() {
    echo "🔍 Verifying installation..."
    
    # Check if GPU is detected
    nvidia-smi || { echo "❌ Error: nvidia-smi failed"; exit 1; }
    
    # Check service status
    systemctl status ollama-k80-gpu0 --no-pager
    systemctl status ollama-k80-gpu1 --no-pager
    
    # Test Ollama
    OLLAMA_HOST=127.0.0.1:11436 "$BINDIR/ollama-k80-gpu0" list
    OLLAMA_HOST=127.0.0.1:11437 "$BINDIR/ollama-k80-gpu1" list
    
    echo "✅ Installation verified"
}

main() {
    check_root
    check_dependencies
    check_cuda
    setup_repository
    adjust_compute_capabilities
    setup_build_env
    build_ollama
    install_ollama
    create_services
    verify_installation
    
    echo "
✨ Ollama K80 Dual-GPU Build Complete! ✨

Installation Summary:
- Using austinksmith/ollama37 with GCC-12
- Compute capabilities: 3.5, 3.7, 5.0, 5.2
- GPU 0: ollama-k80-gpu0 (port 11436)
- GPU 1: ollama-k80-gpu1 (port 11437)
- Models stored in $MODEL_STORAGE
- Primary 'ollama' command linked to GPU 0

Usage:
1. List models on GPU 0: ollama-k80-gpu0 list
2. List models on GPU 1: OLLAMA_HOST=127.0.0.1:11437 ollama-k80-gpu1 list
3. Run a model on GPU 0: ollama-k80-gpu0 run llama2
4. Run a model on GPU 1: OLLAMA_HOST=127.0.0.1:11437 ollama-k80-gpu1 run llama2
5. View logs: journalctl -u ollama-k80-gpu0 -f

Note: Both GPU dies share the same model storage for efficiency.
"
}

# Trap errors with line numbers
trap 'echo "❌ Script failed at line $LINENO with exit code $?"' ERR

main
