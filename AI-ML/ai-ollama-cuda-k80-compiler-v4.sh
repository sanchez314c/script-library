#!/bin/bash

# Ollama CUDA Compiler Script for Tesla K80 (Kepler Architecture)
# Updated with fixes from working thread
set -e

echo "🚀 Starting Ollama K80-Optimized Build Process..."

# Delete existing Ollama-K80 install
remove_ollama() {
    echo "Removing Ollama-K80 install if found..."
    sudo rm -rf /home/heathen-admin/ollama37
}

# Clone the repository
setup_repository() {
    echo "📦 Cloning Ollama repository..."
    git clone https://github.com/austinksmith/ollama37.git
    cd ollama37
}

# Configure build environment with GCC-10
setup_build_env() {
    echo "⚙️ Configuring build environment..."
    
    # Install GCC-10 and Go if not installed
    echo "Installing required dependencies..."
    sudo apt-get update
    sudo apt-get install -y gcc-10 g++-10 golang-go
    
    # Set compiler paths explicitly
    export CC=/usr/bin/gcc-10
    export CXX=/usr/bin/g++-10
    
    # CUDA environment setup
    export CUDA_PATH=/usr/local/cuda
    export PATH=$CUDA_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
    
    # Force NVCC to use GCC-10
    export CUDAHOSTCXX=/usr/bin/g++-10
    
    # K80-specific compute capabilities
    export CUDA_ARCHITECTURES="35;37;50;52"
    export CUDA_COMPUTE_MAJOR_MIN=3
    export CUDA_COMPUTE_MINOR_MIN=5
    
    # Compiler flags for CUDA
    export NVCC_FLAGS="--allow-unsupported-compiler"
    export CGO_CFLAGS="-I$CUDA_PATH/include -I/opt/rocm/include -I/opt/rocm/include/hip"
    export CGO_LDFLAGS="-L$CUDA_PATH/lib64 -L/opt/rocm/lib"
    export CUDACXX="$CUDA_PATH/bin/nvcc --allow-unsupported-compiler"
}

# Force GCC compiler version compatibility
set_CUDA_GCC_use_ver() {
    echo "🔧 Adjusting GCC compiler versioning..."
    sudo sed -i 's/__GNUC__ > 12/__GNUC__ > 13/' /usr/local/cuda/include/crt/host_config.h
}
    
# Modify GPU compute capabilities
adjust_compute_capabilities() {
    echo "🔧 Adjusting compute capabilities for K80..."
    sed -i 's/var CudaComputeMin = \[2\]C.int{3, 5}/var CudaComputeMin = \[2\]C.int{3, 0}/' ./gpu/gpu.go
    
    # Check if Go is installed
    if ! command -v go &> /dev/null; then
        echo "Go is not installed. Installing Go..."
        sudo apt-get install -y golang-go
    fi
    
    # Clean any previous builds
    echo "Cleaning Go cache..."
    go clean -cache || echo "Go cache cleaning skipped"
}
    
# Build Ollama
build_ollama() {
    echo "🔨 Building Ollama with Kepler Compute..."
    # Create a build script to maintain environment variables
    cat > build_with_gcc10.sh << 'EOF'
#!/bin/bash
# Set compiler paths
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
# Set CUDA paths and flags
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
# Force NVCC to use GCC 10
export CUDAHOSTCXX=/usr/bin/g++-10
# Set environment variables for Go build
export CGO_CFLAGS="-I$CUDA_PATH/include -I/opt/rocm/include -I/opt/rocm/include/hip"
export CGO_LDFLAGS="-L$CUDA_PATH/lib64 -L/opt/rocm/lib"
export CUDA_ARCHITECTURES="35;37;50;52"
export CUDA_COMPUTE_MAJOR_MIN=3
export CUDA_COMPUTE_MINOR_MIN=5
export NVCC_FLAGS="--allow-unsupported-compiler"
export CUDACXX="$CUDA_PATH/bin/nvcc --allow-unsupported-compiler"

echo "Building with GCC 10 and CUDA 11.4..."
go generate ./...
go build .
echo "Build completed!"
EOF
    chmod +x build_with_gcc10.sh
    
    # Run the build script to ensure environment is preserved
    ./build_with_gcc10.sh
}

# Install Ollama
install_ollama() {
    echo "📥 Installing Ollama..."
    
    # Create ollama user and group
    sudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama || true
    sudo usermod -a -G ollama $(whoami)
    
    # Setup directories and permissions
    sudo mkdir -p /usr/share/ollama/.ollama
    sudo chown -R ollama:ollama /usr/share/ollama
    sudo chmod 755 /usr/share/ollama
    
    # Install binary
    OLLAMA_DIR=$(pwd)
    sudo cp $OLLAMA_DIR/ollama /usr/local/bin/ollama-k80
    sudo ln -sf /usr/local/bin/ollama-k80 /usr/local/bin/ollama
}

# Create service file
create_service() {
    echo "🔧 Creating systemd service..."
    
    sudo tee /etc/systemd/system/ollama.service > /dev/null << EOF
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama-k80 serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/cuda/bin:${PATH}"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
Environment="CUDA_VISIBLE_DEVICES=0,1"
Environment="CUDA_ARCHITECTURES=35;37;50;52"

[Install]
WantedBy=default.target
EOF

    # Enable and start service
    sudo systemctl daemon-reload
    sudo systemctl enable ollama
    sudo systemctl start ollama
}

# Verify installation
verify_installation() {
    echo "🔍 Verifying installation..."
    
    echo "Service status:"
    sudo systemctl status ollama
    
    echo "Testing Ollama:"
    ollama-k80 list
    
    echo "GPU Detection:"
    nvidia-smi
}

# Main installation process
main() {
    remove_ollama
    setup_repository
    setup_build_env
    set_CUDA_GCC_use_ver
    adjust_compute_capabilities
    build_ollama
    install_ollama
    create_service
    verify_installation
    
    echo "
✨ Ollama K80 Build Complete! ✨

Installation Summary:
- Built with CUDA support for Tesla K80 using GCC-10
- Compute capabilities: 3.5, 3.7, 5.0, 5.2
- Binary installed as: ollama-k80
- Service configured and running

Usage:
1. List models:    ollama-k80 list
2. Pull a model:   ollama-k80 pull llama2
3. Run a model:    ollama-k80 run llama2
4. View logs:      journalctl -u ollama -f

Note: The service is running as user 'ollama' with optimized K80 settings.
"
}

# Start installation
main
