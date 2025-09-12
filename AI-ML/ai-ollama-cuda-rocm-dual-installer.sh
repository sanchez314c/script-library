#!/bin/bash

# Unified Ollama Build Script for Both CUDA and ROCm Support
# Created by Cortana for Jason
set -e

echo "🚀 Starting Unified Ollama Build for CUDA + ROCm..."

# Install all dependencies
install_dependencies() {
    echo "📦 Installing dependencies..."
    
    # Common dependencies
    sudo apt-get update
    sudo apt-get install -y \
        git \
        golang-1.21 \
        build-essential \
        cmake \
        curl \
        pkg-config \
        python3 \
        python3-pip
        
    # CUDA dependencies
    if [ -x "$(command -v nvidia-smi)" ]; then
        echo "Installing CUDA dependencies..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        sudo apt-get update
        sudo apt-get install -y \
            nvidia-cuda-toolkit \
            nvidia-cuda-toolkit-gcc
    fi
    
    # ROCm dependencies
    if [ -x "$(command -v rocminfo)" ] || [ "$FORCE_ROCM" = "true" ]; then
        echo "Installing ROCm dependencies..."
        sudo mkdir --parents --mode=0755 /etc/apt/keyrings
        sudo wget https://repo.radeon.com/rocm/rocm.gpg.key -O /etc/apt/keyrings/rocm.gpg
        
        echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/debian $(lsb_release -cs) main" \
            | sudo tee /etc/apt/sources.list.d/rocm.list
            
        echo -e 'Package: *\nPin: release o=AMD\nPin-Priority: 600' \
            | sudo tee /etc/apt/preferences.d/rocm-pin-600
            
        sudo apt-get update
        sudo apt-get install -y \
            rocm-dev \
            rocm-libs \
            rocm-utils \
            rocm-hip-sdk \
            hipblas \
            rocblas \
            miopen-hip
            
        sudo usermod -a -G video,render $USER
    fi
}

# Set up build environment
setup_environment() {
    echo "🔧 Setting up build environment..."
    
    # Go environment
    export PATH="/usr/lib/go-1.21/bin:$PATH"
    export GOPATH=$HOME/go
    export PATH=$PATH:$GOPATH/bin
    
    # CUDA environment
    if [ -x "$(command -v nvidia-smi)" ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        export OLLAMA_CUSTOM_CUDA_ARCH="35;37;50;52;60;61;70;75;80;86"
        export CGO_CFLAGS="$CGO_CFLAGS -I/usr/local/cuda/include"
        export CGO_LDFLAGS="$CGO_LDFLAGS -L/usr/local/cuda/lib64"
    fi
    
    # ROCm environment
    if [ -x "$(command -v rocminfo)" ] || [ "$FORCE_ROCM" = "true" ]; then
        export PATH="/opt/rocm/bin:$PATH"
        export HIP_PLATFORM="amd"
        export HIPCC_COMPILE_FLAGS_APPEND="--amdgpu-target=`rocminfo | grep -m1 gfx | awk '{print $2}'`"
        export HSA_OVERRIDE_GFX_VERSION=`rocminfo | grep -m1 gfx | awk '{print $2}'`
        export CGO_CFLAGS="$CGO_CFLAGS -I/opt/rocm/include"
        export CGO_LDFLAGS="$CGO_LDFLAGS -L/opt/rocm/lib -L/opt/rocm/lib64"
    fi
}

# Build Ollama with both CUDA and ROCm support
build_ollama() {
    echo "🔨 Building Ollama with unified GPU support..."
    
    BUILD_DIR=$HOME/ollama-build
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    
    echo "📥 Cloning Ollama repository..."
    git clone https://github.com/ollama/ollama.git
    cd ollama
    
    # Create custom build tags file
    cat << 'EOF' > custom_build.go
//go:build cuda && rocm
package main

/*
#cgo CFLAGS: ${CFLAGS}
#cgo LDFLAGS: ${LDFLAGS}
*/
import "C"
EOF

    # Generate and build
    go generate ./...
    
    # Build with both CUDA and ROCm tags
    go build -tags "cuda rocm" -o ollama ./cmd/ollama
}

# Install and configure
install_and_configure() {
    echo "📦 Installing Ollama..."
    sudo mv ollama /usr/local/bin/
    sudo chown root:root /usr/local/bin/ollama
    sudo chmod 755 /usr/local/bin/ollama
    
    # Create service with both GPU support
    cat << 'EOF' | sudo tee /etc/systemd/system/ollama.service
[Unit]
Description=Ollama Service
After=network-online.target
Wants=network-online.target

[Service]
# CUDA Environment
Environment="PATH=/usr/local/cuda/bin:/opt/rocm/bin:$PATH"
Environment="OLLAMA_CUSTOM_CUDA_ARCH=35;37;50;52;60;61;70;75;80;86"

# ROCm Environment
Environment="HIP_PLATFORM=amd"
Environment="HSA_OVERRIDE_GFX_VERSION=$(rocminfo | grep -m1 gfx | awk '{print $2}' 2>/dev/null || echo '')"

ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=3
User=$USER
Group=$USER
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

    # Create unified configuration
    mkdir -p ~/.ollama
    cat << EOF > ~/.ollama/config
{
  "gpu": true,
  "cuda": $(if [ -x "$(command -v nvidia-smi)" ]; then echo "true"; else echo "false"; fi),
  "rocm": $(if [ -x "$(command -v rocminfo)" ]; then echo "true"; else echo "false"; fi),
  "cuda_arch": ["35", "37", "50", "52", "60", "61", "70", "75", "80", "86"],
  "hip_device": "$(rocminfo | grep -m1 gfx | awk '{print $2}' 2>/dev/null || echo '')"
}
EOF
}

# Create verification script
create_verification_script() {
    cat << 'EOF' | sudo tee /usr/local/bin/check-ollama-gpus
#!/bin/bash
echo "Ollama Unified GPU Status Check"
echo "=============================="
echo
echo "NVIDIA GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "No NVIDIA GPU detected"
fi
echo
echo "ROCm GPU Information:"
if command -v rocminfo &> /dev/null; then
    rocminfo | grep -A 5 "GPU Agent"
else
    echo "No ROCm GPU detected"
fi
echo
echo "Ollama Build Information:"
ollama --version
echo
echo "Ollama Process Information:"
ps aux | grep ollama
echo
echo "Ollama Service Status:"
systemctl status ollama
echo
echo "Ollama GPU Configuration:"
cat ~/.ollama/config
echo
echo "Testing Ollama with GPU..."
ollama run llama2 "What GPU are you using?" 2>&1 | grep -i "cuda\|gpu\|rocm\|hip"
EOF

    sudo chmod +x /usr/local/bin/check-ollama-gpus
}

# Add convenience aliases
add_aliases() {
    cat << 'EOF' >> ~/.bashrc

# Ollama Unified GPU Aliases
alias ollama-status='systemctl status ollama'
alias ollama-restart='sudo systemctl restart ollama'
alias ollama-logs='journalctl -u ollama -f'
alias ollama-gpu-check='check-ollama-gpus'
alias gpu-monitor='watch -n1 "nvidia-smi; echo ""; echo "ROCm Status:"; rocm-smi"'
EOF
}

# Main installation
main() {
    install_dependencies
    setup_environment
    build_ollama
    install_and_configure
    create_verification_script
    add_aliases
    
    # Start service
    sudo systemctl daemon-reload
    sudo systemctl enable ollama
    sudo systemctl start ollama
    
    echo "
✨ Ollama Unified GPU Build Complete! ✨

Installation Summary:
- Built with both CUDA and ROCm support
- CUDA compute capabilities: 3.5-8.6
- ROCm/HIP support enabled
- Unified GPU configuration
- Auto-detection of available GPUs

Available Commands:
1. ollama-status    : Check Ollama service status
2. ollama-restart   : Restart Ollama service
3. ollama-logs      : View Ollama logs
4. ollama-gpu-check : Verify all GPU configurations
5. gpu-monitor      : Monitor all GPU metrics

To verify the installation:
1. Run 'ollama-gpu-check' to see all GPU configurations
2. Pull a model: 'ollama pull llama2'
3. Test with: 'ollama run llama2 \"Hello, what GPUs do you see?\"'

Note: While Ollama is built with support for both GPU types,
it will typically use the first available GPU it detects.

Important: You may need to log out and back in for group changes to take effect.
"

    # Final verification
    sleep 5
    check-ollama-gpus
}

# Start installation
main
