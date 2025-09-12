#!/bin/bash

# CUDA Installation and Configuration Script
# Created by Cortana for Jason
# Designed to work with existing NVIDIA 470 driver
set -e

echo "🚀 Starting CUDA Installation and Configuration..."

# Check for existing NVIDIA driver
check_nvidia_driver() {
    echo "🔍 Checking NVIDIA driver..."
    if ! nvidia-smi &>/dev/null; then
        echo "❌ NVIDIA driver not found or not functioning. Please install driver 470 first."
        exit 1
    fi
    
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    echo "✅ Found NVIDIA driver version: $DRIVER_VERSION"
}

# Blacklist Nouveau driver
configure_nouveau_blacklist() {
    echo "🛠 Configuring Nouveau blacklist..."
    
    sudo bash -c 'cat > /etc/modprobe.d/blacklist-nouveau.conf << EOF
blacklist nouveau
options nouveau modeset=0
EOF'
    
    sudo update-initramfs -u
}

# Install CUDA dependencies
install_cuda_dependencies() {
    echo "📦 Installing CUDA dependencies..."
    
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        dkms \
        gcc-10 \
        g++-10 \
        pkg-config \
        linux-headers-$(uname -r)
}

# Download and install CUDA toolkit
install_cuda_toolkit() {
    echo "📥 Downloading CUDA 11.4 toolkit..."
    
    CUDA_RUN="cuda_11.4.0_470.42.01_linux.run"
    if [ ! -f "$CUDA_RUN" ]; then
        wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/$CUDA_RUN
    fi
    
    echo "🛠 Installing CUDA toolkit (excluding driver)..."
    sudo sh $CUDA_RUN --toolkit --samples --silent --override --no-opengl-libs --no-man-page
}

# Configure CUDA environment
configure_cuda_environment() {
    echo "⚙️ Configuring CUDA environment..."
    
    # Add CUDA paths to bashrc if not already present
    if ! grep -q "CUDA-11.4" ~/.bashrc; then
        cat << 'EOF' >> ~/.bashrc

# CUDA Environment Variables
export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-11.4
export CUDA_PATH=/usr/local/cuda-11.4
export CUDA_VERSION_OVERRIDE=11.4
EOF
    fi

    # Set up current session
    export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export CUDA_HOME=/usr/local/cuda-11.4
    export CUDA_PATH=/usr/local/cuda-11.4
    export CUDA_VERSION_OVERRIDE=11.4
}

# Install additional CUDA tools and libraries
install_cuda_extras() {
    echo "🛠 Installing additional CUDA tools and libraries..."
    
    sudo apt-get install -y \
        cuda-command-line-tools-11-4 \
        cuda-libraries-dev-11-4 \
        cuda-minimal-build-11-4 \
        cuda-cudart-dev-11-4 \
        cuda-nvcc-11-4 \
        cuda-nvprof-11-4 \
        cuda-cupti-11-4 \
        cuda-cuobjdump-11-4 \
        cuda-memcheck-11-4 \
        cuda-sanitizer-11-4
}

# Create CUDA verification script
create_cuda_check_script() {
    echo "📝 Creating CUDA verification script..."
    
    cat << 'EOF' | sudo tee /usr/local/bin/check-cuda
#!/bin/bash
echo "CUDA Configuration Check"
echo "======================="
echo
echo "NVIDIA Driver Information:"
nvidia-smi
echo
echo "CUDA Version Information:"
nvcc --version
echo
echo "CUDA Library Path:"
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep cuda
echo
echo "CUDA Binary Path:"
echo $PATH | tr ':' '\n' | grep cuda
echo
echo "Installed CUDA Components:"
ls -l /usr/local/cuda/
echo
echo "Testing CUDA Compiler:"
if [ -d "/usr/local/cuda/samples/1_Utilities/deviceQuery" ]; then
    cd /usr/local/cuda/samples/1_Utilities/deviceQuery
    make
    ./deviceQuery
else
    echo "CUDA samples not found"
fi
EOF

    sudo chmod +x /usr/local/bin/check-cuda
}

# Add CUDA-related aliases
add_cuda_aliases() {
    echo "📝 Adding CUDA aliases..."
    
    cat << 'EOF' >> ~/.bashrc

# CUDA Aliases and Functions
alias cuda-check='/usr/local/bin/check-cuda'
alias cuda-version='nvcc --version'
alias cuda-smi='watch -n1 nvidia-smi'
alias cuda-clean='make clean && make distclean'
alias cuda-samples='cd /usr/local/cuda/samples'

# CUDA compiler wrapper with compute capability flags
cuda-compile() {
    nvcc -arch=sm_35 -gencode=arch=compute_35,code=sm_35 \
         -gencode=arch=compute_37,code=sm_37 \
         -gencode=arch=compute_50,code=sm_50 \
         -gencode=arch=compute_52,code=sm_52 \
         "$@"
}
EOF
}

# Main installation process
main() {
    check_nvidia_driver
    configure_nouveau_blacklist
    install_cuda_dependencies
    install_cuda_toolkit
    configure_cuda_environment
    install_cuda_extras
    create_cuda_check_script
    add_cuda_aliases
    
    echo "
✨ CUDA Installation Complete! ✨

Installation Summary:
- CUDA 11.4 toolkit installed
- Environment configured for NVIDIA driver $DRIVER_VERSION
- CUDA samples installed
- Additional tools and libraries added
- Verification scripts created

Available Commands:
1. cuda-check    : Run comprehensive CUDA verification
2. cuda-version  : Show CUDA compiler version
3. cuda-smi      : Monitor GPU status (nvidia-smi)
4. cuda-samples  : Navigate to CUDA samples directory
5. cuda-compile  : Compile with K80 compute capabilities

Environment Variables Set:
- CUDA_HOME=/usr/local/cuda-11.4
- PATH includes CUDA binary directory
- LD_LIBRARY_PATH includes CUDA libraries

Note: Please log out and back in for all changes to take effect.

To verify installation:
1. Run 'cuda-check' for full system verification
2. Try compiling a CUDA sample:
   cd /usr/local/cuda/samples/1_Utilities/deviceQuery
   make
   ./deviceQuery
"

    # Final verification
    echo "🔍 Running final CUDA verification..."
    source ~/.bashrc
    nvcc --version
    nvidia-smi
}

# Start installation
main