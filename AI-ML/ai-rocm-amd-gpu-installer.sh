#!/bin/bash
# ROCm Installation Script for Ubuntu 24.04
# Created by Cortana for Jason
# Version 2.1 with fixed repository configuration
set -e
echo "🚀 Starting ROCm Installation..."

# Install ROCm dependencies
install_rocm() {
    echo "📦 Installing ROCm dependencies..."
    
    # Add ROCm repository with proper authentication
    sudo mkdir --parents --mode=0755 /etc/apt/keyrings
    
    # Download and import the key properly
    wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
    
    # Use noble (Ubuntu 24.04) repository
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.3.3 noble main" \
        | sudo tee /etc/apt/sources.list.d/rocm.list
    
    # Import the GPG key
    wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb
    sudo apt install ./amdgpu-install_6.3.60303-1_all.deb
    
    # Set repository priority
    echo -e 'Package: *\nPin: release o=AMD\nPin-Priority: 600' \
        | sudo tee /etc/apt/preferences.d/rocm-pin-600
    
    # Update package lists
    sudo apt update
    
    # Install ROCm stack
    sudo apt install -y \
        rocm-dkms \
        rocm-dev \
        rocm-libs \
        rocm-utils \
        rocm-hip-sdk \
        rocm-opencl-sdk \
        rocminfo \
        miopen-hip \
        hipblas \
        rocblas
        
    # Add user to video group for GPU access
    sudo usermod -a -G video,render $USER
}

# Set up environment
setup_environment() {
    echo "🔧 Setting up ROCm environment..."
    
    # Set up ROCm environment variables
    cat << 'EOF' >> ~/.bashrc
# ROCm Environment Variables
export PATH="/opt/rocm/bin:$PATH"
export HIP_PLATFORM="amd"
export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
EOF
}

# Add ROCm monitoring aliases
add_aliases() {
    echo "🔧 Adding ROCm convenience aliases..."
    cat << 'EOF' >> ~/.bashrc
# ROCm Aliases
alias rocm-info='rocminfo'
alias rocm-smi='rocm-smi'
alias rocm-monitor='watch -n1 rocm-smi'
EOF
}

# Main installation process
main() {
    install_rocm
    setup_environment
    add_aliases
    
    echo "
✨ ROCm Installation Complete! ✨
Installation Summary:
- ROCm stack installed and configured
- Environment variables set
- User added to video/render groups

Available Commands:
1. rocm-info    : Display ROCm device information
2. rocm-smi     : Monitor GPU metrics
3. rocm-monitor : Continuously monitor GPU status

Important Notes:
1. Please reboot your system for all changes to take effect
2. After reboot, run 'rocminfo' to verify the installation

To verify installation after reboot:
1. Run 'rocm-info' to check device detection
2. Run 'rocm-smi' to monitor GPU metrics
"
}

# Start installation
main
