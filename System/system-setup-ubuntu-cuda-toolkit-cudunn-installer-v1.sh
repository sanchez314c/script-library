#!/bin/bash
#########################################################################
#     ____  _____ _____ ____   ___        ____ _   _ ____    _          #
#    |  _ \| ____|_   _|  _ \ / _ \      / ___| | | |  _ \  / \         #
#    | |_) |  _|   | | | |_) | | | |____| |   | | | | | | |/ _ \        #
#    |  _ <| |___  | | |  _ <| |_| |____| |___| |_| | |_| / ___ \       #
#    |_| \_\_____| |_| |_| \_\\___/      \____|\___/|____/_/   \_\      #
#                                                                       #
#########################################################################
#
# RETRO-CUDA: CUDA 11.4.3 + cuDNN 8.2.4 Installation Script
# For Ubuntu 24.04
# Version: 2.0.2
#
# Description:
#   Installs CUDA Toolkit 11.4.3 and cuDNN 8.2.4 on Ubuntu 24.04. Assumes NVIDIA
#   driver 470 is already installed.

# Usage:
#   sudo bash ./install-cuda-toolkit.sh
#
# Requirements:
#   - Ubuntu 24.04
#   - NVIDIA GPU with driver 470 installed
#   - Internet connection
#   - Root privileges
#
#########################################################################

set -e  # Exit on error

echo "🚀 Starting CUDA 11.4.3 + cuDNN Installation..."

# Define important variables
CUDA_VERSION="11.4.3"
CUDA_DIRNAME="cuda-$CUDA_VERSION"
CUDA_PATH="/usr/local/$CUDA_DIRNAME"
CUDA_SAMPLES_PATH="/usr/local/cuda-samples"

# cuDNN information (use direct link from NVIDIA public CDN)
CUDNN_VERSION="8.2.4"
CUDNN_FOLDER="cudnn-8.2.4-linux-x64-v$CUDNN_VERSION"
CUDNN_ARCHIVE="$CUDNN_FOLDER.tgz"
CUDNN_URL="https://developer.download.nvidia.com/compute/redist/cudnn/v$CUDNN_VERSION/cudnn-11.4-linux-x64-v$CUDNN_VERSION.tgz"

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "❌ This script requires root privileges. Run with sudo."
        exit 1
    fi
    echo "✅ Running as root"
}

check_for_nvidia_gpu() {
    echo "🔍 Checking for NVIDIA GPU hardware..."
    if ! lspci | grep -i nvidia > /dev/null; then
        echo "❌ No NVIDIA GPU hardware detected. Installation cannot proceed."
        exit 1
    fi
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ NVIDIA driver not found. Please install NVIDIA driver 470 first."
        exit 1
    fi
    
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    if [[ "$DRIVER_VERSION" != 470* ]]; then
        echo "⚠️ Warning: Current NVIDIA driver version is $DRIVER_VERSION."
        echo "⚠️ CUDA 11.4.3 works best with NVIDIA driver 470.x"
        read -p "Continue anyway? (y/N): " CONTINUE
        if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "✅ NVIDIA GPU with driver $DRIVER_VERSION detected"
    fi
}

cleanup_previous_installations() {
    echo "🧹 Checking for previous CUDA/cuDNN installations..."
    
    # Check for existing CUDA
    if [ -d /usr/local/cuda ] || ls -d /usr/local/cuda-* &>/dev/null; then
        echo "⚠️ Existing CUDA installation(s) detected"
        read -p "Remove existing CUDA installations? (y/N): " REMOVE_CUDA
        if [[ "$REMOVE_CUDA" =~ ^[Yy]$ ]]; then
            echo "🔄 Removing existing CUDA installations..."
            rm -rf /usr/local/cuda /usr/local/cuda-*
            sed -i '/cuda/d' /etc/environment
            find /etc/ld.so.conf.d/ -name "*cuda*.conf" -delete
            ldconfig
            echo "✅ Existing CUDA removed"
        else
            echo "👉 Keeping existing CUDA"
        fi
    fi
    
    # Check for existing cuDNN
    CUDNN_PACKAGES=$(dpkg -l | grep -i cudnn | awk '{print $2}')
    if [ -n "$CUDNN_PACKAGES" ]; then
        echo "⚠️ Existing cuDNN packages detected:"
        echo "$CUDNN_PACKAGES"
        read -p "Remove existing cuDNN packages? (y/N): " REMOVE_CUDNN
        if [[ "$REMOVE_CUDNN" =~ ^[Yy]$ ]]; then
            echo "🔄 Removing existing cuDNN packages..."
            apt purge -y libcudnn*
            echo "✅ Existing cuDNN removed"
        else
            echo "👉 Keeping existing cuDNN"
        fi
    fi
    
    echo "✅ Cleanup checks complete"
}

step1_system_preparation() {
    echo "📦 Step 1: System Preparation..."
    
    # Update system
    apt update
    
    # Install build tools
    apt install -y build-essential freeglut3-dev libx11-dev libxmu-dev libxi-dev \
                   libglu1-mesa libglu1-mesa-dev
    
    # Install GCC 10 (CUDA 11.4 works best with GCC 10)
    apt install -y gcc-10 g++-10
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100
    update-alternatives --set gcc /usr/bin/gcc-10
    update-alternatives --set g++ /usr/bin/g++-10
    
    echo "✅ System preparation complete"
}

step2_install_cuda() {
    echo "🌐 Step 2: Installing CUDA 11.4.3..."
    
    # Download CUDA installer
    cd /tmp
    echo "📥 Downloading CUDA installer (this may take a while)..."
    wget -q --show-progress https://developer.download.nvidia.com/compute/cuda/11.4.3/local_installers/cuda_11.4.3_470.82.01_linux.run
    
    # Make installer executable
    chmod +x cuda_11.4.3_470.82.01_linux.run
    
    # Install CUDA (skip driver)
    echo "🔧 Installing CUDA Toolkit (this may take a while)..."
    ./cuda_11.4.3_470.82.01_linux.run --silent --toolkit --samples --no-drm --no-opengl-libs
    
    # Clean up
    rm -f cuda_11.4.3_470.82.01_linux.run
    
    # Verify
    if [ ! -f /usr/local/cuda-11.4/bin/nvcc ]; then
        echo "❌ CUDA installation failed - nvcc not found"
        exit 1
    fi
    
    # Environment variables
    echo 'export PATH=/usr/local/cuda-11.4/bin:$PATH' >> /etc/environment
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH' >> /etc/environment
    
    # Create symlink
    ln -sf /usr/local/cuda-11.4 /usr/local/cuda
    
    # Add to current shell
    export PATH=/usr/local/cuda-11.4/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
    
    # Update ldconfig
    echo "/usr/local/cuda-11.4/lib64" > /etc/ld.so.conf.d/cuda-11.4.conf
    ldconfig
    
    echo "✅ CUDA 11.4.3 installed successfully"
}

step3_install_cudnn() {
    echo "📥 Step 3: Installing cuDNN 8.2.4..."
    
    # Download cuDNN
    cd /tmp
    
    # Try alternative sources for cuDNN - no login required
    echo "📥 Downloading cuDNN..."
    if wget -q --show-progress "https://developer.download.nvidia.com/compute/redist/cudnn/v8.2.4/cudnn-11.4-linux-x64-v8.2.4.15.tgz"; then
        echo "✅ cuDNN download successful"
    else
        echo "⚠️ Could not download cuDNN from direct link"
        echo "Please download cuDNN v8.2.4 for CUDA 11.4 manually from NVIDIA website:"
        echo "1. Go to https://developer.nvidia.com/rdp/cudnn-archive"
        echo "2. Download 'cuDNN v8.2.4 (July 6th, 2021), for CUDA 11.4'"
        echo "3. Place the downloaded file in /tmp directory"
        echo "4. Rename it to cudnn-11.4-linux-x64-v8.2.4.15.tgz"
        read -p "Press Enter once you've downloaded the file to continue..."
        
        if [ ! -f /tmp/cudnn-11.4-linux-x64-v8.2.4.15.tgz ]; then
            echo "❌ cuDNN file not found. Installation cannot proceed."
            exit 1
        fi
    fi
    
    # Extract and install cuDNN
    echo "🔧 Installing cuDNN..."
    tar -xzf cudnn-11.4-linux-x64-v8.2.4.15.tgz
    
    # Copy files to CUDA toolkit directory
    cp -P cuda/include/cudnn*.h /usr/local/cuda-11.4/include
    cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.4/lib64
    chmod a+r /usr/local/cuda-11.4/include/cudnn*.h /usr/local/cuda-11.4/lib64/libcudnn*
    
    # Clean up
    rm -rf cuda
    rm -f cudnn-11.4-linux-x64-v8.2.4.15.tgz
    
    echo "✅ cuDNN 8.2.4 installed successfully"
}

step4_test_installation() {
    echo "🧪 Step 4: Testing Installation..."
    
    # Apply PATH for this session
    export PATH=/usr/local/cuda-11.4/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
    
    # Check CUDA
    if command -v nvcc &> /dev/null; then
        echo "✅ CUDA is working:"
        nvcc --version
    else
        echo "❌ CUDA not working. Check installation."
    fi
    
    # Check cuDNN
    if [ -f /usr/local/cuda-11.4/include/cudnn_version.h ]; then
        echo "✅ cuDNN is installed:"
        cat /usr/local/cuda-11.4/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
    else
        echo "❌ cuDNN header not found."
    fi
    
    # Test CUDA sample
    if [ -d /usr/local/cuda-11.4/samples/1_Utilities/deviceQuery/ ]; then
        cd /usr/local/cuda-11.4/samples/1_Utilities/deviceQuery/
        make -j$(nproc) >/dev/null 2>&1
        if [ -f deviceQuery ]; then
            echo "✅ Running deviceQuery sample:"
            ./deviceQuery | grep "Result\|CUDA Driver"
        else
            echo "❌ Failed to build deviceQuery sample."
        fi
    fi
    
    echo "✅ CUDA and cuDNN installation tests completed."
}

main() {
    clear
    echo "========================================================"
    echo " CUDA 11.4.3 + cuDNN 8.2.4 Installer for Ubuntu 24.04 "
    echo "========================================================"
    
    check_root
    check_for_nvidia_gpu
    cleanup_previous_installations
    
    step1_system_preparation
    step2_install_cuda
    step3_install_cudnn
    step4_test_installation
    
    echo "
✨ Installation Complete! ✨

CUDA Version: 11.4.3
CUDA Location: /usr/local/cuda-11.4
cuDNN Version: 8.2.4

Next Steps:
1. Log out and log back in to apply environment changes
2. Verify installation with: nvcc -V
3. For Python frameworks, make sure to install compatible versions:
   - TensorFlow 2.6-2.8 (for CUDA 11.4)
   - PyTorch 1.10-1.12 (for CUDA 11.4)

If you need to uninstall:
- Run: sudo rm -rf /usr/local/cuda*
- Remove environment variables from /etc/environment
"
}

main