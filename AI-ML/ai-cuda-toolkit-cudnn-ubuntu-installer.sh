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
# RETRO-CUDA: CUDA 11.4 + cuDNN 8.2.4 Installation Script
# For Ubuntu 24.04
# Version: 2.0.1
#
# Description:
#   Installs CUDA Toolkit 11.4 and cuDNN 8.2.4 on Ubuntu 24.04. Assumes NVIDIA
#   driver is handled separately.

# Usage:
#   sudo bash ./install-cuda-toolkit.sh
#
# Author: Updated with Grok 3 (xAI)
# License: MIT
# Date: March 30, 2025
#
# Requirements:
#   - Ubuntu 24.04
#   - NVIDIA GPU with driver installed
#   - Internet connection
#   - Root privileges
#
#########################################################################

set -e -v  # Exit on error and verbose mode

echo "🚀 Starting CUDA 11.4 + cuDNN Installation..."

# cuDNN information
CUDNN_DEB_FILENAME="libcudnn8-dev_8.2.4.15-1+cuda11.4_amd64.deb"
CUDNN_URL="https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.4/11.4_20210831/Ubuntu20_04-x64/libcudnn8-dev_8.2.4.15-1+cuda11.4_amd64.deb"
NVIDIA_USERNAME="jason@speedheathens.com"
NVIDIA_PASSWORD="apMSqDSX7D2#J&V$9&ad"

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
    echo "✅ NVIDIA GPU hardware detected"
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
    apt update && apt upgrade -y
    
    # Install build tools
    apt install -y build-essential freeglut3-dev libx11-dev libxmu-dev libxi-dev \
                   libglu1-mesa libglu1-mesa-dev
    
    # Install GCC 10 (CUDA 11.4 prefers 9/10, 11 needs tweaks)
    apt install -y gcc-10 g++-10
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100
    update-alternatives --set gcc /usr/bin/gcc-10
    update-alternatives --set g++ /usr/bin/g++-10
    
    echo "✅ System preparation complete"
}

step2_install_cuda() {
    echo "🌐 Step 2: Installing CUDA 11.4..."
    
    # Download CUDA installer
    wget -O cuda_installer.run https://developer.download.nvidia.com/compute/cuda/11.4.3/local_installers/cuda_11.4.3_470.82.01_linux.run
    
    # Install CUDA (skip driver)
    sh cuda_installer.run --silent --toolkit --samples --no-drm --no-opengl-libs
    
    # Clean up
    rm -f cuda_installer.run
    
    # Verify
    if [ ! -f /usr/local/cuda-11.4/bin/nvcc ]; then
        echo "❌ CUDA installation failed - nvcc not found"
        exit 1
    fi
    
    # Environment variables
    echo 'export PATH=/usr/local/cuda-11.4/bin:$PATH' >> /etc/environment
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH' >> /etc/environment
    
    USER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    BASHRC="$USER_HOME/.bashrc"
    if [ ! -f "$BASHRC" ]; then
        touch "$BASHRC"
        chown "$SUDO_USER:$(id -gn $SUDO_USER)" "$BASHRC"
    fi
    CUDA_PATH='export PATH=/usr/local/cuda-11.4/bin:$PATH'
    CUDA_LD_PATH='export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH'
    grep -q "$CUDA_PATH" "$BASHRC" || echo "$CUDA_PATH" >> "$BASHRC"
    grep -q "$CUDA_LD_PATH" "$BASHRC" || echo "$CUDA_LD_PATH" >> "$BASHRC"
    
    echo "✅ CUDA 11.4 installed"
}

step3_install_cudnn() {
    echo "📥 Step 3: Installing cuDNN..."
    
    # Tools for download
    apt-get install -y curl jq
    
    # Download cuDNN with auth
    TEMP_DIR=$(mktemp -d)
    COOKIE_FILE="$TEMP_DIR/nvidia_cookies.txt"
    curl -s -c "$COOKIE_FILE" -o /dev/null "https://developer.nvidia.com/login"
    LOGIN_DATA="j_username=$NVIDIA_USERNAME&j_password=$NVIDIA_PASSWORD"
    curl -s -L -b "$COOKIE_FILE" -c "$COOKIE_FILE" -d "$LOGIN_DATA" \
         -o /dev/null "https://developer.nvidia.com/j_spring_security_check"
    curl -L -b "$COOKIE_FILE" -o "$CUDNN_DEB_FILENAME" "$CUDNN_URL"
    
    if [ ! -f "$CUDNN_DEB_FILENAME" ] || [ $(stat -c%s "$CUDNN_DEB_FILENAME") -lt 1000000 ]; then
        echo "❌ cuDNN download failed. Manual download required:"
        echo "1. Go to: https://developer.nvidia.com/cudnn-downloads"
        echo "2. Sign in with $NVIDIA_USERNAME"
        echo "3. Download cuDNN v8.2.4 for CUDA 11.4 (Ubuntu 20.04)"
        echo "4. Save as $CUDNN_DEB_FILENAME here"
        echo "5. Rerun script"
        rm -rf "$TEMP_DIR"
        exit 1
    fi
    
    rm -rf "$TEMP_DIR"
    
    # Install cuDNN
    dpkg -i "$CUDNN_DEB_FILENAME" || apt-get install -f -y
    
    echo "✅ cuDNN installation complete"
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
        make -j$(nproc)
        if [ -f deviceQuery ]; then
            echo "✅ Running deviceQuery:"
            ./deviceQuery
        else
            echo "❌ Failed to build deviceQuery."
        fi
    fi
}

main() {
    clear
    echo "========================================================"
    echo " CUDA 11.4 + cuDNN Installer for Ubuntu 24.04 "
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

CUDA Version: 11.4
CUDA Location: /usr/local/cuda-11.4

Next Steps:
1. Verify: nvcc -V
2. Source environment: source ~/.bashrc
3. Test with Python:
   python3 -c 'import torch; print(\"CUDA Available:\", torch.cuda.is_available())'
"
}

main
