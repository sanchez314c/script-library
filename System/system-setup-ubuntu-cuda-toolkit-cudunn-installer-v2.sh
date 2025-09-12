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
# Version: 2.2.0
#
# Description:
#   Installs CUDA Toolkit 11.4.3 and cuDNN 8.2.4 on Ubuntu 24.04.
#   Compatible with NVIDIA driver 470.

# Usage:
#   sudo bash ./install-cuda-toolkit.sh
#
#########################################################################

set -e  # Exit on error

echo "🚀 Starting CUDA 11.4.3 + cuDNN Installation..."

# Get username even when run with sudo
REAL_USER="${SUDO_USER:-$USER}"

# Log file on desktop
LOG_FILE="/home/$REAL_USER/Desktop/cuda_install_$(date +%Y%m%d_%H%M%S).log"
touch "$LOG_FILE"
chown $REAL_USER:$REAL_USER "$LOG_FILE"
chmod 644 "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# Define important variables
CUDA_VERSION="11.4.3"
CUDA_SHORT_VERSION="11.4"
CUDA_INSTALLER_VERSION="470.82.01"
CUDA_PATH="/usr/local/cuda-$CUDA_SHORT_VERSION"
CUDA_RUNFILE="cuda_${CUDA_VERSION}_${CUDA_INSTALLER_VERSION}_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${CUDA_RUNFILE}"

# cuDNN information
CUDNN_VERSION="8.2.4"
CUDNN_PATCH_VERSION="8.2.4.15"
CUDNN_ARCHIVE="cudnn-${CUDA_SHORT_VERSION}-linux-x64-v${CUDNN_PATCH_VERSION}.tgz"
CUDNN_URL="https://developer.download.nvidia.com/compute/redist/cudnn/v${CUDNN_VERSION}/cudnn-${CUDA_SHORT_VERSION}-linux-x64-v${CUDNN_PATCH_VERSION}.tgz"

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
    echo "🌐 Step 2: Installing CUDA ${CUDA_VERSION}..."
    
    # Download CUDA installer
    cd /tmp
    echo "📥 Downloading CUDA installer (this may take a while)..."
    wget -q --show-progress "$CUDA_URL"
    
    # Make installer executable
    chmod +x "$CUDA_RUNFILE"
    
    # Install CUDA (skip driver)
    echo "🔧 Installing CUDA Toolkit (this may take a while)..."
    ./"$CUDA_RUNFILE" --silent --toolkit --samples --no-drm --no-opengl-libs --override
    
    # Clean up
    rm -f "$CUDA_RUNFILE"
    
    # Verify
    if [ ! -f "${CUDA_PATH}/bin/nvcc" ]; then
        echo "❌ CUDA installation failed - nvcc not found"
        exit 1
    fi
    
    # Environment variables
    echo "export PATH=${CUDA_PATH}/bin:\$PATH" >> /etc/environment
    echo "export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH" >> /etc/environment
    
    # Create symlink
    ln -sf "${CUDA_PATH}" /usr/local/cuda
    
    # Add to current shell
    export PATH="${CUDA_PATH}/bin:$PATH"
    export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:$LD_LIBRARY_PATH"
    
    # Update ldconfig
    echo "${CUDA_PATH}/lib64" > /etc/ld.so.conf.d/cuda-${CUDA_SHORT_VERSION}.conf
    ldconfig
    
    echo "✅ CUDA ${CUDA_VERSION} installed successfully"
}

step3_install_cudnn() {
    echo "📥 Step 3: Installing cuDNN ${CUDNN_VERSION}..."
    
    # Download cuDNN
    cd /tmp
    
    # Try direct download first
    echo "📥 Downloading cuDNN..."
    if wget -q --show-progress "$CUDNN_URL" -O "$CUDNN_ARCHIVE"; then
        echo "✅ cuDNN download successful"
    else
        echo "⚠️ Could not download cuDNN from direct link"
        echo "Please download cuDNN v${CUDNN_VERSION} for CUDA ${CUDA_SHORT_VERSION} manually from NVIDIA website:"
        echo "1. Go to https://developer.nvidia.com/rdp/cudnn-archive"
        echo "2. Download 'cuDNN v${CUDNN_VERSION} (July 6th, 2021), for CUDA ${CUDA_SHORT_VERSION}'"
        echo "3. Place the downloaded file in /tmp directory"
        echo "4. Rename it to ${CUDNN_ARCHIVE}"
        read -p "Press Enter once you've downloaded the file to continue..."
        
        if [ ! -f "/tmp/${CUDNN_ARCHIVE}" ]; then
            echo "❌ cuDNN file not found. Installation cannot proceed."
            exit 1
        fi
    fi
    
    # Extract and install cuDNN
    echo "🔧 Installing cuDNN..."
    tar -xzf "$CUDNN_ARCHIVE"
    
    # Copy files to CUDA toolkit directory
    cp -P cuda/include/cudnn*.h "${CUDA_PATH}/include/"
    cp -P cuda/lib64/libcudnn* "${CUDA_PATH}/lib64/"
    chmod a+r "${CUDA_PATH}/include/cudnn*.h" "${CUDA_PATH}/lib64/libcudnn*"
    
    # Clean up
    rm -rf cuda
    rm -f "$CUDNN_ARCHIVE"
    
    # Run ldconfig again to ensure libraries are found
    ldconfig
    
    echo "✅ cuDNN ${CUDNN_VERSION} installed successfully"
}

step4_test_installation() {
    echo "🧪 Step 4: Testing Installation..."
    
    # Apply PATH for this session
    export PATH="${CUDA_PATH}/bin:$PATH"
    export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:$LD_LIBRARY_PATH"
    
    # Check CUDA
    if command -v nvcc &> /dev/null; then
        echo "✅ CUDA is working:"
        nvcc --version
    else
        echo "❌ CUDA not working. Check installation."
    fi
    
    # Check cuDNN
    if [ -f "${CUDA_PATH}/include/cudnn_version.h" ]; then
        echo "✅ cuDNN is installed:"
        cat "${CUDA_PATH}/include/cudnn_version.h" | grep CUDNN_MAJOR -A 2
    else
        echo "❌ cuDNN header not found."
    fi
    
    # Test CUDA sample
    if [ -d "${CUDA_PATH}/samples/1_Utilities/deviceQuery/" ]; then
        cd "${CUDA_PATH}/samples/1_Utilities/deviceQuery/"
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
    echo " CUDA ${CUDA_VERSION} + cuDNN ${CUDNN_VERSION} Installer for Ubuntu 24.04 "
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

CUDA Version: ${CUDA_VERSION}
CUDA Location: ${CUDA_PATH}
cuDNN Version: ${CUDNN_VERSION}

Next Steps:
1. Log out and log back in to apply environment changes
2. Verify installation with: nvcc -V
3. For Python frameworks:
   - TensorFlow 2.6-2.8 (for CUDA 11.4)
   - PyTorch 1.10-1.12 (for CUDA 11.4)

If you need to uninstall:
- Run: sudo rm -rf /usr/local/cuda*
- Remove environment variables from /etc/environment
- Remove the CUDA configuration from /etc/ld.so.conf.d/
"
}

main