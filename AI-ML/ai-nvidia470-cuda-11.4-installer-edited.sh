########ADD CUDATOOL KIT AND UPGRADE CUDA TO 14.8


#!/bin/bash

# NVIDIA Driver and CUDA 11.4 Installation Script
# Version: 1.2.1 - Built by Grok 3, xAI
# Date: March 12, 2025
set -e

echo "🚀 Starting NVIDIA Driver 470 and CUDA 11.4 Installation..."

check_root() {
    echo "🔍 Checking for root privileges..."
    [ "$(id -u)" != "0" ] && { echo "❌ Requires root. Run with sudo."; exit 1; }
    echo "✅ Running as root"
}

remove_existing_nvidia() {
    echo "🧹 Removing existing NVIDIA installations..."
    apt-get remove --purge -y '^nvidia-.*' '^cuda-.*' || true
    if command -v nvidia-uninstall &>/dev/null; then
        echo "📥 Removing NVIDIA .run driver installation..."
        nvidia-uninstall --silent || true
    fi
    apt-get autoremove -y
    apt-get clean
    echo "✅ Existing NVIDIA installations removed"
}

install_dependencies() {
    echo "📦 Installing dependencies for driver and CUDA..."
    apt update
    apt install -y build-essential dkms libglvnd-dev
    echo "✅ Dependencies installed"
}

add_nvidia_repo() {
    echo "🌐 Adding NVIDIA driver repository..."
    add-apt-repository -y ppa:graphics-drivers/ppa
    apt update
    echo "✅ NVIDIA PPA added"
}

install_nvidia_driver() {
    echo "📥 Installing NVIDIA driver 470..."
    apt install -y nvidia-driver-470
    echo "✅ NVIDIA driver 470 installed"
}

install_cuda() {
    echo "🌐 Installing CUDA 11.4 Toolkit..."

    # Check disk space (~10GB needed)
    echo "🔍 Checking disk space (need ~10GB free)..."
    FREE_SPACE=$(df -h /usr/local | awk 'NR==2 {print $4}' | sed 's/G//')
    echo "Free space: $FREE_SPACE GB"
    [ "$(echo "$FREE_SPACE < 10" | bc)" -eq 1 ] && { echo "❌ Insufficient disk space ($FREE_SPACE GB < 10 GB)"; exit 1; }
    echo "✅ $FREE_SPACE GB free—proceeding"

    # Check GCC version (CUDA 11.4 supports up to GCC 10.2)
    echo "🔍 Checking GCC version..."
    GCC_VERSION=$(gcc --version | head -n1 | awk '{print $3}')
    echo "Current GCC version: $GCC_VERSION"
    if [[ "$GCC_VERSION" > "10.2" ]]; then
        echo "⚠️ GCC $GCC_VERSION too new for CUDA 11.4—installing GCC 10"
        apt install -y gcc-10 g++-10
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
        update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100
        echo "✅ GCC 10 set as default"
    fi

    # Download and install CUDA 11.4 Toolkit
    echo "📥 Downloading CUDA 11.4 runfile installer..."
    wget -q --show-progress https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run -O cuda.run
    echo "✅ CUDA installer downloaded"
    echo "📥 Running CUDA 11.4 installer..."
    sh cuda.run --silent --toolkit --no-opengl-libs --override
    echo "✅ CUDA 11.4 Toolkit installed"

    # Clean up
    echo "🧹 Cleaning up installer..."
    rm -fv cuda.run
    echo "✅ Installer cleaned up"

    # Configure library paths
    echo "🔧 Configuring library paths..."
    echo "/usr/local/cuda-11.4/lib64" > /etc/ld.so.conf.d/cuda-11.4.conf
    ldconfig
    echo "✅ Library paths configured"

    # Update environment
    echo "🔧 Setting CUDA_HOME in /etc/environment..."
    grep -q "CUDA_HOME=/usr/local/cuda-11.4" /etc/environment || echo "CUDA_HOME=/usr/local/cuda-11.4" >> /etc/environment
    echo "✅ CUDA_HOME set"

    # Update PATH in user's .bashrc
    echo "🔧 Updating PATH in ~/.bashrc..."
    BASHRC="$HOME/.bashrc"
    [ ! -f "$BASHRC" ] && touch "$BASHRC"
    grep -q "/usr/local/cuda-11.4/bin" "$BASHRC" || echo 'export PATH="$PATH:/usr/local/cuda-11.4/bin"' >> "$BASHRC"
    export PATH="$PATH:/usr/local/cuda-11.4/bin"  # Apply to current session
    echo "✅ PATH updated"
}

verify_installation() {
    echo "🔍 Verifying installations..."

    # Verify NVIDIA driver
    echo "🔍 Checking NVIDIA driver..."
    export PATH="$PATH:/usr/bin:/usr/local/bin"
    nvidia-smi || { echo "❌ nvidia-smi failed—reboot and retry"; exit 1; }
    echo "✅ NVIDIA driver verified"

    # Verify CUDA Toolkit
    echo "🔍 Checking CUDA Toolkit..."
    [ ! -f /usr/local/cuda-11.4/bin/nvcc ] && { echo "❌ nvcc not found—check /usr/local/cuda-11.4/bin"; exit 1; }
    nvcc --version || { echo "❌ nvcc failed—check PATH and installation"; exit 1; }
    echo "✅ CUDA Toolkit verified"
}

main() {
    check_root
    remove_existing_nvidia
    install_dependencies
    add_nvidia_repo
    install_nvidia_driver
    install_cuda
    verify_installation
    echo "
✨ NVIDIA Driver 470 and CUDA 11.4 Installation Complete! ✨
- Driver: NVIDIA 470
- CUDA Toolkit: 11.4
Commands:
- nvidia-smi : Check GPU status
- nvcc --version : Check CUDA version
Notes:
- Reboot recommended to ensure all changes take effect
- Source ~/.bashrc or relogin for PATH updates
"
}

main
