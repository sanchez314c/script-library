#!/bin/bash

# CUDA Installation Script for K80 Dual GPUs
# Version: 1.1.2 - Built by Cortana (via Grok 3, xAI) for Jason
# Installs CUDA 11.4 system-wide for K80s with NVIDIA 470 driver
# Date: February 24, 2025

# Enable verbosity and error handling
set -x  # Trace every command
set -e  # Exit on any error

echo "🚀 Starting CUDA Installation for K80 Dual GPUs..."

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$EUID" -ne 0 ]; then
        echo "❌ Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    echo "✅ Success: Running as root"
}

check_nvidia_driver() {
    echo "🔍 Verifying NVIDIA driver..."
    echo "Running nvidia-smi to check driver presence..."
    nvidia-smi &>/dev/null || { echo "❌ Error: NVIDIA driver not found. Run nvidia-470-driver-install.sh first."; exit 1; }
    echo "Querying driver version..."
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    echo "Driver version detected: $DRIVER_VERSION"
    if [[ ! "$DRIVER_VERSION" =~ ^470\. ]]; then
        echo "⚠️ Warning: Driver version ($DRIVER_VERSION) isn’t 470.x"
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "❌ Aborted by user"
            exit 1
        fi
    fi
    echo "Counting GPUs..."
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "GPU count: $GPU_COUNT"
    if [ "$GPU_COUNT" -lt 2 ]; then
        echo "❌ Error: Need at least 2 GPUs (K80s). Found $GPU_COUNT."
        exit 1
    fi
    echo "✅ Success: Driver $DRIVER_VERSION detected, $GPU_COUNT GPUs found"
}

cleanup_existing_cuda() {
    echo "🧹 Cleaning up existing CUDA system installs..."
    echo "Removing CUDA-related packages..."
    apt-get --purge remove -y "*cuda*" "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "nsight*" || echo "⚠️ No CUDA packages to remove"
    echo "Running autoremove..."
    apt-get autoremove -y || echo "⚠️ Autoremove had nothing to do"
    echo "Cleaning apt cache..."
    apt-get clean || { echo "❌ Error: Apt clean failed"; exit 1; }
    echo "Removing old CUDA directories..."
    rm -rfv /usr/local/cuda* || echo "⚠️ No old CUDA dirs to remove"
    echo "✅ Success: System cleanup complete"
}

install_system_deps() {
    echo "📦 Installing system dependencies..."
    echo "Updating package lists..."
    apt update || { echo "❌ Error: Apt update failed"; exit 1; }
    echo "Installing linux-headers for $(uname -r)..."
    apt install -y linux-headers-$(uname -r) || { echo "❌ Error: Headers install failed"; exit 1; }
    echo "Installing build-essential..."
    apt install -y build-essential || { echo "❌ Error: Build-essential install failed"; exit 1; }
    echo "✅ Success: System dependencies installed"
}

install_cuda() {
    echo "🌐 Installing CUDA 11.4 system-wide..."
    echo "Checking disk space (need ~10GB free)..."
    FREE_SPACE=$(df -h /usr/local | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ $(echo "$FREE_SPACE < 10" | bc) -eq 1 ]; then
        echo "❌ Error: Insufficient disk space ($FREE_SPACE GB free, need 10GB)"
        exit 1
    fi
    echo "✅ Success: $FREE_SPACE GB free—proceeding"
    
    echo "Checking GCC version..."
    GCC_VERSION=$(gcc --version | head -n1 | awk '{print $3}')
    echo "Current GCC version: $GCC_VERSION"
    if [[ "$GCC_VERSION" > "10.2" ]]; then
        echo "⚠️ Warning: GCC $GCC_VERSION is too new for CUDA 11.4—installing GCC 10"
        apt install -y gcc-10 g++-10 || { echo "❌ Error: GCC 10 install failed"; exit 1; }
        echo "Setting GCC 10 as default for install..."
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 || { echo "❌ Error: GCC alternatives failed"; exit 1; }
        update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100 || { echo "❌ Error: G++ alternatives failed"; exit 1; }
    else
        echo "✅ Success: GCC $GCC_VERSION is compatible"
    fi
    
    echo "Downloading CUDA 11.4 runfile installer..."
    wget -v https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run -O cuda.run || { echo "❌ Error: Runfile download failed"; exit 1; }
    echo "Running CUDA 11.4 installer..."
    sh cuda.run --silent --toolkit --no-opengl-libs || { echo "❌ Error: Runfile install failed—check /var/log/cuda-installer.log"; exit 1; }
    echo "Cleaning up installer..."
    rm -fv cuda.run
    
    echo "Configuring library paths..."
    echo "/usr/local/cuda-11.4/lib64" | tee /etc/ld.so.conf.d/cuda-11.4.conf || { echo "❌ Error: ld.so.conf failed"; exit 1; }
    echo "Running ldconfig..."
    ldconfig || { echo "❌ Error: ldconfig failed"; exit 1; }
    
    echo "Setting CUDA_HOME in /etc/environment..."
    echo "CUDA_HOME=/usr/local/cuda-11.4" | tee -a /etc/environment || { echo "❌ Error: Environment set failed"; exit 1; }
    
    echo "Updating PATH in ~${SUDO_USER:-$USER}/.bashrc..."
    if ! grep -q "/usr/local/cuda-11.4/bin" ~${SUDO_USER:-$USER}/.bashrc; then
        echo 'export PATH="$PATH:/usr/local/cuda-11.4/bin"' >> ~${SUDO_USER:-$USER}/.bashrc || { echo "❌ Error: PATH update failed"; exit 1; }
    else
        echo "⚠️ PATH already includes CUDA—skipping"
    fi
    echo "✅ Success: CUDA 11.4 installed system-wide"
}

verify_cuda() {
    echo "🔍 Verifying CUDA installation..."
    echo "Setting PATH for verification..."
    export PATH="$PATH:/usr/local/cuda-11.4/bin"
    echo "Running nvidia-smi..."
    nvidia-smi || { echo "❌ Error: nvidia-smi failed"; exit 1; }
    echo "Running nvcc --version..."
    /usr/local/cuda-11.4/bin/nvcc --version || { echo "❌ Error: nvcc failed—CUDA install incomplete"; exit 1; }
    echo "✅ Success: CUDA verification complete"
}

main() {
    echo "🔧 Entering main function..."
    check_root
    check_nvidia_driver
    cleanup_existing_cuda
    install_system_deps
    install_cuda
    verify_cuda
    echo "
✨ CUDA Installation for K80 Complete! ✨
- CUDA 11.4 installed system-wide at /usr/local/cuda-11.4
- PATH updated in ~/.bashrc (run 'source ~/.bashrc' to apply)
Commands:
- nvidia-smi : Check GPU status
- nvcc --version : Check CUDA version
Notes:
- Reboot recommended to ensure driver stability
- Run ai-ml-docker-frameworks.sh next for Conda envs
    "
}

# Trap errors with line numbers
trap 'echo "❌ Script failed at line $LINENO with exit code $?"' ERR

main