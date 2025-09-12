#!/bin/bash
# ROCm Installation/Uninstallation Script for RX580 with HIP Fix
# Version: 4.5
# Date: March 23, 2025
# Author: Cortana & Jason Paul Michaels
# Description: Installs/Uninstalls ROCm 6.3.4 for RX580, applies HIP fix, and handles package conflicts

set -e

# Control verbosity with DEBUG variable (set to true for debugging)
DEBUG=${DEBUG:-false}
if [ "$DEBUG" = true ]; then
    set -x
fi

# Set TARGET_USER at the beginning of the script
TARGET_USER=$(logname || echo $SUDO_USER || echo $USER)

# Create log file
LOG_FILE="/tmp/rocm_install_$(date +%Y%m%d_%H%M%S).log"
exec &> >(tee -a "$LOG_FILE")

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$EUID" -ne 0 ]; then
        echo "❌ Error: Requires root privileges. Run with sudo or as root."
        exit 1
    fi
    echo "✅ Running as root"
}

uninstall_rocm() {
    echo "🧹 Uninstalling ROCm and cleaning up..."
    
    # Kill any processes using ROCm
    echo "Stopping any processes using ROCm..."
    sudo fuser -k /opt/rocm/lib/lib* 2>/dev/null || true
    
    # Remove packages with dpkg force
    echo "Force removing problematic packages..."
    sudo dpkg --force-all -P rocm-llvm rocm-hip-sdk rocm-hip-libraries rocm-libs 2>/dev/null || true
    sudo dpkg --force-all -P amdgpu-install 2>/dev/null || true
    
    # Clean up any remaining packages
    echo "Removing ROCm packages..."
    local packages=(
        "rocm*"
        "hip*"
        "comgr*"
        "hsa*"
        "rocblas*"
        "amdgpu*"
        "libdrm-amdgpu*"
        "libgl1-amdgpu-mesa*"
        "libglapi-amdgpu-mesa*"
        "libllvm*-amdgpu"
        "openmp-extras*"
        "roctracer*"
    )
    
    for pkg in "${packages[@]}"; do
        sudo dpkg --force-all -P $pkg 2>/dev/null || true
    done
    
    echo "Running final package cleanup..."
    sudo apt-get --fix-broken install -y
    sudo apt-get purge -y "${packages[@]}" 2>/dev/null || true
    sudo apt-get autoremove -y
    
    echo "Removing repositories and keys..."
    sudo rm -f /etc/apt/sources.list.d/amdgpu.list
    sudo rm -f /etc/apt/sources.list.d/rocm.list
    sudo rm -f /etc/apt/trusted.gpg.d/rocm-*
    
    echo "Clearing APT cache..."
    sudo rm -rf /var/cache/apt/*
    sudo apt clean all
    sudo apt update
    
    echo "Removing ROCm directories..."
    sudo rm -rf /opt/rocm* || true
    
    # Remove any remaining config files
    sudo rm -f /etc/ld.so.conf.d/rocm.conf
    sudo ldconfig
    
    echo "✅ Uninstallation and cleanup complete"
}

install_rocm() {
    export DEBIAN_FRONTEND=noninteractive
    echo "📦 Starting ROCm 6.3.4 system-wide installation..."

    # Complete removal first
    uninstall_rocm

    echo "📦 Installing dependencies..."
    sudo apt update && sudo apt install -y \
        "linux-headers-$(uname -r)" \
        "linux-modules-extra-$(uname -r)" \
        libopenmpi3 \
        libstdc++-12-dev \
        libdnnl-dev \
        ninja-build \
        libopenblas-dev \
        libpng-dev \
        libjpeg-dev \
        cmake \
        git \
        python3-dev \
        python3-pip \
        build-essential || { echo "❌ Dependency installation failed"; exit 1; }

    echo "📥 Downloading AMDGPU installer..."
    cd /tmp
    rm -fv amdgpu-install_6.3.60304-1_all.deb*
    wget -v https://repo.radeon.com/amdgpu-install/6.3.4/ubuntu/noble/amdgpu-install_6.3.60304-1_all.deb || { 
        echo "❌ Download failed. Checking connectivity to repo.radeon.com..."
        if ! ping -c 1 repo.radeon.com; then
            echo "❌ Cannot reach repo.radeon.com. Check your internet connection."
            exit 1
        fi
        exit 1
    }

    echo "🔧 Installing AMDGPU installer..."
    sudo dpkg -i amdgpu-install_6.3.60304-1_all.deb || {
        echo "❌ dpkg failed—running apt fix"
        sudo apt install -f -y
        sudo dpkg -i amdgpu-install_6.3.60304-1_all.deb || exit 1
    }

    echo "🔄 Updating package lists..."
    sudo apt update || { echo "❌ apt update failed"; exit 1; }

    echo "🔨 Installing ROCm packages..."
    sudo amdgpu-install --usecase=graphics,rocm,rocmdev,rocmdevtools,opencl,openclsdk,hip,hiplibsdk,mllib,mlsdk -y || {
        echo "❌ ROCm installation failed"
        echo "Checking GPU detection..."
        lspci | grep -i "RX 580" || echo "⚠️ RX580 not detected!"
        exit 1
    }

    echo "🔧 Ensuring core components are present..."
    sudo apt install -y \
        hip-dev \
        hip-runtime-amd \
        rocm-dev \
        rocm-libs || { echo "❌ Core package installation failed"; exit 1; }

    # Now that ROCm is installed, set up the environment
    export ROCM_PATH=/opt/rocm-6.3.4
    export HIP_PLATFORM=amd
    export PATH=$ROCM_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

    echo "🔧 Setting ROCm library paths..."
    sudo tee /etc/ld.so.conf.d/rocm.conf > /dev/null <<EOF
/opt/rocm-6.3.4/lib
/opt/rocm-6.3.4/lib64
EOF
    sudo ldconfig || { echo "❌ ldconfig failed"; exit 1; }

    echo "🔧 Adding ROCm paths to system environment..."
    # Backup the original /etc/environment file
    sudo cp /etc/environment /etc/environment.bak
    
    # Check if variables already exist and update or add them
    if grep -q "ROCM_PATH=" /etc/environment; then
        sudo sed -i 's|ROCM_PATH=.*|ROCM_PATH=/opt/rocm-6.3.4|' /etc/environment
    else
        echo 'ROCM_PATH=/opt/rocm-6.3.4' | sudo tee -a /etc/environment > /dev/null
    fi
    
    if grep -q "HIP_PLATFORM=" /etc/environment; then
        sudo sed -i 's|HIP_PLATFORM=.*|HIP_PLATFORM=amd|' /etc/environment
    else
        echo 'HIP_PLATFORM=amd' | sudo tee -a /etc/environment > /dev/null
    fi
    
    # Update PATH if it exists, otherwise add it
    if grep -q "^PATH=" /etc/environment; then
        # Check if ROCm path is already in PATH
        if ! grep -q "PATH=.*rocm-6.3.4/bin" /etc/environment; then
            sudo sed -i 's|PATH="\(.*\)"|PATH="/opt/rocm-6.3.4/bin:\1"|' /etc/environment
        fi
    else
        echo 'PATH="/opt/rocm-6.3.4/bin:$PATH"' | sudo tee -a /etc/environment > /dev/null
    fi
    
    # Update LD_LIBRARY_PATH if it exists, otherwise add it
    if grep -q "^LD_LIBRARY_PATH=" /etc/environment; then
        # Check if ROCm lib path is already in LD_LIBRARY_PATH
        if ! grep -q "LD_LIBRARY_PATH=.*rocm-6.3.4/lib" /etc/environment; then
            sudo sed -i 's|LD_LIBRARY_PATH="\(.*\)"|LD_LIBRARY_PATH="/opt/rocm-6.3.4/lib:\1"|' /etc/environment
        fi
    else
        echo 'LD_LIBRARY_PATH="/opt/rocm-6.3.4/lib"' | sudo tee -a /etc/environment > /dev/null
    fi

    echo "🔧 Applying HIP directory fix..."
    sudo mkdir -p /opt/rocm-6.3.4/hip
    sudo ln -sf /opt/rocm-6.3.4/include/hip /opt/rocm-6.3.4/hip/include
    sudo ln -sf /opt/rocm-6.3.4/lib /opt/rocm-6.3.4/hip/lib
    sudo ln -sf /opt/rocm-6.3.4/hip /opt/rocm/hip

    echo "🔄 Setting up user permissions..."
    sudo usermod -a -G video,render "$TARGET_USER" || { echo "❌ Group modification failed"; exit 1; }

    echo "🔧 Setting device permissions..."
    [ -e /dev/kfd ] && sudo chmod 660 /dev/kfd && sudo chown root:render /dev/kfd
    for dev in /dev/dri/card* /dev/dri/render*; do
        [ -e "$dev" ] && sudo chmod 660 "$dev" && sudo chown root:render "$dev"
    done

    if lspci | grep -i "RX 580"; then
        echo "✅ RX580 detected"
        RX580_DEV=$(ls /dev/dri/card* | grep -i "RX 580" || echo "")
        if [ -n "$RX580_DEV" ]; then
            echo "🔧 Setting RX580 permissions..."
            sudo chmod 660 "$RX580_DEV"
            sudo chown root:render "$RX580_DEV"
        fi
    else
        echo "⚠️ RX580 not detected"
    fi

    echo "🔄 Reloading udev rules..."
    sudo udevadm control --reload-rules && sudo udevadm trigger
}

apply_group_changes() {
    echo "🔄 Applying group changes..."
    echo "✅ Group changes will take effect after you log out and log back in"
    # Don't try to use newgrp which can hang the script
    sudo usermod -a -G video,render "$TARGET_USER"
    echo "Groups updated for $TARGET_USER"
}

run_final_checks() {
    echo "🔍 Running final checks..."
    # Load environment variables from /etc/environment
export $(grep -v '^#' /etc/environment | xargs -d '\n')
    
    export PATH=$PATH:/opt/rocm-6.3.4/bin
    export LD_LIBRARY_PATH=/opt/rocm-6.3.4/lib:$LD_LIBRARY_PATH

    local check_failed=0

    # Function to run checks
    run_check() {
        local cmd=$1
        local name=$2
        echo "Running $name..."
        if ! eval $cmd > /dev/null 2>&1; then
            echo "❌ $name failed"
            check_failed=1
        else
            echo "✅ $name executed successfully"
            eval $cmd | head -n 5
        fi
    }

    run_check "rocminfo" "rocminfo"
    run_check "rocm-smi" "rocm-smi"
    run_check "/opt/rocm-6.3.4/bin/hipcc --version" "hipcc"

    echo "Checking HIP paths..."
    if [ -d "/opt/rocm-6.3.4/hip" ] && [ -L "/opt/rocm/hip" ]; then
        echo "✅ HIP directory structure verified"
    else
        echo "❌ HIP directory structure incomplete"
        check_failed=1
    fi

    if [ $check_failed -eq 1 ]; then
        echo "⚠️ Some checks failed - review output above"
    else
        echo "✅ All checks passed successfully"
    fi
}

main() {
    check_root
    
    if [ "$1" = "uninstall" ]; then
        echo "🧹 Starting ROCm uninstallation process..."
        uninstall_rocm
        echo "ROCm has been uninstalled and the system cleaned up."
    else
        echo "🔧 Starting ROCm 6.3.4 installation..."
        install_rocm
        apply_group_changes
        run_final_checks
        echo "
✨ ROCm 6.3.4 Installation Complete! ✨

📋 IMPORTANT NOTES:
- When installing other packages, if you get dependency errors, use:
  \`sudo apt --fix-broken install --no-remove\`
- Group changes require logging out and back in to take effect

🔄 After a reboot, you may need to run:
\`sudo ldconfig\`

For the current session, load the environment with:
\`source /etc/environment\`

📝 Installation log saved to: $LOG_FILE
"
    fi
}

trap 'echo "❌ Script failed at line $LINENO with exit code $?"' ERR

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [ "$1" = "uninstall" ]; then
        main uninstall
    else
        main
    fi
fi
```