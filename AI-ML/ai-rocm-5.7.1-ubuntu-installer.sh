#!/bin/bash

# Exit on error
set -e

# ROCm 5.7.1 Installation Script for Ubuntu 24.04.2 LTS with amdgpu-install --no-dkms
# Installs graphics, ROCm, HIP, and OpenCL support

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$EUID" -ne 0 ]; then
        echo "❌ Error: This script requires root privileges. Run with sudo or as root."
        exit 1
    fi
    echo "✅ Success: Running as root"
}

install_prerequisites() {
    echo "📋 Installing prerequisites..."

    # Update system
    echo "📦 Updating package lists and upgrading system..."
    sudo apt update && sudo apt upgrade -y || { echo "❌ Error: System update failed"; exit 1; }

    # Install basic tools
    echo "📦 Installing wget and curl..."
    sudo apt install -y wget curl || { echo "❌ Error: Failed to install basic tools"; exit 1; }

    # Install kernel headers and modules
    echo "📦 Installing kernel headers and extra modules..."
    sudo apt install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)" || { echo "❌ Error: Kernel dependencies failed"; exit 1; }

    # Add user to GPU groups
    TARGET_USER="${SUDO_USER:-${LOGNAME:-heathen-admin}}"
    echo "ℹ️ Target user identified as: $TARGET_USER"
    if [ -z "$TARGET_USER" ] || [ "$TARGET_USER" = "root" ]; then
        TARGET_USER="heathen-admin"
        echo "ℹ️ Defaulting to user: $TARGET_USER"
    fi
    sudo usermod -a -G video "$TARGET_USER" || { echo "❌ Error: Adding $TARGET_USER to video group failed"; exit 1; }
    sudo usermod -a -G render "$TARGET_USER" || { echo "❌ Error: Adding $TARGET_USER to render group failed"; exit 1; }
    echo "✅ Success: $TARGET_USER added to video and render groups"
}

install_amdgpu_installer() {
    echo "📥 Installing amdgpu-install..."

    # Clean up any old installer
    echo "🧹 Cleaning up old amdgpu-install..."
    sudo apt purge -y amdgpu-install 2>/dev/null || echo "⚠️ No prior amdgpu-install to purge"
    rm -fv amdgpu-install_5.7.50701-1_all.deb*

    # Download and install
    echo "📦 Downloading amdgpu-install 5.7.1..."
    wget https://repo.radeon.com/amdgpu-install/5.7.1/ubuntu/jammy/amdgpu-install_5.7.50701-1_all.deb || { echo "❌ Error: Failed to download amdgpu-install—check network or URL"; exit 1; }
    sudo apt install -y ./amdgpu-install_5.7.50701-1_all.deb || { echo "❌ Error: Failed to install amdgpu-install"; exit 1; }
    sudo apt update || { echo "❌ Error: Apt update failed after installing amdgpu-install"; exit 1; }
}

resolve_dependencies() {
    echo "🔧 Resolving dependencies for rocm-gdb..."

    # Install libtinfo5 and libncurses5 from Ubuntu 22.04
    echo "📦 Downloading and installing libtinfo5 and libncurses5..."
    wget http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2_amd64.deb || { echo "❌ Error: Failed to download libtinfo5"; exit 1; }
    wget http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libncurses5_6.3-2_amd64.deb || { echo "❌ Error: Failed to download libncurses5"; exit 1; }
    sudo dpkg -i libtinfo5_6.3-2_amd64.deb libncurses5_6.3-2_amd64.deb || { echo "❌ Error: Failed to install ncurses libraries"; exit 1; }
    sudo apt install -f || { echo "❌ Error: Failed to fix dependencies after ncurses install"; exit 1; }

    # Install Python 3.10 via Deadsnakes PPA
    echo "📦 Adding Deadsnakes PPA for Python 3.10..."
    sudo add-apt-repository ppa:deadsnakes/ppa -y || { echo "❌ Error: Failed to add Deadsnakes PPA"; exit 1; }
    sudo apt update || { echo "❌ Error: Apt update failed after adding PPA"; exit 1; }
    sudo apt install -y python3.10 python3.10-minimal libpython3.10-stdlib libpython3.10 || { echo "❌ Error: Failed to install Python 3.10"; exit 1; }
}

install_rocm() {
    echo "🔨 Installing ROCm 5.7.1 with graphics, ROCm, HIP, and OpenCL..."

    # Run amdgpu-install
    sudo amdgpu-install --usecase=graphics,rocm,hip,opencl --no-dkms -y || { echo "❌ Error: ROCm installation failed"; exit 1; }

    # Update package cache
    sudo apt update || { echo "❌ Error: Final apt update failed"; exit 1; }
}

configure_environment() {
    echo "🔧 Configuring environment..."

    # Set library paths
    echo "📂 Setting library paths..."
    printf "/opt/rocm/lib\n/opt/rocm/lib64\n" | sudo tee /etc/ld.so.conf.d/rocm.conf || { echo "❌ Error: Failed to configure library paths"; exit 1; }
    sudo ldconfig || { echo "❌ Error: ldconfig failed"; exit 1; }

    # Set system-wide environment variable
    echo "🌐 Setting ROCM_PATH in /etc/environment..."
    echo "ROCM_PATH=/opt/rocm" | sudo tee -a /etc/environment || { echo "❌ Error: Failed to set ROCM_PATH"; exit 1; }

    # Append to user's .bashrc
    TARGET_USER="${SUDO_USER:-${LOGNAME:-heathen-admin}}"
    BASHRC="/home/$TARGET_USER/.bashrc"
    echo "📝 Adding ROCm variables to $BASHRC..."
    {
        echo "# ROCm 5.7.1 environment variables"
        echo "export ROCM_PATH=/opt/rocm"
        echo "export HIP_PATH=/opt/rocm/hip"
        echo "export PATH=\$ROCM_PATH/bin:\$HIP_PATH/bin:\$PATH"
        echo "export LD_LIBRARY_PATH=\$ROCM_PATH/lib:\$LD_LIBRARY_PATH"
    } | sudo tee -a "$BASHRC" || { echo "❌ Error: Failed to update .bashrc"; exit 1; }
    sudo -u "$TARGET_USER" bash -c "source $BASHRC" || echo "⚠️ Failed to source .bashrc (non-critical, requires logout)"
}

verify_installation() {
    echo "🔍 Verifying ROCm 5.7.1 installation..."
    export PATH="$PATH:/opt/rocm/bin:/opt/rocm/hip/bin"

    # Check rocminfo
    echo "Running rocminfo..."
    if ! rocminfo >/dev/null 2>&1; then
        echo "❌ Error: rocminfo failed"
    else
        echo "✅ Success: rocminfo executed"
        rocminfo | grep "Name:" -A 5
    fi

    # Check rocm-smi
    echo "Running rocm-smi..."
    if ! rocm-smi >/dev/null 2>&1; then
        echo "❌ Error: rocm-smi failed"
    else
        echo "✅ Success: rocm-smi executed"
    fi

    # Check hipcc
    echo "Running hipcc --version..."
    if ! hipcc --version >/dev/null 2>&1; then
        echo "❌ Error: hipcc failed"
    else
        echo "✅ Success: HIP compiler detected"
        hipcc --version
    fi

    # Check clinfo
    echo "Running clinfo..."
    if ! clinfo >/dev/null 2>&1; then
        echo "❌ Error: clinfo failed"
    else
        echo "✅ Success: OpenCL detected"
    fi

    # Check rocm-gdb
    echo "Running rocm-gdb --version..."
    if ! rocm-gdb --version >/dev/null 2>&1; then
        echo "❌ Error: rocm-gdb failed"
    else
        echo "✅ Success: ROCm debugger detected"
        rocm-gdb --version
    fi
}

main() {
    echo "🚀 Starting ROCm 5.7.1 Installation on Ubuntu 24.04.2 LTS..."
    check_root
    install_prerequisites
    install_amdgpu_installer
    resolve_dependencies
    install_rocm
    configure_environment
    echo "ℹ️ Note: Log out and back in to apply group changes."
    verify_installation
    echo "
✨ ROCm 5.7.1 Installation Complete! ✨
Commands to verify:
- rocminfo : GPU details
- rocm-smi : GPU status
- hipcc --version : HIP compiler version
- clinfo : OpenCL details
- rocm-gdb --version : Debugger version
Reboot recommended for full functionality.
"
}

# Trap errors
trap 'echo "❌ Script failed at line $LINENO with exit code $?"' ERR

main