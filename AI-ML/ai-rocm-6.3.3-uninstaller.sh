#!/bin/bash

# Enable verbose output and exit on error
set -x  # Print commands as they execute
set -e  # Exit on any error

# ROCm 6.3.3 Removal Script
check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$EUID" -ne 0 ]; then
        echo "❌ Error: Requires root privileges. Run with sudo."
        exit 1
    fi
    echo "✅ Success: Running as root"
}

remove_rocm() {
    echo "🧹 Starting ROCm 6.3.3 removal..."

    # Uninstall ROCm packages and amdgpu-install
    echo "📦 Uninstalling ROCm and AMDGPU packages..."
    sudo amdgpu-install --uninstall || echo "⚠️ amdgpu-install uninstall failed (continuing)"
    sudo apt remove --purge -y 'rocm*' 'amdgpu-install' || echo "⚠️ No ROCm packages found"
    sudo apt autoremove -y

    # Remove ROCm repo and GPG key
    echo "🗑️ Removing ROCm repo and key..."
    sudo rm -fv /etc/apt/sources.list.d/rocm.list
    sudo rm -fv /etc/apt/keyrings/rocm.gpg
    sudo apt update || echo "⚠️ Apt update failed (non-critical)"

    # Clean up configuration files
    echo "🧹 Cleaning ROCm directories and configs..."
    sudo rm -rfv /opt/rocm-6.3.3 /opt/rocm
    sudo rm -fv /etc/ld.so.conf.d/rocm.conf
    sudo ldconfig

    # Remove environment variable
    echo "🔧 Removing ROCm environment variable..."
    sudo sed -i '/ROCM_PATH=\/opt\/rocm-6.3.3/d' /etc/environment || echo "⚠️ ROCM_PATH not found in /etc/environment"

    # Revert .bashrc
    echo "🔧 Reverting .bashrc..."
    if grep -q "/opt/rocm-6.3.3/bin" ~/.bashrc; then
        sed -i '/rocm-6.3.3\/bin/d' ~/.bashrc
        echo "✅ Removed ROCm path from .bashrc"
    else
        echo "⚠️ ROCm path not found in .bashrc"
    fi

    # Refresh group membership (optional cleanup)
    echo "🔄 Refreshing group membership..."
    sg video -c "echo '✅ Group membership refreshed'" || echo "⚠️ Group refresh failed (non-critical)"

    echo "✅ Success: ROCm 6.3.3 removed"
}

verify_removal() {
    echo "🔍 Verifying ROCm removal..."
    if [ -d "/opt/rocm-6.3.3" ] || [ -f "/etc/ld.so.conf.d/rocm.conf" ]; then
        echo "❌ Warning: Residual ROCm files remain"
    else
        echo "✅ No residual ROCm 6.3.3 files detected"
    fi
    if command -v rocminfo >/dev/null 2>&1; then
        echo "❌ Warning: rocminfo still accessible—check PATH"
    else
        echo "✅ rocminfo not found"
    fi
}

main() {
    echo "🔧 Entering main function..."
    check_root
    remove_rocm
    verify_removal
    echo "
✨ ROCm 6.3.3 Removal Complete! ✨
Your system should now use Mesa drivers for the RX580.
"
}

# Trap errors and print a summary
trap 'echo "❌ Script failed at line $LINENO with exit code $?"' ERR

main
