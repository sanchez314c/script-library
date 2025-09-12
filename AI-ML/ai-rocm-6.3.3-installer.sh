#!/bin/bash

# Enable verbose output and exit on error
set -x  # Print commands as they execute
set -e  # Exit on any error

# ROCm-Only Installation Script for RX580
check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$EUID" -ne 0 ]; then
        echo "❌ Error: Requires root privileges. Run with sudo."
        exit 1
    fi
    echo "✅ Success: Running as root"
}

install_rocm() {
    echo "📦 Starting ROCm 6.3.3 system-wide installation..."

    # Clean old ROCm and AMDGPU repo lines
    echo "🧹 Cleaning old ROCm/AMDGPU references..."
    echo "Searching for repo.radeon.com lines to comment out..."
    sudo find /etc/apt/ -type f -exec echo "Checking file: {}" \; -exec sed -i 's|^deb.*repo\.radeon\.com.*|#&|g' {} \; || echo "⚠️ No old repo lines found or sed failed (non-critical)"
    echo "Removing old amdgpu.list if it exists..."
    sudo rm -fv /etc/apt/sources.list.d/amdgpu.list

    # Install minimal dependencies
    echo "📦 Installing dependencies..."
    echo "Installing linux-headers-$(uname -r) and linux-modules-extra-$(uname -r)..."
    sudo apt install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)" || { echo "❌ Error: Dependency installation failed"; exit 1; }

    # Add ROCm 6.3.3 repo with GPG key
    echo "🔑 Setting up ROCm GPG key and repo..."
    sudo mkdir -pv /etc/apt/keyrings
    echo "Fetching ROCm GPG key..."
    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg --yes || { echo "❌ Error: GPG key setup failed"; exit 1; }
    echo "Writing ROCm repo to /etc/apt/sources.list.d/rocm.list..."
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.3.3 noble main" | sudo tee /etc/apt/sources.list.d/rocm.list || { echo "❌ Error: Repo file creation failed"; exit 1; }
    echo "Cleaning apt cache..."
    sudo apt clean
    echo "🔄 Running apt update..."
    sudo apt update || { echo "❌ Error: Apt update failed—check network or repo availability"; exit 1; }

    # Install AMDGPU installer
    echo "📥 Downloading AMDGPU installer..."
    rm -fv amdgpu-install_6.3.60303-1_all.deb*
    echo "Fetching installer package..."
    wget -v https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb || { echo "❌ Error: Wget failed—check network or URL"; exit 1; }
    echo "🔧 Installing AMDGPU installer..."
    sudo dpkg -i amdgpu-install_6.3.60303-1_all.deb || { echo "❌ Error: dpkg failed—running apt fix"; sudo apt install -f -y; }
    echo "🔄 Running second apt update..."
    sudo apt update || { echo "❌ Error: Second apt update failed"; exit 1; }

    # Install ROCm with specific use cases
    echo "🔨 Installing ROCm packages..."
    sudo amdgpu-install --usecase=graphics,rocm,rocmdev,rocmdevtools,opencl,openclsdk,hip,hiplibsdk,mllib,mlsdk -y || { echo "❌ Error: ROCm installation failed"; exit 1; }

    # Configure library paths
    echo "🔧 Configuring library paths..."
    echo "Writing ROCm library paths to /etc/ld.so.conf.d/rocm.conf..."
    printf "/opt/rocm-6.3.3/lib\n/opt/rocm-6.3.3/lib64\n" | sudo tee /etc/ld.so.conf.d/rocm.conf || { echo "❌ Error: Library path config failed"; exit 1; }
    echo "Updating ldconfig..."
    sudo ldconfig || { echo "❌ Error: ldconfig failed"; exit 1; }

    # Set minimal system-wide environment variable
    echo "🔧 Setting environment variable..."
    echo "ROCM_PATH=/opt/rocm-6.3.3" | sudo tee -a /etc/environment || { echo "❌ Error: Environment variable setup failed"; exit 1; }

    # Add user to groups and set device permissions
    echo "🔄 Adding user to groups..."
    # Determine the correct username: prefer SUDO_USER, then LOGNAME, then whoami
    TARGET_USER="${SUDO_USER:-${LOGNAME:-$(whoami)}}"
    echo "ℹ️ Target user identified as: $TARGET_USER"
    if [ -z "$TARGET_USER" ] || [ "$TARGET_USER" = "root" ]; then
        echo "❌ Error: Could not determine non-root username (got: $TARGET_USER)"
        exit 1
    fi
    # Add to video and render groups
    sudo usermod -a -G video "$TARGET_USER" || { echo "❌ Error: Adding $TARGET_USER to video group failed"; exit 1; }
    sudo usermod -a -G render "$TARGET_USER" || { echo "❌ Error: Adding $TARGET_USER to render group failed"; exit 1; }
    # Verify group membership
    echo "🔍 Verifying group membership..."
    if grep -E "video.*$TARGET_USER" /etc/group && grep -E "render.*$TARGET_USER" /etc/group; then
        echo "✅ Success: $TARGET_USER added to video and render groups"
    else
        echo "❌ Error: $TARGET_USER not found in video or render groups"
        echo "Debug: Current /etc/group entries:"
        grep -E 'render|video' /etc/group
        exit 1
    fi
    echo "🔧 Setting device permissions..."
    [ -e /dev/kfd ] && sudo chmod 660 /dev/kfd && sudo chown root:render /dev/kfd || echo "⚠️ /dev/kfd not found (may appear after reboot)"
    for dev in /dev/dri/card* /dev/dri/render*; do
        [ -e "$dev" ] && sudo chmod 660 "$dev" && sudo chown root:render "$dev"
    done
    echo "ℹ️ Note: Group changes applied; refreshing session next."
}

verify_rocm() {
    echo "🔍 Verifying ROCm installation..."
    export PATH="$PATH:/opt/rocm-6.3.3/bin"
    echo "Running rocminfo..."
    if ! rocminfo | grep -A 5 "Name:.*gfx803"; then
        echo "❌ Error: rocminfo failed or RX580 (gfx803) not detected"
    else
        echo "✅ Success: rocminfo detected RX580"
    fi
    echo "Running rocm-smi..."
    if ! rocm-smi; then
        echo "❌ Error: rocm-smi failed"
    else
        echo "✅ Success: rocm-smi executed"
    fi
    echo "✅ Verification complete"
}

main() {
    echo "🔧 Entering main function..."
    check_root
    install_rocm
    # Refresh group membership before verification
    echo "🔄 Refreshing group membership in current session..."
    newgrp video || echo "⚠️ Failed to refresh video group (may require manual newgrp or logout)"
    newgrp render || echo "⚠️ Failed to refresh render group (may require manual newgrp or logout)"
    echo "✅ Group membership refreshed; proceeding with verification..."
    verify_rocm
    echo "
✨ ROCm 6.3.3 Installation Complete! ✨
Commands to verify installation:
- rocminfo : Check GPU details
- rocm-smi : Check GPU status
"
}

# Trap errors and print a summary
trap 'echo "❌ Script failed at line $LINENO with exit code $?"' ERR

main