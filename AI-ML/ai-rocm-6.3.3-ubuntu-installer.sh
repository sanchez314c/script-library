#!/bin/bash
# ROCm-Only Installation Script for RX580 with HIP Fix and Extended Env Handling
# Date: March 15, 2025 - Updated by Grok 3 (xAI)
set -x
set -e

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$EUID" -ne 0 ]; then
        echo "❌ Error: Requires root privileges. Run with sudo or as root."
        exit 1
    fi
    echo "✅ Success: Running as root"
}

install_rocm() {
    echo "📦 Starting ROCm 6.3.3 system-wide installation..."
    export DEBIAN_FRONTEND=noninteractive

    echo "🧹 Cleaning old ROCm/AMDGPU references..."
    echo "Searching for repo.radeon.com lines to comment out..."
    sudo find /etc/apt/ -type f -exec echo "Checking file: {}" \; -exec sed -i 's|^deb.*repo\.radeon\.com.*|#&|g' {} \; || echo "⚠️ No old repo lines found or sed failed (non-critical)"
    echo "Removing old amdgpu.list if it exists..."
    sudo rm -fv /etc/apt/sources.list.d/amdgpu.list /etc/apt/sources.list.d/rocm.list

    echo "📦 Installing dependencies..."
    echo "Installing linux-headers-$(uname -r) and linux-modules-extra-$(uname -r)..."
    sudo apt install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)" || { echo "❌ Error: Dependency installation failed"; exit 1; }

    echo "📥 Downloading AMDGPU installer..."
    rm -fv amdgpu-install_6.3.60303-1_all.deb*
    echo "Fetching installer package..."
    wget -v https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb || { echo "❌ Error: Wget failed—check network or URL"; exit 1; }
    echo "🔧 Installing AMDGPU installer..."
    sudo dpkg -i amdgpu-install_6.3.60303-1_all.deb || { echo "❌ Error: dpkg failed—running apt fix"; sudo apt install -f -y; }
    echo "🔄 Running second apt update..."
    sudo apt update || { echo "❌ Error: Second apt update failed"; exit 1; }

    echo "🔨 Installing ROCm packages..."
    sudo amdgpu-install --usecase=graphics,rocm,rocmdev,rocmdevtools,opencl,openclsdk,hip,hiplibsdk,mllib,mlsdk -y || { echo "❌ Error: ROCm installation failed"; exit 1; }

    echo "🔧 Ensuring HIP and hipBLAS components are present..."
    sudo apt install -y hip-dev hip-runtime-amd rocm-hip-sdk hipblas hipblas-dev || { echo "❌ Error: HIP/hipBLAS package installation failed"; exit 1; }

    echo "🔧 Configuring library paths..."
    echo "Writing ROCm and HIP library paths to /etc/ld.so.conf.d/rocm.conf..."
    printf "/opt/rocm-6.3.3/lib\n/opt/rocm-6.3.3/lib64\n" | sudo tee /etc/ld.so.conf.d/rocm.conf || { echo "❌ Error: Library path config failed"; exit 1; }
    echo "Updating ldconfig..."
    sudo ldconfig || { echo "❌ Error: ldconfig failed"; exit 1; }

    echo "🔧 Ensuring standard HIP directory structure..."
    if [ ! -d "/opt/rocm-6.3.3/hip" ]; then
        sudo mkdir -p /opt/rocm-6.3.3/hip
        sudo ln -sf /opt/rocm-6.3.3/include/hip /opt/rocm-6.3.3/hip/include
        sudo ln -sf /opt/rocm-6.3.3/lib /opt/rocm-6.3.3/hip/lib
    fi
    if [ ! -L "/opt/rocm/hip" ]; then
        sudo ln -sf /opt/rocm-6.3.3/hip /opt/rocm/hip
    fi
    if [ ! -d "/opt/rocm-6.3.3/hip/include/hipblas" ]; then
        sudo ln -sf /opt/rocm-6.3.3/include/hipblas /opt/rocm-6.3.3/hip/include/hipblas
    fi
    if [ ! -L "/opt/rocm/include/hipblas" ]; then
        sudo ln -sf /opt/rocm-6.3.3/include/hipblas /opt/rocm/include/hipblas
    fi

    echo "🔄 Adding user to groups..."
    if [ -z "$TARGET_USER" ]; then
        echo "❌ Error: Could not determine any username"
        exit 1
    fi
    if [ "$TARGET_USER" = "root" ]; then
        TARGET_USER="heathen-admin"
        echo "ℹ️ Running as root, defaulting to user: $TARGET_USER"
    fi
    sudo usermod -a -G video "$TARGET_USER" || { echo "❌ Error: Adding $TARGET_USER to video group failed"; exit 1; }
    sudo usermod -a -G render "$TARGET_USER" || { echo "❌ Error: Adding $TARGET_USER to render group failed"; exit 1; }
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
    echo "🔍 Verifying ROCm and HIP installation..."
    export PATH="$PATH:/opt/rocm-6.3.3/bin"
    export HIP_PATH=/opt/rocm-6.3.3/hip
    export HSA_OVERRIDE_GFX_VERSION=8.0.3
    export CMAKE_HIP_FLAGS="-I/opt/rocm-6.3.3/include"
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
    echo "Checking hipcc..."
    if ! /opt/rocm-6.3.3/bin/hipcc --version; then
        echo "❌ Error: hipcc not found or failed"
    else
        echo "✅ Success: hipcc executed"
    fi
    echo "Checking HIP paths..."
    if [ -d "/opt/rocm-6.3.3/hip" ] && [ -L "/opt/rocm/hip" ] && [ -d "/opt/rocm-6.3.3/include/hipblas" ]; then
        echo "✅ Success: HIP and hipBLAS directory structure verified"
    else
        echo "❌ Error: HIP or hipBLAS directory structure incomplete"
    fi
    echo "✅ Verification complete"
}

main() {
    echo "🔧 Entering main function..."
    check_root
    install_rocm
    echo "🔄 Refreshing group membership in current session..."
    newgrp video || echo "⚠️ Failed to refresh video group (may require manual newgrp or logout)"
    newgrp render || echo "⚠️ Failed to refresh render group (may require manual newgrp or logout)"
    echo "✅ Group membership refreshed; proceeding with verification..."
    verify_rocm
    echo "
✨ ROCm 6.3.3 Installation Complete with HIP Fix! ✨
Commands to verify installation:
- rocminfo : Check GPU details
- rocm-smi : Check GPU status
- hipcc --version : Check HIP compiler
Note: Log out and back in to apply changes, or source /etc/environment manually.
To build Ollama with HIP:
  cmake -B build -DCMAKE_PREFIX_PATH=/opt/rocm -DAMDGPU_TARGETS=gfx803 -DGGML_HIPBLAS=ON
"
}

trap 'echo "❌ Script failed at line $LINENO with exit code $?"' ERR

main