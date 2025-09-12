#!/bin/bash

# NVIDIA Driver Installation Script
# Version: 1.1.0 - Built by Cortana (via Grok 3, xAI) for Jason
# Date: February 23, 2025
set -e

echo "🚀 Starting NVIDIA Driver 470 Installation..."

check_root() {
    echo "🔍 Checking for root privileges..."
    [ "$(id -u)" != "0" ] && { echo "❌ Requires root. Run with sudo."; exit 1; }
    echo "✅ Running as root"
}

remove_existing_nvidia() {
    echo "🧹 Removing existing NVIDIA installations..."
    apt-get remove --purge -y '^nvidia-.*' || true
    if command -v nvidia-uninstall &>/dev/null; then
        echo "📥 Removing NVIDIA .run driver installation..."
        nvidia-uninstall --silent || true
    fi
    apt-get autoremove -y
    apt-get clean
    echo "✅ Existing NVIDIA installations removed"
}

install_dependencies() {
    echo "📦 Installing driver-specific dependencies..."
    apt update
    apt install -y dkms libglvnd-dev
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

verify_installation() {
    echo "🔍 Verifying NVIDIA driver installation..."
    export PATH="$PATH:/usr/bin:/usr/local/bin"  # Ensure nvidia-smi is found
    echo "Running nvidia-smi..."
    nvidia-smi || { echo "❌ nvidia-smi failed—reboot and retry"; exit 1; }
    echo "✅ Driver verified"
}

main() {
    check_root
    remove_existing_nvidia
    install_dependencies
    add_nvidia_repo
    install_nvidia_driver
    verify_installation
    echo "
✨ NVIDIA Driver 470 Installation Complete! ✨
- Driver 470 installed for K80 GPUs
Commands:
- nvidia-smi : Check GPU status
Notes:
- Reboot recommended before CUDA install
"
}

main
