#!/bin/bash
# NVIDIA Driver Installation Script
# Version: 1.1.2
# Date: March 21, 2025
# Author: Grok 3 (xAI) for heathen-admin
# Description: Installs NVIDIA driver 470 for K80 GPUs with best practices

set -e
set -x

LOG_FILE="/home/heathen-admin/nvidia_driver_setup.log"

echo "🚀 Starting NVIDIA Driver 470 Installation..." | tee -a "$LOG_FILE"

check_root() {
    echo "🔍 Checking for root privileges..." | tee -a "$LOG_FILE"
    [ "$(id -u)" != "0" ] && { echo "❌ Requires root. Run with sudo." | tee -a "$LOG_FILE"; exit 1; }
    echo "✅ Running as root" | tee -a "$LOG_FILE"
}

remove_existing_nvidia() {
    echo "🧹 Removing existing NVIDIA installations..." | tee -a "$LOG_FILE"
    if dpkg -l | grep -q "nvidia-driver-470"; then
        echo "✅ NVIDIA 470 already installed—skipping removal" | tee -a "$LOG_FILE"
        return
    fi
    apt-get remove --purge -y '^nvidia-.*' || true
    if command -v nvidia-uninstall &>/dev/null; then
        echo "📥 Removing NVIDIA .run driver installation..." | tee -a "$LOG_FILE"
        nvidia-uninstall --silent || true
    fi
    apt-get autoremove -y
    apt-get clean
    echo "✅ Existing NVIDIA installations removed" | tee -a "$LOG_FILE"
}

install_dependencies() {
    echo "📦 Installing driver-specific dependencies..." | tee -a "$LOG_FILE"
    apt update
    apt install -y dkms libglvnd-dev linux-headers-$(uname -r) || { echo "❌ Dependency install failed" | tee -a "$LOG_FILE"; exit 1; }
    echo "✅ Dependencies installed" | tee -a "$LOG_FILE"
}

add_nvidia_repo() {
    echo "🌐 Adding NVIDIA driver repository..." | tee -a "$LOG_FILE"
    if ! grep -r "ppa:graphics-drivers/ppa" /etc/apt/sources.list /etc/apt/sources.list.d/ &>/dev/null; then
        add-apt-repository -y ppa:graphics-drivers/ppa || { echo "❌ PPA add failed—check network" | tee -a "$LOG_FILE"; exit 1; }
    fi
    apt update || { echo "❌ Apt update failed" | tee -a "$LOG_FILE"; exit 1; }
    echo "✅ NVIDIA PPA added" | tee -a "$LOG_FILE"
}

install_nvidia_driver() {
    echo "📥 Installing NVIDIA driver 470..." | tee -a "$LOG_FILE"
    apt install -y nvidia-driver-470 || { echo "❌ Driver install failed" | tee -a "$LOG_FILE"; exit 1; }
    echo "✅ NVIDIA driver 470 installed" | tee -a "$LOG_FILE"
}

verify_installation() {
    echo "🔍 Verifying NVIDIA driver installation..." | tee -a "$LOG_FILE"
    echo "Loading NVIDIA module..." | tee -a "$LOG_FILE"
    modprobe nvidia || echo "⚠️ Module load may require reboot—continuing" | tee -a "$LOG_FILE"
    echo "Running nvidia-smi..." | tee -a "$LOG_FILE"
    nvidia-smi || { echo "❌ nvidia-smi failed—reboot and retry" | tee -a "$LOG_FILE"; exit 1; }
    echo "✅ Driver verified" | tee -a "$LOG_FILE"
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
- Reboot required before CUDA install or full verification
" | tee -a "$LOG_FILE"
}

main