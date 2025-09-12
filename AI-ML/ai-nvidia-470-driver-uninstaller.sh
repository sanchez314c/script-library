#!/bin/bash

# NVIDIA Driver Uninstaller Script
# Version: 1.1.0 - Built by Grok 3 (xAI) for Jason
# Date: March 2, 2025
set -e
set -x

LOG_FILE="/home/${SUDO_USER:-$USER}/nvidia_driver_uninstall.log"
echo "🗑️ Starting NVIDIA Driver 470 Uninstallation..." | tee -a "$LOG_FILE"

check_root() {
    echo "🔍 Checking for root privileges..." | tee -a "$LOG_FILE"
    [ "$(id -u)" != "0" ] && { echo "❌ Requires root. Run with sudo." | tee -a "$LOG_FILE"; exit 1; }
    echo "✅ Running as root" | tee -a "$LOG_FILE"
}

remove_nvidia_driver() {
    echo "🧹 Removing NVIDIA driver 470..." | tee -a "$LOG_FILE"
    apt-get remove --purge -y nvidia-driver-470 2>/dev/null || echo "⚠️ NVIDIA driver 470 not found or already removed" | tee -a "$LOG_FILE"
    apt-get autoremove -y || true
    apt-get clean
    echo "✅ NVIDIA driver 470 removed" | tee -a "$LOG_FILE"
}

remove_nvidia_repo() {
    echo "🌐 Removing NVIDIA driver repository..." | tee -a "$LOG_FILE"
    add-apt-repository -y -r ppa:graphics-drivers/ppa 2>/dev/null || echo "⚠️ NVIDIA PPA not found or already removed" | tee -a "$LOG_FILE"
    apt update || { echo "❌ Apt update failed" | tee -a "$LOG_FILE"; exit 1; }
    echo "✅ NVIDIA PPA removed" | tee -a "$LOG_FILE"
}

remove_dependencies() {
    echo "🧹 Removing driver-specific dependencies..." | tee -a "$LOG_FILE"
    apt-get remove --purge -y dkms libglvnd-dev 2>/dev/null || echo "⚠️ Dependencies not found or already removed" | tee -a "$LOG_FILE"
    apt-get autoremove -y || true
    echo "✅ Dependencies removed" | tee -a "$LOG_FILE"
}

verify_removal() {
    echo "🔍 Verifying NVIDIA driver removal..." | tee -a "$LOG_FILE"
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi; then
        echo "⚠️ NVIDIA driver still active—reboot may be required" | tee -a "$LOG_FILE"
    else
        echo "✅ No NVIDIA driver detected" | tee -a "$LOG_FILE"
    fi
    dpkg -l | grep -q nvidia-driver-470 && echo "⚠️ NVIDIA driver 470 packages still installed" | tee -a "$LOG_FILE" || echo "✅ NVIDIA driver 470 packages gone" | tee -a "$LOG_FILE"
    grep -r "graphics-drivers" /etc/apt/sources.list.d/* >/dev/null 2>&1 && echo "⚠️ NVIDIA PPA still in sources" | tee -a "$LOG_FILE" || echo "✅ NVIDIA PPA removed from sources" | tee -a "$LOG_FILE"
    echo "✅ Verification complete" | tee -a "$LOG_FILE"
}

main() {
    check_root
    remove_nvidia_driver
    remove_nvidia_repo
    remove_dependencies
    verify_removal
    echo "
🗑️ NVIDIA Driver 470 Uninstallation Complete!
- Driver 470 removed
- NVIDIA PPA removed
- Dependencies cleaned up
- Log: $LOG_FILE
- Note: Reboot recommended to ensure driver unloading
" | tee -a "$LOG_FILE"
}

main
