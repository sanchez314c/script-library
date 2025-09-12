#!/bin/bash

# CUDA Uninstaller Script for CUDA 11.4
# Version: 1.1.2 - Built by Grok 3 (xAI) for Jason
# Date: March 2, 2025
set -e
set -x

LOG_FILE="/home/${SUDO_USER:-$USER}/cuda_uninstall.log"
echo "🗑️ Starting CUDA 11.4 Uninstallation..." | tee -a "$LOG_FILE"

check_root() {
    echo "🔍 Checking for root privileges..." | tee -a "$LOG_FILE"
    if [ "$EUID" -ne 0 ]; then
        echo "❌ Error: Script requires root privileges. Run with sudo." | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "✅ Success: Running as root" | tee -a "$LOG_FILE"
}

remove_cuda() {
    echo "🧹 Removing CUDA 11.4..." | tee -a "$LOG_FILE"
    # Remove CUDA toolkit directory
    if [ -d "/usr/local/cuda-11.4" ]; then
        rm -rfv /usr/local/cuda-11.4 || { echo "❌ Error: Failed to remove /usr/local/cuda-11.4" | tee -a "$LOG_FILE"; exit 1; }
        echo "✅ CUDA 11.4 directory removed" | tee -a "$LOG_FILE"
    else
        echo "⚠️ CUDA 11.4 directory not found" | tee -a "$LOG_FILE"
    fi
    
    # Remove symbolic link if it exists
    if [ -L "/usr/local/cuda" ] && [ "$(readlink /usr/local/cuda)" = "/usr/local/cuda-11.4" ]; then
        rm -fv /usr/local/cuda || { echo "❌ Error: Failed to remove /usr/local/cuda symlink" | tee -a "$LOG_FILE"; exit 1; }
        echo "✅ CUDA symlink removed" | tee -a "$LOG_FILE"
    fi
    
    # Remove ld.so.conf.d entry
    if [ -f "/etc/ld.so.conf.d/cuda-11.4.conf" ]; then
        rm -fv /etc/ld.so.conf.d/cuda-11.4.conf || { echo "❌ Error: Failed to remove cuda-11.4.conf" | tee -a "$LOG_FILE"; exit 1; }
        ldconfig || { echo "❌ Error: ldconfig failed after removal" | tee -a "$LOG_FILE"; exit 1; }
        echo "✅ Library path configuration removed" | tee -a "$LOG_FILE"
    else
        echo "⚠️ cuda-11.4.conf not found" | tee -a "$LOG_FILE"
    fi
    
    echo "✅ CUDA 11.4 removal complete" | tee -a "$LOG_FILE"
}

clean_environment() {
    echo "🧹 Cleaning environment variables..." | tee -a "$LOG_FILE"
    # Remove CUDA_HOME from /etc/environment
    if grep -q "CUDA_HOME=/usr/local/cuda-11.4" /etc/environment; then
        sed -i '/CUDA_HOME=\/usr\/local\/cuda-11.4/d' /etc/environment || { echo "❌ Error: Failed to remove CUDA_HOME from /etc/environment" | tee -a "$LOG_FILE"; exit 1; }
        echo "✅ CUDA_HOME removed from /etc/environment" | tee -a "$LOG_FILE"
    else
        echo "⚠️ CUDA_HOME not found in /etc/environment" | tee -a "$LOG_FILE"
    fi
    
    # Remove PATH entry from ~/.bashrc
    if [ -f "$HOME/.bashrc" ] && grep -q "/usr/local/cuda-11.4/bin" "$HOME/.bashrc"; then
        cp "$HOME/.bashrc" "$HOME/.bashrc.bak.$(date +%Y%m%d_%H%M%S)" || { echo "❌ Error: Failed to backup .bashrc" | tee -a "$LOG_FILE"; exit 1; }
        sed -i 's|export PATH="$PATH:/usr/local/cuda-11.4/bin"|# Removed CUDA 11.4 PATH|g' "$HOME/.bashrc" || { echo "❌ Error: Failed to clean .bashrc" | tee -a "$LOG_FILE"; exit 1; }
        echo "✅ PATH cleaned from ~/.bashrc (backup in ~/.bashrc.bak.*)" | tee -a "$LOG_FILE"
    else
        echo "⚠️ CUDA PATH not found in ~/.bashrc or file missing" | tee -a "$LOG_FILE"
    fi
    
    echo "✅ Environment cleanup complete" | tee -a "$LOG_FILE"
}

remove_dependencies() {
    echo "🧹 Removing installed dependencies..." | tee -a "$LOG_FILE"
    apt-get remove --purge -y linux-headers-$(uname -r) build-essential gcc-10 g++-10 2>/dev/null || echo "⚠️ Some dependencies not found or already removed" | tee -a "$LOG_FILE"
    apt-get autoremove -y || true
    apt-get clean || { echo "❌ Error: Apt clean failed" | tee -a "$LOG_FILE"; exit 1; }
    
    # Reset GCC alternatives if modified
    if update-alternatives --get-selections | grep -q "gcc.*gcc-10"; then
        update-alternatives --remove gcc /usr/bin/gcc-10 || echo "⚠️ Failed to remove gcc-10 alternative" | tee -a "$LOG_FILE"
        update-alternatives --remove g++ /usr/bin/g++-10 || echo "⚠️ Failed to remove g++-10 alternative" | tee -a "$LOG_FILE"
        echo "✅ GCC/G++ alternatives reset" | tee -a "$LOG_FILE"
    else
        echo "⚠️ No GCC-10 alternatives to reset" | tee -a "$LOG_FILE"
    fi
    
    echo "✅ Dependencies removal complete" | tee -a "$LOG_FILE"
}

verify_removal() {
    echo "🔍 Verifying CUDA removal..." | tee -a "$LOG_FILE"
    [ -d "/usr/local/cuda-11.4" ] && echo "⚠️ CUDA 11.4 directory still exists" | tee -a "$LOG_FILE" || echo "✅ CUDA 11.4 directory gone" | tee -a "$LOG_FILE"
    command -v nvcc >/dev/null 2>&1 && echo "⚠️ nvcc still available in PATH" | tee -a "$LOG_FILE" || echo "✅ nvcc not found" | tee -a "$LOG_FILE"
    grep -q "/usr/local/cuda-11.4" /etc/environment && echo "⚠️ CUDA_HOME still in /etc/environment" | tee -a "$LOG_FILE" || echo "✅ CUDA_HOME removed from /etc/environment" | tee -a "$LOG_FILE"
    [ -f "$HOME/.bashrc" ] && grep -q "/usr/local/cuda-11.4/bin" "$HOME/.bashrc" && echo "⚠️ CUDA PATH still in ~/.bashrc" | tee -a "$LOG_FILE" || echo "✅ CUDA PATH removed from ~/.bashrc" | tee -a "$LOG_FILE"
    echo "✅ Verification complete" | tee -a "$LOG_FILE"
}

main() {
    check_root
    remove_cuda
    clean_environment
    remove_dependencies
    verify_removal
    echo "
🗑️ CUDA 11.4 Uninstallation Complete!
- CUDA 11.4 removed from /usr/local/cuda-11.4
- Environment variables cleaned (/etc/environment, ~/.bashrc)
- Dependencies removed
- Log: $LOG_FILE
- Note: Reboot recommended to clear any loaded libraries
" | tee -a "$LOG_FILE"
}

main
