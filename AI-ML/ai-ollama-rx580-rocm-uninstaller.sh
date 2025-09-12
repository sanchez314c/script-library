#!/bin/bash

# Ollama ROCm Uninstaller Script
# Version: 1.0.0 - Built by Cortana (via Grok 3, xAI) for Jason
# Date: February 23, 2025
set -e

echo "🧹 Starting Ollama ROCm Uninstallation Process..."

# Global variables
OLLAMA_DIR="/home/$USER/ollama-rocm"
SERVICE_NAME="ollama-rocm"
ROCM_PATH="/opt/rocm"

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$EUID" -ne 0 ]; then
        echo "❌ This script requires root privileges. Please run with sudo."
        exit 1
    fi
    echo "✅ Running as root"
}

stop_and_remove_service() {
    echo "🔧 Stopping and removing Ollama ROCm service..."
    if systemctl is-active "$SERVICE_NAME" &>/dev/null; then
        sudo systemctl stop "$SERVICE_NAME"
    fi
    if systemctl is-enabled "$SERVICE_NAME" &>/dev/null; then
        sudo systemctl disable "$SERVICE_NAME"
    fi
    sudo rm -f "/etc/systemd/system/$SERVICE_NAME.service"
    sudo systemctl daemon-reload
    sudo systemctl reset-failed
    echo "✅ Service removed"
}

remove_installed_files() {
    echo "🗑️ Removing installed Ollama ROCm files..."
    sudo rm -f /usr/local/bin/ollama-rocm
    sudo rm -rf /etc/ollama-rocm /usr/share/ollama-rocm "$OLLAMA_DIR"
    echo "✅ Files removed"
}

remove_ollama_user() {
    echo "👤 Removing Ollama user and group..."
    if id ollama &>/dev/null; then
        sudo userdel -r ollama 2>/dev/null || true
        sudo groupdel ollama 2>/dev/null || true
    fi
    echo "✅ Ollama user and group removed"
}

cleanup_environment() {
    echo "🧼 Cleaning up environment variables..."
    # Note: This only affects the current session; persistent changes (e.g., in ~/.bashrc) aren't touched
    unset HSA_OVERRIDE_GFX_VERSION ROC_ENABLE_PRE_VEGA HSA_ENABLE_SDMA HIP_VISIBLE_DEVICES
    unset CGO_CFLAGS CGO_LDFLAGS GOFLAGS
    echo "✅ Environment variables unset for this session"
}

verify_removal() {
    echo "🔍 Verifying uninstallation..."
    if [ -f /usr/local/bin/ollama-rocm ] || [ -d /etc/ollama-rocm ] || [ -d "$OLLAMA_DIR" ]; then
        echo "⚠️ Warning: Some Ollama files remain. Manual cleanup may be needed."
    else
        echo "✅ No Ollama ROCm files detected"
    fi
    if systemctl list-unit-files | grep -q "$SERVICE_NAME"; then
        echo "⚠️ Warning: Service still listed in systemd."
    else
        echo "✅ Service fully removed from systemd"
    fi
    if id ollama &>/dev/null; then
        echo "⚠️ Warning: Ollama user still exists."
    else
        echo "✅ Ollama user fully removed"
    fi
}

main() {
    check_root
    stop_and_remove_service
    remove_installed_files
    remove_ollama_user
    cleanup_environment
    verify_removal

    echo "✨ Ollama ROCm Uninstallation Complete! ✨"
    echo "System is reset to pre-install state for Ollama ROCm."
    echo "Notes:"
    echo "- Prerequisites (e.g., Go, ROCm, Conda) remain installed."
    echo "- Conda environment 'darkpool-rocm' is untouched."
    echo "- Run 'rocm-smi' to verify GPU status if needed."
    echo "Ready for a fresh reinstall whenever you are, J!"
}

main
