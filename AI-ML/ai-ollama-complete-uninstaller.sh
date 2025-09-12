#!/bin/bash

# Ollama Full Uninstaller Script
# Version: 1.1 - Cortana (Grok 3, xAI) for Jason
# Date: February 26, 2025
# Removes ALL Ollama variants, preserves CUDA/ROCm

set -e

echo "🧹 Starting Full Ollama Uninstaller..."

check_root() {
    echo "Checking root privileges..."
    [ "$(id -u)" -eq 0 ] || { echo "Error: Run with sudo."; exit 1; }
    echo "Success: Root confirmed."
}

remove_services() {
    echo "⚙️ Stopping and removing all Ollama services..."
    for svc in ollama ollama-rocm ollama-cuda; do
        systemctl is-active --quiet "$svc" && sudo systemctl stop "$svc"
        systemctl is-enabled --quiet "$svc" && sudo systemctl disable "$svc"
        [ -f "/etc/systemd/system/$svc.service" ] && sudo rm -f "/etc/systemd/system/$svc.service"
    done
    sudo systemctl daemon-reload
    echo "✅ All Ollama services removed"
}

remove_binaries() {
    echo "🗑️ Removing all Ollama binaries..."
    [ -f "/ollama" ] && sudo rm -f /ollama && echo "✅ Removed /ollama"
    [ -f "/usr/bin/ollama" ] && sudo rm -f /usr/bin/ollama && echo "✅ Removed /usr/bin/ollama"
    [ -f "/usr/local/bin/ollama" ] && sudo rm -f /usr/local/bin/ollama && echo "✅ Removed /usr/local/bin/ollama"
    [ -f "/usr/local/bin/ollama-cuda" ] && sudo rm -f /usr/local/bin/ollama-cuda && echo "✅ Removed ollama-cuda"
    [ -f "/usr/local/bin/ollama-rocm" ] && sudo rm -f /usr/local/bin/ollama-rocm && echo "✅ Removed ollama-rocm"
    sudo find /usr/local/bin -name "*ollama*" -exec rm -f {} \; && echo "✅ Removed any stray ollama binaries"
}

remove_user_and_data() {
    echo "🗑️ Removing all Ollama data and users..."
    sudo find /usr/share -maxdepth 1 -name "ollama*" -exec rm -rf {} \; && echo "✅ Removed /usr/share/ollama* dirs"
    sudo find /var/lib -maxdepth 1 -name "ollama*" -exec rm -rf {} \; && echo "✅ Removed /var/lib/ollama* dirs"
    sudo find "$HOME" -maxdepth 1 -name "ollama*" -exec rm -rf {} \; && echo "✅ Removed $HOME/ollama* dirs"
    id -nG "$USER" | grep -qw "ollama" && sudo gpasswd -d "$USER" ollama
    id "ollama" &>/dev/null && sudo userdel -r ollama 2>/dev/null || true
    getent group ollama >/dev/null && sudo groupdel ollama 2>/dev/null || true
    echo "✅ Removed ollama user/group"
}

remove_build_dirs() {
    echo "🗑️ Cleaning up all Ollama build directories..."
    sudo find "$HOME" -maxdepth 1 -name "*ollama*" -exec rm -rf {} \; && echo "✅ Removed all $HOME/*ollama* dirs"
}

clean_environment() {
    echo "🧹 Cleaning up Ollama environment variables..."
    [ -f ~/.bashrc ] && cp ~/.bashrc ~/.bashrc.bak && sed -i '/ollama/Id' ~/.bashrc && echo "✅ Cleaned ~/.bashrc (backup at ~/.bashrc.bak)"
}

verify_cleanup() {
    echo "🔍 Verifying Ollama removal..."
    local found_items=false
    for path in "/ollama" "/usr/bin/ollama" "/usr/local/bin/ollama" "/usr/local/bin/ollama-cuda" "/usr/local/bin/ollama-rocm" \
               "/usr/share/ollama" "/usr/share/ollama-rocm" "/usr/share/ollama-cuda" "/var/lib/ollama" \
               "/etc/systemd/system/ollama.service" "/etc/systemd/system/ollama-rocm.service" "/etc/systemd/system/ollama-cuda.service"; do
        [ -e "$path" ] && echo "⚠️ Warning: $path still exists" && found_items=true
    done
    sudo find "$HOME" -maxdepth 1 -name "*ollama*" | grep -q . && echo "⚠️ Warning: Ollama build dirs remain in $HOME" && found_items=true
    id "ollama" &>/dev/null && echo "⚠️ Warning: ollama user still exists" && found_items=true
    getent group ollama >/dev/null && echo "⚠️ Warning: ollama group still exists" && found_items=true
    ! $found_items && echo "✅ All Ollama traces successfully removed"
}

main() {
    check_root
    echo "⚠️ This will remove ALL Ollama variants (standard, CPU, ROCm, CUDA)—CUDA/ROCm stay safe."
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        remove_services
        remove_binaries
        remove_user_and_data
        remove_build_dirs
        clean_environment
        verify_cleanup
        echo "
✨ Full Ollama Uninstallation Complete! ✨
- All services stopped/removed
- All binaries (ollama, ollama-rocm, ollama-cuda) gone
- User/data/build dirs wiped
- Environment scrubbed (backup at ~/.bashrc.bak)
CUDA and ROCm are untouched—ready for a fresh start!"
    else
        echo "Uninstallation cancelled."
    fi
}

main
