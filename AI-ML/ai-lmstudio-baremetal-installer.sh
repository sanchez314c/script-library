#!/bin/bash

# LM Studio Bare-Metal Install Script
# Version: 1.0.0 - Built by Grok 3 (xAI) for Jason
# Date: March 2, 2025

# Enable verbosity and error handling
set -x  # Trace every command
set -e  # Exit on any error

echo "🚀 Starting LM Studio Install..."

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "❌ Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    echo "✅ Success: Running as root"
}

install_prerequisites() {
    echo "🛠️ Installing system prerequisites..."
    sudo apt update
    sudo apt install -y libfuse2  # Required for AppImage support
}

download_lmstudio() {
    echo "📥 Downloading LM Studio AppImage..."
    cd /tmp
    sudo -u "${SUDO_USER:-$USER}" wget -q -O LM-Studio-0.3.10-6-x64.AppImage \
        https://installers.lmstudio.ai/linux/x64/0.3.10-6/LM-Studio-0.3.10-6-x64.AppImage || {
        echo "❌ Error: Failed to download LM Studio AppImage"
        exit 1
    }
    echo "✅ Success: LM Studio AppImage downloaded to /tmp"
}

install_lmstudio() {
    echo "📦 Installing LM Studio to ~/AI/LMStudio..."
    INSTALL_DIR="/home/${SUDO_USER:-$USER}/AI/LMStudio"
    sudo -u "${SUDO_USER:-$USER}" mkdir -p "$INSTALL_DIR"
    sudo -u "${SUDO_USER:-$USER}" mv /tmp/LM-Studio-0.3.10-6-x64.AppImage "$INSTALL_DIR/LM-Studio.AppImage"
    sudo -u "${SUDO_USER:-$USER}" chmod +x "$INSTALL_DIR/LM-Studio.AppImage"
    echo "✅ Success: LM Studio installed to $INSTALL_DIR"
}

create_desktop_entry() {
    echo "📝 Creating desktop entry for LM Studio..."
    DESKTOP_FILE="/usr/share/applications/lm-studio.desktop"
    sudo tee "$DESKTOP_FILE" > /dev/null << EOF || { echo "❌ Error: Failed to create desktop file"; exit 1; }
[Desktop Entry]
Name=LM Studio
Comment=Run Large Language Models locally with LM Studio
Exec=/home/${SUDO_USER:-$USER}/AI/LMStudio/LM-Studio.AppImage --no-sandbox
Icon=application-x-executable
Terminal=false
Type=Application
Categories=Utility;Development;
EOF
    sudo chmod +x "$DESKTOP_FILE"
    echo "✅ Success: Desktop entry created"
}

verify_install() {
    echo "🔍 Verifying LM Studio installation..."
    INSTALL_DIR="/home/${SUDO_USER:-$USER}/AI/LMStudio"
    if [ -x "$INSTALL_DIR/LM-Studio.AppImage" ]; then
        echo "✅ Success: LM Studio AppImage is executable"
        echo "
✨ LM Studio Installation Complete! ✨
- Run: $INSTALL_DIR/LM-Studio.AppImage --no-sandbox
- Or use the desktop application menu
- Version: 0.3.10-6
- Note: Use '--no-sandbox' if running manually as non-root
        "
    else
        echo "❌ Error: LM Studio AppImage not found or not executable at $INSTALL_DIR"
        exit 1
    fi
}

main() {
    echo "🔧 Entering main function..."
    check_root
    install_prerequisites
    download_lmstudio
    install_lmstudio
    create_desktop_entry
    verify_install
}

# Trap errors with line numbers
trap 'echo "❌ Script failed at line $LINENO with exit code $?"' ERR

main