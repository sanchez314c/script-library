#!/bin/bash

# LM Studio Installer Script
# Version: 1.0.0 - Built by Cortana (via Grok 3, xAI) for Jason
# Date: February 24, 2025

set -x  # Trace every command
set -e  # Exit on any error

echo "🚀 Starting LM Studio Installation on Ubuntu..."

# Variables
APPIMAGE_SRC="$HOME/Downloads/LM-Studio-0.3.10-6-x64.AppImage"
INSTALL_DIR="/opt/LMStudio"
APPIMAGE_DEST="$INSTALL_DIR/LM-Studio-0.3.10-6-x64.AppImage"
DESKTOP_FILE="/usr/share/applications/lm-studio.desktop"
ICON_DEST="$INSTALL_DIR/lm-studio-icon.png"

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "❌ Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    echo "✅ Success: Running as root"
}

check_appimage() {
    echo "🔍 Checking for LM Studio AppImage..."
    if [ ! -f "$APPIMAGE_SRC" ]; then
        echo "❌ Error: $APPIMAGE_SRC not found in Downloads. Please download it first."
        exit 1
    fi
    echo "✅ Success: AppImage found at $APPIMAGE_SRC"
}

install_fuse() {
    echo "📦 Ensuring FUSE support for AppImage..."
    if ! dpkg -l | grep -q libfuse2t64; then
        echo "Installing libfuse2t64..."
        apt update || { echo "❌ Error: Apt update failed"; exit 1; }
        apt install -y libfuse2t64 || { echo "❌ Error: libfuse2t64 install failed"; exit 1; }
    else
        echo "✅ libfuse2t64 already installed"
    fi
}

move_and_setup_appimage() {
    echo "📦 Moving and setting up LM Studio AppImage..."
    mkdir -pv "$INSTALL_DIR" || { echo "❌ Error: Failed to create $INSTALL_DIR"; exit 1; }
    mv -v "$APPIMAGE_SRC" "$APPIMAGE_DEST" || { echo "❌ Error: Failed to move AppImage"; exit 1; }
    chmod +x "$APPIMAGE_DEST" || { echo "❌ Error: Failed to make AppImage executable"; exit 1; }
    echo "✅ Success: AppImage moved to $INSTALL_DIR and made executable"
}

extract_icon() {
    echo "🖼️ Extracting icon from AppImage..."
    cd "$INSTALL_DIR" || { echo "❌ Error: Failed to cd to $INSTALL_DIR"; exit 1; }
    "$APPIMAGE_DEST" --appimage-extract >/dev/null 2>&1 || { echo "⚠️ Warning: Icon extraction failed—using fallback"; return; }
    ICON_SRC=$(find squashfs-root -name "*.png" -o -name "*.svg" | head -1)
    if [ -n "$ICON_SRC" ]; then
        cp -v "$ICON_SRC" "$ICON_DEST" || { echo "❌ Error: Failed to copy icon"; exit 1; }
        echo "✅ Success: Icon extracted to $ICON_DEST"
    else
        echo "⚠️ No icon found—using fallback"
    fi
    rm -rf squashfs-root
}

create_desktop_entry() {
    echo "📝 Creating desktop entry for LM Studio..."
    cat > "$DESKTOP_FILE" << EOF || { echo "❌ Error: Failed to create desktop file"; exit 1; }
[Desktop Entry]
Name=LM Studio
Comment=Run Large Language Models locally with LM Studio
Exec=$APPIMAGE_DEST
Icon=$([ -f "$ICON_DEST" ] && echo "$ICON_DEST" || echo "application-x-executable")
Terminal=false
Type=Application
Categories=Utility;Development;
EOF
    chmod +x "$DESKTOP_FILE" || { echo "❌ Error: Failed to make desktop file executable"; exit 1; }
    echo "✅ Success: Desktop entry created at $DESKTOP_FILE"
}

main() {
    echo "🔧 Entering main function..."
    check_root
    check_appimage
    install_fuse
    move_and_setup_appimage
    extract_icon
    create_desktop_entry
    echo "
✨ LM Studio Installation Complete! ✨
- AppImage: $APPIMAGE_DEST
- Desktop Entry: $DESKTOP_FILE
- Icon: $([ -f "$ICON_DEST" ] && echo "$ICON_DEST" || echo "Using system fallback")
Instructions:
- Find 'LM Studio' in Ubuntu’s apps view (Show Applications)
- Launch it anytime from there!
    "
}

trap 'echo "❌ Script failed at line $LINENO with exit code $?"' ERR

main
