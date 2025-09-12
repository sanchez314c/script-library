#!/bin/bash

# Application Installation Script
# Version: 1.0.3 - Updated by Claude 3.7 for Jason
# Date: March 2, 2025

set -x  # Trace commands
# Removed set -e to prevent exit on errors

# Function to log success or failure
log_status() {
    if [ $? -eq 0 ]; then
        echo "✅ Success: $1"
    else
        echo "⚠️ Warning: $1 failed, continuing anyway"
    fi
}

echo "Starting Application Installation on Ubuntu..."

check_root() {
    echo "Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "Error: Requires root. Run with sudo."
        exit 1
    fi
    echo "Success: Running as root"
}

install_prerequisites() {
    echo "Installing prerequisites..."
    apt update
    apt install -y gpg apt-transport-https ca-certificates software-properties-common snapd wget curl
    log_status "Prerequisites"
}

install_brave() {
    echo "Installing Brave Browser..."
    curl -fsSL https://brave-browser-apt-release.s3.brave.com/brave-browser-archive-keyring.gpg | gpg --dearmor -o /usr/share/keyrings/brave-browser-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/brave-browser-archive-keyring.gpg] https://brave-browser-apt-release.s3.brave.com/ stable main" | tee /etc/apt/sources.list.d/brave-browser-release.list
    apt update
    apt install -y brave-browser
    log_status "Brave Browser"
}

install_nordvpn() {
    echo "Installing NordVPN..."
    # Try to install via snap first (preferred method for Ubuntu)
    snap install nordvpn
    log_status "NordVPN"
}

install_nordpass() {
    echo "Installing NordPass..."
    # Try snap installation first
    snap install nordpass
    log_status "NordPass"
}

install_teamviewer() {
    echo "Installing TeamViewer..."
    cd /tmp
    wget -q https://download.teamviewer.com/download/linux/teamviewer_amd64.deb
    apt install -y ./teamviewer_amd64.deb
    rm -f teamviewer_amd64.deb
    log_status "TeamViewer"
}

install_obs() {
    echo "Installing OBS Studio..."
    add-apt-repository ppa:obsproject/obs-studio -y
    apt update
    apt install -y obs-studio
    log_status "OBS Studio"
}

install_spotify() {
    echo "Installing Spotify..."
    # Try snap installation first (recommended for Ubuntu)
    snap install spotify
    log_status "Spotify"
}

install_vlc() {
    echo "Installing VLC..."
    apt install -y vlc
    log_status "VLC"
}

install_ffmpeg() {
    echo "Installing FFmpeg..."
    apt install -y ffmpeg
    log_status "FFmpeg"
}

install_qbittorrent() {
    echo "Installing qBittorrent..."
    apt install -y qbittorrent
    log_status "qBittorrent"
}

install_telegram() {
    echo "Installing Telegram..."
    snap install telegram-desktop
    log_status "Telegram"
}

install_sublime() {
    echo "Installing Sublime Text..."
    wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | gpg --dearmor -o /usr/share/keyrings/sublimehq-pub.gpg
    echo "deb [signed-by=/usr/share/keyrings/sublimehq-pub.gpg] https://download.sublimetext.com/ apt/stable/" | tee /etc/apt/sources.list.d/sublime-text.list
    apt update
    apt install -y sublime-text
    log_status "Sublime Text"
}

install_plex() {
    echo "Installing Plex Media Server..."
    echo "deb [signed-by=/usr/share/keyrings/plexmediaserver-keyring.gpg] https://downloads.plex.tv/repo/deb public main" | tee /etc/apt/sources.list.d/plexmediaserver.list
    curl -fsSL https://downloads.plex.tv/plex-keys/PlexSign.key | gpg --dearmor -o /usr/share/keyrings/plexmediaserver-keyring.gpg
    apt update
    apt install -y plexmediaserver
    log_status "Plex Media Server"
}

install_whatsapp() {
    echo "Installing WhatsApp (via snap)..."
    # The correct snap package name is just "whatsapp"
    snap install whatsapp
    log_status "WhatsApp"
}

install_messenger() {
    echo "Installing Facebook Messenger (Caprine via snap)..."
    snap install caprine
    log_status "Facebook Messenger (Caprine)"
}

install_trello() {
    echo "Installing Trello (via snap)..."
    snap install trello-desktop
    log_status "Trello"
}

install_pygpt() {
    echo "Installing PyGPT (via snap)..."
    snap install pygpt
    log_status "PyGPT"
}

install_lmstudio() {
    echo "Installing LM Studio..."
    # Install via snap if possible
    if snap info lmstudio >/dev/null 2>&1; then
        snap install lmstudio
        log_status "LM Studio (via snap)"
        return
    fi
    
    # Otherwise install via AppImage
    LMSTUDIO_URL="https://installers.lmstudio.ai/linux/x64/0.3.10-6/LM-Studio-0.3.10-6-x64.AppImage"
    INSTALL_DIR="/opt/LMStudio"
    APPIMAGE_DEST="$INSTALL_DIR/LM-Studio-0.3.10-6-x64.AppImage"
    DESKTOP_FILE="/usr/share/applications/lm-studio.desktop"
    ICON_DEST="$INSTALL_DIR/lm-studio-icon.png"

    # Install FUSE for AppImage
    apt install -y libfuse2

    # Download LM Studio AppImage
    echo "Downloading LM Studio from $LMSTUDIO_URL..."
    cd /tmp
    wget -q "$LMSTUDIO_URL" -O LM-Studio-0.3.10-6-x64.AppImage || { 
        echo "⚠️ Warning: Failed to download LM Studio, continuing anyway"
        return
    }

    # Move and set up AppImage
    mkdir -p "$INSTALL_DIR"
    mv LM-Studio-0.3.10-6-x64.AppImage "$APPIMAGE_DEST" 
    chmod +x "$APPIMAGE_DEST"

    # Create desktop entry
    cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Name=LM Studio
Comment=Run Large Language Models locally with LM Studio
Exec=$APPIMAGE_DEST
Icon=application-x-executable
Terminal=false
Type=Application
Categories=Utility;Development;
EOF
    chmod +x "$DESKTOP_FILE"
    log_status "LM Studio (via AppImage)"
}

main() {
    check_root
    install_prerequisites
    install_brave
    install_nordvpn
    install_nordpass
    install_teamviewer
    install_obs
    install_spotify
    install_vlc
    install_ffmpeg
    install_qbittorrent
    install_telegram
    install_sublime
    install_plex
    install_whatsapp
    install_messenger
    install_trello
    install_pygpt
    install_lmstudio

    echo "
✅ Application Installation Complete!

Setup notes:
- NordVPN/NordPass: Configure via their GUIs
- Plex: Access setup at http://localhost:32400/web
- For snap apps: Reboot recommended to see all app icons

Apps attempted to install:
- Browsers: Brave
- Security: NordVPN, NordPass
- Media: VLC, FFmpeg, OBS Studio, Spotify, Plex Media Server
- Communication: Telegram, WhatsApp, Facebook Messenger (Caprine)
- Productivity: TeamViewer, Sublime Text, Trello, PyGPT
- AI: LM Studio

Some apps may have failed installation - check above logs for details
"
}

main
