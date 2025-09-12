#!/bin/bash

# Application Installation Script
# Version: 1.0.4 - Built by Cortana for Jason
# Date: March 12, 2025

# Enable command tracing but continue on errors for individual apps
set -x  # Trace commands

echo "Starting Application Installation on Ubuntu..."

# Log file for installation
LOG_FILE="/tmp/app_install_log.txt"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

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
    apt update || { echo "WARNING: apt update failed, but continuing..."; }
    apt install -y gpg apt-transport-https ca-certificates software-properties-common snapd wget curl gnupg libfuse2 || { 
        echo "ERROR: Failed to install prerequisites. Attempting to continue anyway."
    }
}

install_app() {
    local app_name="$1"
    local install_func="$2"
    
    echo "================================================"
    echo "Installing $app_name..."
    echo "================================================"
    
    # Temporarily disable exit on error for this app
    set +e
    $install_func
    local status=$?
    set -e
    
    if [ $status -eq 0 ]; then
        echo "SUCCESS: $app_name installed"
    else
        echo "WARNING: $app_name installation failed with status $status"
        echo "$app_name installation failed" >> /tmp/failed_installs.txt
    fi
    
    # Small pause between installations
    sleep 1
}

install_brave() {
    curl -fsSLo /usr/share/keyrings/brave-browser-archive-keyring.gpg https://brave-browser-apt-release.s3.brave.com/brave-browser-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/brave-browser-archive-keyring.gpg] https://brave-browser-apt-release.s3.brave.com/ stable main" | tee /etc/apt/sources.list.d/brave-browser-release.list
    apt update
    apt install -y brave-browser
}

install_nordvpn() {
    cd /tmp
    wget -qO - https://downloads.nordcdn.com/apps/linux/install.sh | sh
    usermod -aG nordvpn $USER
    if ! command -v nordvpn &> /dev/null; then
        echo "Trying alternate NordVPN installation method..."
        wget -qnc https://repo.nordvpn.com/deb/nordvpn/debian/pool/main/nordvpn-release_1.0.0_all.deb
        apt install -y ./nordvpn-release_1.0.0_all.deb
        apt update
        apt install -y nordvpn
        rm -f nordvpn-release_1.0.0_all.deb
    fi
}

install_nordpass() {
    cd /tmp
    snap install nordpass
    if ! command -v nordpass &> /dev/null; then
        echo "Trying alternate NordPass installation method..."
        wget -qO- https://downloads.npass.app/gpg/nordpass_pub.gpg | gpg --dearmor -o /usr/share/keyrings/nordpass-archive-keyring.asc
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/nordpass-archive-keyring.asc] https://downloads.npass.app/linux/deb/x86_64 stable main" | tee /etc/apt/sources.list.d/nordpass.list
        apt update
        apt install -y nordpass
    fi
}

install_teamviewer() {
    cd /tmp
    wget -q https://download.teamviewer.com/download/linux/teamviewer_amd64.deb
    apt install -y ./teamviewer_amd64.deb
    rm -f teamviewer_amd64.deb
}

install_obs() {
    add-apt-repository ppa:obsproject/obs-studio -y
    apt update
    apt install -y obs-studio
    if ! command -v obs &> /dev/null; then
        echo "Trying alternate OBS Studio installation method via Flatpak..."
        apt install -y flatpak
        flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
        flatpak install -y flathub com.obsproject.Studio
    fi
}

install_spotify() {
    snap install spotify
    if ! command -v spotify &> /dev/null; then
        echo "Trying alternate Spotify installation method..."
        curl -sS https://download.spotify.com/debian/pubkey_C85668DF69375001.gpg | gpg --dearmor --yes -o /etc/apt/trusted.gpg.d/spotify.gpg
        echo "deb https://repository.spotify.com stable non-free" | tee /etc/apt/sources.list.d/spotify.list
        apt update
        apt install -y spotify-client
    fi
}

install_vlc() {
    apt install -y vlc
}

install_ffmpeg() {
    apt install -y ffmpeg
}

install_qbittorrent() {
    apt install -y qbittorrent
}

install_telegram() {
    snap install telegram-desktop
    if ! command -v telegram-desktop &> /dev/null; then
        echo "Trying alternate Telegram installation method via apt..."
        apt install -y telegram-desktop
    fi
}

install_sublime() {
    wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | gpg --dearmor -o /usr/share/keyrings/sublimehq-pub.gpg
    echo "deb [signed-by=/usr/share/keyrings/sublimehq-pub.gpg] https://download.sublimetext.com/ apt/stable/" | tee /etc/apt/sources.list.d/sublime-text.list
    apt update
    apt install -y sublime-text
}

install_plex() {
    echo "deb [signed-by=/usr/share/keyrings/plexmediaserver-keyring.gpg] https://downloads.plex.tv/repo/deb public main" | tee /etc/apt/sources.list.d/plexmediaserver.list
    curl -fsSL https://downloads.plex.tv/plex-keys/PlexSign.key | gpg --dearmor -o /usr/share/keyrings/plexmediaserver-keyring.gpg
    apt update
    apt install -y plexmediaserver
}

install_whatsapp() {
    snap install whatsapp-for-linux
}

install_messenger() {
    snap install caprine
}

install_trello() {
    snap install trello-desktop
}

main() {
    > /tmp/failed_installs.txt
    check_root
    install_prerequisites
    
    install_app "Brave Browser" install_brave
    install_app "NordVPN" install_nordvpn
    install_app "NordPass" install_nordpass
    install_app "TeamViewer" install_teamviewer
    install_app "OBS Studio" install_obs
    install_app "Spotify" install_spotify
    install_app "VLC" install_vlc
    install_app "FFmpeg" install_ffmpeg
    install_app "qBittorrent" install_qbittorrent
    install_app "Telegram" install_telegram
    install_app "Sublime Text" install_sublime
    install_app "Plex Media Server" install_plex
    install_app "WhatsApp" install_whatsapp
    install_app "Facebook Messenger" install_messenger
    install_app "Trello" install_trello

    if [ -s /tmp/failed_installs.txt ]; then
        echo "
Installation Complete with some warnings!
- The following applications encountered errors during installation:
$(cat /tmp/failed_installs.txt)
- Please check the log file at $LOG_FILE for details
"
    else
        echo "
Installation Complete! All applications installed successfully:
- Brave, NordVPN, NordPass, TeamViewer, OBS, Spotify, VLC, FFmpeg, qBittorrent
- Telegram, Sublime Text, Plex, WhatsApp, Messenger, Trello

Notes:
- NordVPN/NordPass: Configure via their GUIs (run 'nordvpn login' to set up NordVPN)
- Plex: Access setup at http://localhost:32400/web
- Reboot recommended for snap apps and to apply NordVPN group membership
"
    fi
}

main
