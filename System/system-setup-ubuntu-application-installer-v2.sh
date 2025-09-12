#!/bin/bash
###############################################################
#     _    ____  ____       ___ _   _ ____ _____              #
#    / \  |  _ \|  _ \     |_ _| \ | / ___|_   _|             #
#   / _ \ | |_) | |_) |_____| ||  \| \___ \ | |               #
#  / ___ \|  __/|  __/_____|_|| |\  |___) || |                #
# /_/   \_\_|   |_|        |___|_| \_|____/ |_|               #
#                                                             #
###############################################################

# Application Installation Script
# Version: 1.2.0
# Last Updated: April, 2025
# Description: Installs common applications on Ubuntu with improved error handling

# Enable error handling but continue for individual apps
set -e

echo "🚀 Starting Application Installation on Ubuntu..."

# Get username even when run with sudo
REAL_USER="${SUDO_USER:-$USER}"

# Log file with timestamp on desktop
LOG_FILE="/home/$REAL_USER/Desktop/app_install_$(date +%Y%m%d_%H%M%S).log"
FAILED_INSTALLS_LOG="/home/$REAL_USER/Desktop/failed_installs.txt"
touch "$LOG_FILE" "$FAILED_INSTALLS_LOG"
chown $REAL_USER:$REAL_USER "$LOG_FILE" "$FAILED_INSTALLS_LOG"
chmod 644 "$LOG_FILE" "$FAILED_INSTALLS_LOG"
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
    local start_time=$(date +%s)
    
    echo "================================================"
    echo "Installing $app_name..."
    echo "================================================"
    
    # Temporarily disable exit on error for this app
    set +e
    $install_func
    local status=$?
    set -e
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $status -eq 0 ]; then
        echo "SUCCESS: $app_name installed in ${duration}s"
    else
        echo "WARNING: $app_name installation failed with status $status"
        echo "$app_name" >> "$FAILED_INSTALLS_LOG"
    fi
    
    # Small pause between installations
    sleep 1
}

install_brave() {
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSLo /etc/apt/keyrings/brave-browser-archive-keyring.gpg https://brave-browser-apt-release.s3.brave.com/brave-browser-archive-keyring.gpg
    echo "deb [signed-by=/etc/apt/keyrings/brave-browser-archive-keyring.gpg] https://brave-browser-apt-release.s3.brave.com/ stable main" | tee /etc/apt/sources.list.d/brave-browser-release.list
    apt update
    apt install -y brave-browser
}

install_nordvpn() {
    cd /tmp
    wget -qO - https://downloads.nordcdn.com/apps/linux/install.sh | sh
    if [ -n "${SUDO_USER:-}" ]; then
        usermod -aG nordvpn $SUDO_USER
    else
        usermod -aG nordvpn $USER
    fi
    
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
        install -m 0755 -d /etc/apt/keyrings
        wget -qO- https://downloads.npass.app/gpg/nordpass_pub.gpg | gpg --dearmor -o /etc/apt/keyrings/nordpass-archive-keyring.asc
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/nordpass-archive-keyring.asc] https://downloads.npass.app/linux/deb/x86_64 stable main" | tee /etc/apt/sources.list.d/nordpass.list
        apt update
        apt install -y nordpass
    fi
}

install_teamviewer() {
    cd /tmp
    wget -q https://download.teamviewer.com/download/linux/teamviewer_amd64.deb
    apt install -y --fix-broken ./teamviewer_amd64.deb
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
        install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.spotify.com/debian/pubkey_6224F9941A8AA6D1.gpg | gpg --dearmor -o /etc/apt/keyrings/spotify.gpg
        echo "deb [signed-by=/etc/apt/keyrings/spotify.gpg] http://repository.spotify.com stable non-free" | tee /etc/apt/sources.list.d/spotify.list
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
    install -m 0755 -d /etc/apt/keyrings
    wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | gpg --dearmor -o /etc/apt/keyrings/sublimehq-archive-keyring.gpg
    echo "deb [signed-by=/etc/apt/keyrings/sublimehq-archive-keyring.gpg] https://download.sublimetext.com/ apt/stable/" | tee /etc/apt/sources.list.d/sublime-text.list
    apt update
    apt install -y sublime-text
}

install_plex() {
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://downloads.plex.tv/plex-keys/PlexSign.key | gpg --dearmor -o /etc/apt/keyrings/plexmediaserver.gpg
    echo "deb [signed-by=/etc/apt/keyrings/plexmediaserver.gpg] https://downloads.plex.tv/repo/deb public main" | tee /etc/apt/sources.list.d/plexmediaserver.list
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
    # Clear previous failure log
    > "$FAILED_INSTALLS_LOG"
    check_root
    install_prerequisites
    
    # Install all applications using the wrapper function
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

    # Final summary
    if [ -s "$FAILED_INSTALLS_LOG" ]; then
        echo "
Installation Complete with some warnings!
- The following applications encountered errors during installation:
$(cat "$FAILED_INSTALLS_LOG")
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