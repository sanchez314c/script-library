#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "Starting installation of requested applications..."

# Function to print status messages
print_status() {
    echo -e "${GREEN}[*] $1${NC}"
}

# Function to check if command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[✓] $1 completed successfully${NC}"
    else
        echo -e "${RED}[×] $1 failed${NC}"
    fi
}

# Update system first
print_status "Updating system packages"
sudo apt update && sudo apt upgrade -y
check_status "System update"

# Install required dependencies
print_status "Installing required dependencies"
sudo apt install -y wget gpg apt-transport-https curl software-properties-common

# Brave Browser
print_status "Installing Brave Browser"
sudo curl -fsSLo /usr/share/keyrings/brave-browser-archive-keyring.gpg https://brave-browser-apt-release.s3.brave.com/brave-browser-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/brave-browser-archive-keyring.gpg] https://brave-browser-apt-release.s3.brave.com/ stable main" | sudo tee /etc/apt/sources.list.d/brave-browser-release.list
sudo apt update
sudo apt install -y brave-browser
check_status "Brave Browser installation"

# NordVPN
print_status "Installing NordVPN"
cd /tmp
wget -qnc https://repo.nordvpn.com/deb/nordvpn/debian/pool/main/nordvpn-release_1.0.0_all.deb
sudo apt install -y ./nordvpn-release_1.0.0_all.deb
sudo apt update
sudo apt install -y nordvpn
check_status "NordVPN installation"

# NordPass
print_status "Installing NordPass"
cd /tmp
wget -qO- https://downloads.npass.app/gpg/nordpass_pub.gpg | sudo tee /usr/share/keyrings/nordpass-archive-keyring.asc
echo "deb [signed-by=/usr/share/keyrings/nordpass-archive-keyring.asc] https://downloads.npass.app/linux/deb/x86_64 stable main" | sudo tee /etc/apt/sources.list.d/nordpass.list
sudo apt update
sudo apt install -y nordpass
check_status "NordPass installation"

# TeamViewer
print_status "Installing TeamViewer"
wget https://download.teamviewer.com/download/linux/teamviewer_amd64.deb
sudo apt install -y ./teamviewer_amd64.deb
rm teamviewer_amd64.deb
check_status "TeamViewer installation"

# OBS Studio
print_status "Installing OBS Studio"
sudo add-apt-repository ppa:obsproject/obs-studio -y
sudo apt update
sudo apt install -y obs-studio
check_status "OBS Studio installation"

# Spotify
print_status "Installing Spotify"
curl -sS https://download.spotify.com/debian/pubkey_7A3A762FAFD4A51F.gpg | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/spotify.gpg
echo "deb http://repository.spotify.com stable non-free" | sudo tee /etc/apt/sources.list.d/spotify.list
sudo apt update
sudo apt install -y spotify-client
check_status "Spotify installation"

# VLC
print_status "Installing VLC"
sudo apt install -y vlc
check_status "VLC installation"

# FFmpeg
print_status "Installing FFmpeg"
sudo apt install -y ffmpeg
check_status "FFmpeg installation"

# qBittorrent
print_status "Installing qBittorrent"
sudo apt install -y qbittorrent
check_status "qBittorrent installation"

# Telegram
print_status "Installing Telegram"
sudo apt install -y telegram-desktop
check_status "Telegram installation"

# Sublime Text
print_status "Installing Sublime Text"
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -
echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list
sudo apt update
sudo apt install -y sublime-text
check_status "Sublime Text installation"

# Plex Media Server
print_status "Installing Plex Media Server"
echo deb https://downloads.plex.tv/repo/deb public main | sudo tee /etc/apt/sources.list.d/plexmediaserver.list
curl https://downloads.plex.tv/plex-keys/PlexSign.key | sudo apt-key add -
sudo apt update
sudo apt install -y plexmediaserver
check_status "Plex Media Server installation"

# WhatsApp (using WhatsApp for Linux)
print_status "Installing WhatsApp (unofficial client)"
sudo snap install whatsapp-for-linux
check_status "WhatsApp installation"

# Facebook Messenger (using Caprine)
print_status "Installing Facebook Messenger (Caprine)"
sudo snap install caprine
check_status "Facebook Messenger installation"

# Trello (using web app)
print_status "Installing Trello"
sudo snap install trello-desktop
check_status "Trello installation"

echo -e "${GREEN}Installation process completed!${NC}"
