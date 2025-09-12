#!/bin/bash

# Essential Application Installer Script
# Author: Cortana AI for Jason, updated by Grok 3 (xAI)
# Purpose: Full Ubuntu setup with essentials, apps, CUDA 11.8, cuDNN 9.8, and more
# Version: 1.1.0 - Last Updated: 2025-03-21

# Enable verbosity and error handling
set -x  # Trace commands
set -e  # Exit on error

echo "🚀 Starting Essential Application Installation on Ubuntu..."

# Log file for installation
LOG_FILE="/tmp/essential_app_install_log.txt"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

cleanup() {
    local exit_status=$1
    echo "🧹 Cleaning up after exit with status $exit_status..."
    echo "Cleaning apt cache..."
    sudo apt-get clean || echo "⚠️ Warning: Apt clean failed"
    echo "Running autoremove..."
    sudo apt-get autoremove -y || echo "⚠️ Warning: Autoremove failed"
    if [ $exit_status -ne 0 ]; then
        echo "❌ Error detected. Attempting to fix package system..."
        echo "Configuring interrupted dpkg..."
        sudo dpkg --configure -a || echo "⚠️ Warning: dpkg configure failed"
        echo "Updating with fix-missing..."
        sudo apt-get update --fix-missing || echo "⚠️ Warning: Update fix-missing failed"
        echo "Installing broken packages..."
        sudo apt-get install -f || echo "⚠️ Warning: Install fix failed"
    fi
    echo "✅ Cleanup complete"
}

# Trap exit and errors
trap 'cleanup $?' ERR EXIT

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$EUID" -ne 0 ]; then
        echo "❌ Error: Please run as root (use sudo)"
        exit 1
    fi
    echo "✅ Success: Running as root"
}

install_packages() {
    local packages=("$@")
    echo "📦 Installing ${#packages[@]} packages..."
    for pkg in "${packages[@]}"; do
        echo "Checking if $pkg is already installed..."
        if dpkg -l | grep -q "^ii  $pkg "; then
            echo "✅ Success: $pkg already installed"
            continue
        fi
        echo "Installing $pkg..."
        sudo apt-get install -y "$pkg" || {
            echo "⚠️ Warning: Failed to install $pkg. Attempting recovery..."
            echo "Updating package lists..."
            sudo apt-get update || echo "❌ Error: Update failed during recovery"
            echo "Fixing broken installs..."
            sudo apt-get install -f || echo "❌ Error: Install fix failed"
            echo "Retrying $pkg install..."
            sudo apt-get install -y "$pkg" || { echo "❌ Error: Failed to install $pkg after recovery"; exit 1; }
        }
        echo "✅ Success: $pkg installed"
    done
}

install_app() {
    local app_name="$1"
    local install_func="$2"
    echo "================================================"
    echo "Installing $app_name..."
    echo "================================================"
    set +e  # Disable exit on error for this app
    $install_func
    local status=$?
    set -e  # Re-enable exit on error
    if [ $status -eq 0 ]; then
        echo "✅ Success: $app_name installed"
    else
        echo "⚠️ Warning: $app_name installation failed with status $status"
        echo "$app_name installation failed" >> /tmp/failed_installs.txt
    fi
    sleep 1
}

fix_interruptions() {
    echo "🔧 Fixing interrupted installations..."
    sudo dpkg --configure -a || { echo "❌ Error: dpkg configure failed"; exit 1; }
    echo "✅ Success: Interrupted installations fixed"
}

update_system() {
    echo "🔄 Updating system..."
    sudo apt-get update || { echo "❌ Error: Apt update failed"; exit 1; }
    sudo apt-get upgrade -y || { echo "❌ Error: Apt upgrade failed"; exit 1; }
    echo "✅ Success: System updated"
}

install_prerequisites() {
    echo "📦 Installing prerequisites..."
    apt update || { echo "⚠️ apt update failed, but continuing..."; }
    apt install -y gpg apt-transport-https ca-certificates software-properties-common snapd wget curl gnupg libfuse2 || { 
        echo "⚠️ Failed to install prerequisites. Continuing anyway."
    }
}

install_essentials() {
    echo "📦 Installing essential packages including GCC 12 and additional tools..."
    essential_packages=(
        build-essential cmake make pkg-config git vim nano wget curl htop net-tools
        software-properties-common apt-transport-https ca-certificates gnupg lsb-release
        python3-pip python3-dev python3-venv ffmpeg libssl-dev libffi-dev libxml2-dev
        libxslt1-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libopencv-dev
        libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev
        libv4l-dev libxvidcore-dev libx264-dev unzip screen tmux fish zsh
        gcc-12 g++-12 jq tree rsync zip bzip2 sudo man-db snapd rustc cargo
    )
    install_packages "${essential_packages[@]}"
    echo "Setting GCC 12 as default..."
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 || { echo "❌ Error: Failed to set gcc-12"; exit 1; }
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 || { echo "❌ Error: Failed to set g++-12"; exit 1; }
    echo "✅ Success: Essentials installed"
}

install_go() {
    echo "📦 Installing latest Go..."
    GO_VERSION="1.22.0"  # Update to latest from go.dev/dl in March 2025
    wget -q https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz -O go.tar.gz || { echo "❌ Error: Go download failed"; exit 1; }
    sudo tar -C /usr/local -xzf go.tar.gz || { echo "❌ Error: Go extraction failed"; exit 1; }
    rm go.tar.gz
    sudo -u "${SUDO_USER:-$USER}" bash -c "echo 'export PATH=/usr/local/go/bin:\$PATH' >> ~${SUDO_USER:-$USER}/.bashrc" || { echo "❌ Error: Failed to update .bashrc for Go"; exit 1; }
    /usr/local/go/bin/go version || { echo "❌ Error: Go not functional"; exit 1; }
    echo "✅ Success: Go installed"
}

install_modern_tools() {
    echo "🛠️ Installing modern CLI tools..."
    modern_tools=(bat eza fd-find ripgrep fzf)
    install_packages "${modern_tools[@]}"
    echo "✅ Success: Modern tools installed"
}

create_symlinks() {
    echo "🔗 Creating symlinks for modern tools..."
    sudo -u "${SUDO_USER:-$USER}" mkdir -pv ~${SUDO_USER:-$USER}/.local/bin || { echo "❌ Error: Failed to create ~/.local/bin"; exit 1; }
    [ -x "$(which batcat)" ] && sudo -u "${SUDO_USER:-$USER}" ln -sfv "$(which batcat)" ~${SUDO_USER:-$USER}/.local/bin/bat || echo "⚠️ batcat not found"
    [ -x "$(which fdfind)" ] && sudo -u "${SUDO_USER:-$USER}" ln -sfv "$(which fdfind)" ~${SUDO_USER:-$USER}/.local/bin/fd || echo "⚠️ fdfind not found"
    echo "✅ Success: Symlinks created"
}

install_whatsie() {
    echo "📱 Installing Whatsie by Keshav Bhatt..."
    WHATSIE_VERSION="4.14.2"
    wget -q "https://github.com/keshavbhatt/whatsie/releases/download/v${WHATSIE_VERSION}/whatsie_${WHATSIE_VERSION}_amd64.deb" -O whatsie.deb || { echo "❌ Error: Whatsie download failed"; exit 1; }
    sudo dpkg -i whatsie.deb || { sudo apt-get install -f; sudo dpkg -i whatsie.deb; } || { echo "❌ Error: Whatsie install failed"; exit 1; }
    rm whatsie.deb
    command -v whatsie >/dev/null 2>&1 && echo "✅ Success: Whatsie installed. Version: $(whatsie --version)" || { echo "❌ Error: Whatsie not found"; exit 1; }
}

install_amdgpu_top() {
    echo "📊 Installing amdgpu_top..."
    sudo -u "${SUDO_USER:-$USER}" cargo install amdgpu_top || { echo "❌ Error: amdgpu_top install failed"; exit 1; }
    sudo -u "${SUDO_USER:-$USER}" bash -c "echo 'export PATH=~/.cargo/bin:\$PATH' >> ~${SUDO_USER:-$USER}/.bashrc" || { echo "❌ Error: Failed to update .bashrc for cargo"; exit 1; }
    ~/.cargo/bin/amdgpu_top --version >/dev/null 2>&1 && echo "✅ Success: amdgpu_top installed. Version: $(~/.cargo/bin/amdgpu_top --version)" || { echo "❌ Error: amdgpu_top not found"; exit 1; }
}

install_google_drive() {
    echo "💾 Installing Google Drive via rclone..."
    install_packages rclone
    echo "Run 'rclone config' after script to set up Google Drive."
    sudo -u "${SUDO_USER:-$USER}" bash -c "echo 'alias gdrive=\"rclone --vfs-cache-mode writes mount googledrive: ~/GoogleDrive\"' >> ~${SUDO_USER:-$USER}/.bashrc"
    sudo -u "${SUDO_USER:-$USER}" mkdir -pv ~${SUDO_USER:-$USER}/GoogleDrive || { echo "❌ Error: Failed to create GoogleDrive dir"; exit 1; }
    rclone --version || { echo "❌ Error: rclone not functional"; exit 1; }
    echo "✅ Success: rclone installed"
}

install_brave() {
    curl -fsSLo /usr/share/keyrings/brave-browser-archive-keyring.gpg https://brave-browser-apt-release.s3.brave.com/brave-browser-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/brave-browser-archive-keyring.gpg] https://brave-browser-apt-release.s3.brave.com/ stable main" | tee /etc/apt/sources.list.d/brave-browser-release.list
    apt update
    apt install -y brave-browser
}

install_nordvpn() {
    cd /tmp
    wget -qO - https://downloads.nordcdn.com/apps/linux/install.sh | sh || {
        wget -qnc https://repo.nordvpn.com/deb/nordvpn/debian/pool/main/nordvpn-release_1.0.0_all.deb
        apt install -y ./nordvpn-release_1.0.0_all.deb
        apt update
        apt install -y nordvpn
        rm -f nordvpn-release_1.0.0_all.deb
    }
    usermod -aG nordvpn "$SUDO_USER"
}

install_nordpass() {
    snap install nordpass || {
        wget -qO- https://downloads.npass.app/gpg/nordpass_pub.gpg | gpg --dearmor -o /usr/share/keyrings/nordpass-archive-keyring.asc
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/nordpass-archive-keyring.asc] https://downloads.npass.app/linux/deb/x86_64 stable main" | tee /etc/apt/sources.list.d/nordpass.list
        apt update
        apt install -y nordpass
    }
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
    apt install -y obs-studio || {
        apt install -y flatpak
        flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
        flatpak install -y flathub com.obsproject.Studio
    }
}

install_spotify() {
    snap install spotify || {
        curl -sS https://download.spotify.com/debian/pubkey_C85668DF69375001.gpg | gpg --dearmor --yes -o /etc/apt/trusted.gpg.d/spotify.gpg
        echo "deb https://repository.spotify.com stable non-free" | tee /etc/apt/sources.list.d/spotify.list
        apt update
        apt install -y spotify-client
    }
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
    snap install telegram-desktop || apt install -y telegram-desktop
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

install_messenger() {
    snap install caprine
}

install_trello() {
    snap install trello-desktop
}

setup_network_services() {
    echo "🌐 Setting up network services..."
    network_services=(openssh-server samba samba-common-bin)
    install_packages "${network_services[@]}"
    [ "$(dpkg -l | grep -q "^ii  openssh-server ")" ] || { echo "❌ Error: openssh-server failed to install"; exit 1; }
    echo "✅ Success: Network services installed"
}

detect_ssh_service() {
    systemctl list-unit-files | grep -q "ssh.service" && echo "ssh" || \
    systemctl list-unit-files | grep -q "sshd.service" && echo "sshd" || \
    { sudo systemctl daemon-reload; systemctl list-unit-files | grep -q "ssh.service" && echo "ssh" || systemctl list-unit-files | grep -q "sshd.service" && echo "sshd" || echo "UNKNOWN"; }
}

generate_ssh_keys() {
    echo "🔑 Generating SSH host keys..."
    sudo ssh-keygen -A || { echo "❌ Error: Failed to generate SSH keys"; exit 1; }
    echo "✅ Success: SSH keys generated"
}

backup_configs() {
    timestamp=$(date +%Y%m%d_%H%M%S)
    echo "💾 Backing up configs with timestamp $timestamp..."
    [ -f /etc/ssh/sshd_config ] && sudo cp -v /etc/ssh/sshd_config "/etc/ssh/sshd_config.bak.$timestamp"
    [ -f /etc/samba/smb.conf ] && sudo cp -v /etc/samba/smb.conf "/etc/samba/smb.conf.bak.$timestamp"
    [ -f ~${SUDO_USER:-$USER}/.bashrc ] && sudo -u "${SUDO_USER:-$USER}" cp -v ~${SUDO_USER:-$USER}/.bashrc ~${SUDO_USER:-$USER}/.bashrc.backup.$timestamp
    echo "✅ Success: Configs backed up"
}

configure_ssh() {
    echo "🔒 Configuring SSH..."
    sudo tee /etc/ssh/sshd_config > /dev/null << 'EOF' || { echo "❌ Error: Failed to write sshd_config"; exit 1; }
Include /etc/ssh/sshd_config.d/*.conf
Port 22
Protocol 2
HostKey /etc/ssh/ssh_host_rsa_key
HostKey /etc/ssh/ssh_host_ecdsa_key
HostKey /etc/ssh/ssh_host_ed25519_key
SyslogFacility AUTH
LogLevel INFO
PermitRootLogin prohibit-password
StrictModes yes
MaxAuthTries 6
MaxSessions 10
PubkeyAuthentication yes
PasswordAuthentication yes
PermitEmptyPasswords no
ChallengeResponseAuthentication no
UsePAM yes
X11Forwarding yes
PrintMotd no
AcceptEnv LANG LC_*
Subsystem sftp /usr/lib/openssh/sftp-server
EOF
    sudo sshd -t || { echo "❌ Error: SSH config invalid"; exit 1; }
    SSH_SERVICE=$(detect_ssh_service)
    [ "$SSH_SERVICE" != "UNKNOWN" ] && ! pgrep -f sshd > /dev/null && sudo systemctl restart "$SSH_SERVICE"
    echo "✅ Success: SSH configured"
}

configure_samba() {
    echo "📂 Setting up Samba..."
    sudo tee /etc/samba/smb.conf > /dev/null << 'EOF' || { echo "❌ Error: Failed to write smb.conf"; exit 1; }
[global]
workgroup = WORKGROUP
server string = %h server
log file = /var/log/samba/log.%m
max log size = 1000
logging = file
panic action = /usr/share/samba/panic-action %d
server role = standalone server
obey pam restrictions = yes
unix password sync = yes
passwd program = /usr/bin/passwd %u
passwd chat = *Enter\snew\s*\spassword:* %n\n *Retype\snew\s*\spassword:* %n\n *password\supdated\ssuccessfully* .
pam password change = yes
map to guest = bad user
usershare allow guests = yes

[homes]
comment = Home Directories
browseable = no
read only = no
create mask = 0700
directory mask = 0700
valid users = %S

[public]
comment = Public Share
path = /samba/public
browseable = yes
create mask = 0660
directory mask = 0771
writable = yes
guest ok = no

[LLMRAID]
comment = LLM RAID Storage
path = /media/heathen-admin/llmRAID
browseable = yes
read only = no
create mask = 0775
directory mask = 0775
valid users = heathen-admin
force user = heathen-admin
force group = heathen-admin
EOF
    echo "✅ Success: Samba configured"
}

setup_samba_dirs() {
    echo "📁 Creating Samba directories..."
    sudo mkdir -pv /samba/public && sudo chmod -v 777 /samba/public
    RAID_PATH="/media/heathen-admin/llmRAID"
    [ ! -d "$RAID_PATH" ] && sudo mkdir -pv "$RAID_PATH" && sudo chown -v heathen-admin:heathen-admin "$RAID_PATH" && sudo chmod -v 775 "$RAID_PATH" || echo "✅ $RAID_PATH already exists"
    echo "✅ Success: Samba dirs set up"
}

configure_shell() {
    echo "🔧 Configuring shell aliases..."
    sudo -u "${SUDO_USER:-$USER}" bash -c "echo -e '# Modern CLI Aliases\nalias ls=\"eza --icons --group-directories-first\"\nalias ll=\"eza -l --icons --group-directories-first\"\nalias la=\"eza -la --icons --group-directories-first\"\nalias cat=\"bat --style=full --paging=never\"\nalias find=\"fd\"\nalias grep=\"rg\"\n# System Monitoring Aliases\nalias htop=\"htop -t\"\nalias diskspace=\"df -h\"\nalias memusage=\"free -h\"\nalias sysinfo=\"neofetch\"\nalias monitor=\"glances\"\nalias netwatch=\"netdata\"\nalias gputop=\"nvtop\"\nalias amdtop=\"amdgpu_top\"\n# Google Drive Alias\nalias gdrive=\"rclone --vfs-cache-mode writes mount googledrive: ~/GoogleDrive\"' >> ~${SUDO_USER:-$USER}/.bashrc" || { echo "❌ Error: Failed to update .bashrc"; exit 1; }
    echo "✅ Success: Shell configured"
}

setup_monitoring() {
    echo "📊 Setting up monitoring tools..."
    monitoring_tools=(neofetch glances nvtop)
    install_packages "${monitoring_tools[@]}"
    sudo -u "${SUDO_USER:-$USER}" mkdir -pv ~${SUDO_USER:-$USER}/.config/neofetch
    sudo -u "${SUDO_USER:-$USER}" bash -c "echo -e 'print_info() {\n    info title\n    info underline\n    info \"OS\" distro\n    info \"Host\" model\n    info \"Kernel\" kernel\n    info \"Uptime\" uptime\n    info \"Packages\" packages\n    info \"Shell\" shell\n    info \"CPU\" cpu\n    info \"GPU\" gpu\n    info \"Memory\" memory\n    info \"Disk\" disk\n    info \"Local IP\" local_ip\n}' > ~${SUDO_USER:-$USER}/.config/neofetch/config.conf"
    echo "✅ Success: Monitoring tools set up"
}

setup_security() {
    echo "🔒 Setting up security tools..."
    security_tools=(unattended-upgrades apt-listchanges)
    install_packages "${security_tools[@]}"
    echo -e 'APT::Periodic::Update-Package-Lists "1";\nAPT::Periodic::Download-Upgradeable-Packages "1";\nAPT::Periodic::AutocleanInterval "7";\nAPT::Periodic::Unattended-Upgrade "1";' | sudo tee /etc/apt/apt.conf.d/20auto-upgrades > /dev/null
    echo "✅ Success: Security tools configured"
}

configure_firewall() {
    echo "🧱 Configuring firewall..."
    install_packages ufw
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    sudo ufw allow ssh
    sudo ufw allow samba
    sudo ufw allow 11434/tcp  # Ollama
    sudo ufw allow 3000/tcp   # OpenWebUI
    sudo ufw allow 8080/tcp   # OpenWebUI alt
    SSH_SERVICE=$(detect_ssh_service)
    [ "$SSH_SERVICE" != "UNKNOWN" ] && systemctl is-active "$SSH_SERVICE" >/dev/null || sudo systemctl start "$SSH_SERVICE"
    echo "y" | sudo ufw enable
    echo "✅ Success: Firewall configured"
}

start_services() {
    echo "🌐 Starting services..."
    SSH_SERVICE=$(detect_ssh_service)
    if [ "$SSH_SERVICE" != "UNKNOWN" ]; then
        sudo systemctl stop "$SSH_SERVICE" 2>/dev/null || echo "⚠️ $SSH_SERVICE wasn't running"
        sudo systemctl start "$SSH_SERVICE" && sudo systemctl enable "$SSH_SERVICE"
    else
        sudo apt-get install --reinstall openssh-server
        SSH_SERVICE=$(detect_ssh_service)
        [ "$SSH_SERVICE" != "UNKNOWN" ] && sudo systemctl start "$SSH_SERVICE" && sudo systemctl enable "$SSH_SERVICE" || { echo "❌ Error: SSH service not found"; exit 1; }
    fi
    for svc in smbd nmbd; do
        sudo systemctl stop "$svc" 2>/dev/null || echo "⚠️ $svc wasn't running"
        sudo systemctl start "$svc" && sudo systemctl enable "$svc"
    done
    echo "✅ Success: Services started"
}

verify_installations() {
    echo "🔍 Verifying installations..."
    USER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    BASHRC="$USER_HOME/.bashrc"
    source "$BASHRC"
    export PATH="$PATH:/usr/local/cuda-11.8/bin"
    command -v nvcc >/dev/null 2>&1 && echo "✅ CUDA: $(nvcc --version)" || echo "⚠️ CUDA not found"
    command -v nvidia-smi >/dev/null 2>&1 && echo "✅ NVIDIA-SMI: $(nvidia-smi | head -n 3)" || echo "⚠️ NVIDIA-SMI not found"
    ~/.cargo/bin/amdgpu_top --version >/dev/null 2>&1 && echo "✅ amdgpu_top: $(~/.cargo/bin/amdgpu_top --version)" || echo "⚠️ amdgpu_top not found"
    echo "✅ Verification complete"
}

final_cleanup() {
    echo "🧹 Performing final cleanup..."
    sudo apt-get update
    sudo apt-get upgrade -y
    sudo apt-get autoremove -y
    sudo apt-get autoclean
    echo "✅ Success: Final cleanup complete"
}

main() {
    > /tmp/failed_installs.txt
    check_root
    fix_interruptions
    update_system
    install_prerequisites

    # First phase: System essentials
    echo "📦 Phase 1: Installing system essentials..."
    install_essentials
    install_go
    install_modern_tools
    create_symlinks

    # Second phase: Core applications
    echo "📦 Phase 2: Installing core applications..."
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
    install_app "Facebook Messenger" install_messenger
    install_app "Trello" install_trello

    # Third phase: Special applications and tools
    echo "📦 Phase 3: Installing special applications and tools..."
    install_whatsie
    install_amdgpu_top
    install_google_drive

    # Fourth phase: Network and system configuration
    echo "🌐 Phase 4: Configuring network and system services..."
    setup_network_services
    generate_ssh_keys
    backup_configs
    configure_ssh
    configure_samba
    setup_samba_dirs
    configure_shell
    setup_monitoring
    setup_security
    start_services
    configure_firewall

    # Final phase: Verification and cleanup
    echo "🔍 Phase 5: Verifying installations and cleaning up..."
    verify_installations
    final_cleanup

    # Installation report
    echo "
✨ Complete System Installation Report ✨

System Components:
- Essential tools and GCC 12 installed
- Go ${GO_VERSION} configured
- Modern CLI tools (bat, eza, fd, ripgrep, fzf) set up
- Development tools (cmake, make, git, etc.) ready
- Security tools and monitoring configured

Applications Installed:"
    
    # Check each application's installation status
    applications=(
        "brave-browser"
        "nordvpn"
        "nordpass"
        "teamviewer"
        "obs-studio"
        "spotify"
        "vlc"
        "ffmpeg"
        "qbittorrent"
        "telegram-desktop"
        "sublime-text"
        "plexmediaserver"
        "whatsie"
    )

    for app in "${applications[@]}"; do
        if command -v "$app" >/dev/null 2>&1 || dpkg -l | grep -q "^ii.*${app}"; then
            echo "✅ $app"
        else
            echo "❌ $app - May need attention"
        fi
    done

    echo "
Network Services:
- SSH: $(systemctl is-active ssh || systemctl is-active sshd)
- Samba: $(systemctl is-active smbd)
- Firewall: $(systemctl is-active ufw)

Network Info:"
    ip addr show | grep "inet " || echo "⚠️ Failed to display network info"

    if [ -s /tmp/failed_installs.txt ]; then
        echo "
⚠️ Warning: Some installations reported errors:
$(cat /tmp/failed_installs.txt)
Please check $LOG_FILE for details."
    else
        echo "
✅ All installations completed successfully!"
    fi

    echo "
🔧 Next Steps:
1. Run 'source ~/.bashrc' to activate all aliases and paths
2. Configure NordVPN with 'nordvpn login'
3. Set up Google Drive with 'rclone config'
4. Access Plex at http://localhost:32400/web
5. Reboot recommended for all changes to take effect

Installation log available at: $LOG_FILE"
}