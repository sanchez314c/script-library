#!/bin/bash

# Complete Ubuntu Setup Script
# Author: Cortana AI for Jason, updated by Grok 3 (xAI)
# Purpose: Full system setup with essentials, dev tools, Go, GCC 12, security, nvtop, Whatsie, amdgpu_top, and Google Drive (via rclone)
# Version: 1.0.9 - Last Updated: 2025-03-21

# Enable verbosity and error handling
set -x  # Trace every command
set -e  # Exit on any error

echo "🚀 Starting Complete Ubuntu Setup..."

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

fix_interruptions() {
    echo "🔧 Fixing interrupted installations..."
    echo "Configuring any interrupted dpkg packages..."
    sudo dpkg --configure -a || { echo "❌ Error: dpkg configure failed"; exit 1; }
    echo "✅ Success: Interrupted installations fixed"
}

update_system() {
    echo "🔄 Updating system..."
    echo "Updating package lists..."
    sudo apt-get update || { echo "❌ Error: Apt update failed"; exit 1; }
    echo "Upgrading installed packages..."
    sudo apt-get upgrade -y || { echo "❌ Error: Apt upgrade failed"; exit 1; }
    echo "✅ Success: System updated"
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
        gcc-12 g++-12 jq tree rsync zip bzip2 sudo man-db rustc cargo  # rustc/cargo for amdgpu_top
    )
    install_packages "${essential_packages[@]}"
    echo "Setting GCC 12 as default..."
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 || { echo "❌ Error: Failed to set gcc-12 as default"; exit 1; }
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 || { echo "❌ Error: Failed to set g++-12 as default"; exit 1; }
    echo "✅ Success: Essential packages, GCC 12, and additional tools installed"
}

install_go() {
    echo "📦 Installing latest Go..."
    GO_VERSION="1.22.0"  # Placeholder—update to latest from go.dev/dl in March 2025
    echo "Downloading Go $GO_VERSION..."
    wget -q https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz -O go.tar.gz || { echo "❌ Error: Go download failed"; exit 1; }
    echo "Extracting Go to /usr/local..."
    sudo tar -C /usr/local -xzf go.tar.gz || { echo "❌ Error: Go extraction failed"; exit 1; }
    rm go.tar.gz
    echo "Adding Go to PATH in .bashrc..."
    sudo -u "${SUDO_USER:-$USER}" bash -c "echo 'export PATH=/usr/local/go/bin:\$PATH' >> ~${SUDO_USER:-$USER}/.bashrc" || { echo "❌ Error: Failed to update .bashrc for Go"; exit 1; }
    echo "Verifying Go installation..."
    /usr/local/go/bin/go version || { echo "❌ Error: Go not functional"; exit 1; }
    echo "✅ Success: Latest Go installed"
}

install_modern_tools() {
    echo "🛠️ Installing modern CLI tools..."
    modern_tools=(bat eza fd-find ripgrep fzf)
    install_packages "${modern_tools[@]}"
    echo "✅ Success: Modern CLI tools installed"
}

create_symlinks() {
    echo "🔗 Creating symlinks for modern tools..."
    echo "Ensuring ~${SUDO_USER:-$USER}/.local/bin exists..."
    sudo -u "${SUDO_USER:-$USER}" mkdir -pv ~${SUDO_USER:-$USER}/.local/bin || { echo "❌ Error: Failed to create ~/.local/bin"; exit 1; }
    if [ -x "$(which batcat)" ]; then
        echo "Linking batcat to bat..."
        sudo -u "${SUDO_USER:-$USER}" ln -sfv "$(which batcat)" ~${SUDO_USER:-$USER}/.local/bin/bat || { echo "❌ Error: Failed to link bat"; exit 1; }
    else
        echo "⚠️ Warning: batcat not found—skipping bat symlink"
    fi
    if [ -x "$(which fdfind)" ]; then
        echo "Linking fdfind to fd..."
        sudo -u "${SUDO_USER:-$USER}" ln -sfv "$(which fdfind)" ~${SUDO_USER:-$USER}/.local/bin/fd || { echo "❌ Error: Failed to link fd"; exit 1; }
    else
        echo "⚠️ Warning: fdfind not found—skipping fd symlink"
    fi
    echo "✅ Success: Symlinks created"
}

install_whatsie() {
    echo "📱 Installing Whatsie by Keshav Bhatt..."
    WHATSIE_VERSION="4.14.2"  # Latest stable as of last known release; check https://github.com/keshavbhatt/whatsie/releases for updates
    WHATSIE_URL="https://github.com/keshavbhatt/whatsie/releases/download/v${WHATSIE_VERSION}/whatsie_${WHATSIE_VERSION}_amd64.deb"
    echo "Downloading Whatsie v${WHATSIE_VERSION}..."
    wget -q "$WHATSIE_URL" -O whatsie.deb || { echo "❌ Error: Whatsie download failed"; exit 1; }
    echo "Installing Whatsie..."
    sudo dpkg -i whatsie.deb || {
        echo "⚠️ Warning: Whatsie install failed. Fixing dependencies..."
        sudo apt-get install -f || { echo "❌ Error: Failed to fix dependencies"; exit 1; }
        sudo dpkg -i whatsie.deb || { echo "❌ Error: Whatsie install failed after fixing"; exit 1; }
    }
    rm whatsie.deb
    echo "Verifying Whatsie installation..."
    if command -v whatsie >/dev/null 2>&1; then
        echo "✅ Success: Whatsie installed. Version:"
        whatsie --version
    else
        echo "❌ Error: Whatsie not found after install"
        exit 1
    fi
}

install_amdgpu_top() {
    echo "📊 Installing amdgpu_top..."
    echo "Installing amdgpu_top via cargo..."
    sudo -u "${SUDO_USER:-$USER}" cargo install amdgpu_top || { echo "❌ Error: amdgpu_top install failed"; exit 1; }
    echo "Adding cargo bin to PATH in .bashrc..."
    sudo -u "${SUDO_USER:-$USER}" bash -c "echo 'export PATH=~/.cargo/bin:\$PATH' >> ~${SUDO_USER:-$USER}/.bashrc" || { echo "❌ Error: Failed to update .bashrc for cargo"; exit 1; }
    echo "Verifying amdgpu_top installation..."
    if ~/.cargo/bin/amdgpu_top --version >/dev/null 2>&1; then
        echo "✅ Success: amdgpu_top installed. Version:"
        ~/.cargo/bin/amdgpu_top --version
    else
        echo "❌ Error: amdgpu_top not found after install"
        exit 1
    fi
}

install_google_drive() {
    echo "💾 Installing Google Drive support via rclone..."
    install_packages rclone
    echo "Configuring rclone for Google Drive (manual step required)..."
    echo "Run 'rclone config' after this script to set up Google Drive. Use 'Google Drive' as the storage type and follow the OAuth prompts."
    echo "Adding rclone alias to .bashrc..."
    sudo -u "${SUDO_USER:-$USER}" bash -c "echo 'alias gdrive=\"rclone --vfs-cache-mode writes mount googledrive: ~/GoogleDrive\"' >> ~${SUDO_USER:-$USER}/.bashrc" || { echo "❌ Error: Failed to update .bashrc for rclone"; exit 1; }
    echo "Creating Google Drive mount point..."
    sudo -u "${SUDO_USER:-$USER}" mkdir -pv ~${SUDO_USER:-$USER}/GoogleDrive || { echo "❌ Error: Failed to create GoogleDrive dir"; exit 1; }
    echo "Verifying rclone installation..."
    rclone --version || { echo "❌ Error: rclone not functional"; exit 1; }
    echo "✅ Success: rclone installed for Google Drive. Complete setup with 'rclone config'."
}

setup_network_services() {
    echo "🌐 Setting up network services..."
    network_services=(openssh-server samba samba-common-bin)
    install_packages "${network_services[@]}"
    if ! dpkg -l | grep -q "^ii  openssh-server "; then
        echo "❌ Error: openssh-server failed to install correctly"
        exit 1
    fi
    echo "✅ Success: Network services installed"
}

detect_ssh_service() {
    if systemctl list-unit-files | grep -q "ssh.service"; then
        echo "ssh"
    elif systemctl list-unit-files | grep -q "sshd.service"; then
        echo "sshd"
    else
        echo "❌ Error: Cannot find SSH service"
        sudo systemctl daemon-reload
        if systemctl list-unit-files | grep -q "ssh.service"; then
            echo "ssh"
        elif systemctl list-unit-files | grep -q "sshd.service"; then
            echo "sshd"
        else
            echo "UNKNOWN"
        fi
    fi
}

generate_ssh_keys() {
    echo "🔑 Generating SSH host keys..."
    sudo ssh-keygen -A || { echo "❌ Error: Failed to generate SSH host keys"; exit 1; }
    echo "✅ Success: SSH host keys generated"
}

backup_configs() {
    timestamp=$(date +%Y%m%d_%H%M%S)
    echo "💾 Backing up configs with timestamp $timestamp..."
    if [ -f /etc/ssh/sshd_config ]; then
        echo "Backing up /etc/ssh/sshd_config..."
        sudo cp -v /etc/ssh/sshd_config "/etc/ssh/sshd_config.bak.$timestamp" || { echo "❌ Error: Failed to backup sshd_config"; exit 1; }
    fi
    if [ -f /etc/samba/smb.conf ]; then
        echo "Backing up /etc/samba/smb.conf..."
        sudo cp -v /etc/samba/smb.conf "/etc/samba/smb.conf.bak.$timestamp" || { echo "❌ Error: Failed to backup smb.conf"; exit 1; }
    fi
    if [ -f ~${SUDO_USER:-$USER}/.bashrc ]; then
        echo "Backing up ~${SUDO_USER:-$USER}/.bashrc..."
        sudo -u "${SUDO_USER:-$USER}" cp -v ~${SUDO_USER:-$USER}/.bashrc ~${SUDO_USER:-$USER}/.bashrc.backup.$timestamp || { echo "❌ Error: Failed to backup .bashrc"; exit 1; }
    fi
    echo "✅ Success: Configs backed up"
}

configure_ssh() {
    echo "🔒 Configuring SSH..."
    echo "Writing SSH config to /etc/ssh/sshd_config..."
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
    echo "Testing SSH config..."
    if sudo sshd -t; then
        echo "✅ Success: SSH config valid"
    else
        echo "❌ Error: SSH config invalid"
        exit 1
    fi
    echo "Verifying SSH daemon..."
    if ! pgrep -f sshd > /dev/null; then
        echo "⚠️ Warning: SSH daemon not running"
        SSH_SERVICE=$(detect_ssh_service)
        if [ "$SSH_SERVICE" != "UNKNOWN" ]; then
            echo "Restarting SSH service ($SSH_SERVICE)..."
            sudo systemctl restart "$SSH_SERVICE" || echo "❌ Error: Failed to restart SSH"
        else
            echo "❌ Error: Could not determine SSH service to restart"
        fi
    fi
    echo "✅ Success: SSH configured"
}

configure_samba() {
    echo "📂 Setting up Samba..."
    echo "Writing Samba config to /etc/samba/smb.conf..."
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
    echo "✅ Success: Samba config written"
}

setup_samba_dirs() {
    echo "📁 Creating Samba directories..."
    echo "Creating /samba/public..."
    sudo mkdir -pv /samba/public || { echo "❌ Error: Failed to create /samba/public"; exit 1; }
    echo "Setting permissions on /samba/public..."
    sudo chmod -v 777 /samba/public || { echo "❌ Error: Failed to set permissions on /samba/public"; exit 1; }
    RAID_PATH="/media/heathen-admin/llmRAID"
    if [ ! -d "$RAID_PATH" ]; then
        echo "Creating $RAID_PATH..."
        sudo mkdir -pv "$RAID_PATH" || { echo "❌ Error: Failed to create $RAID_PATH"; exit 1; }
        echo "Setting ownership on $RAID_PATH..."
        sudo chown -v heathen-admin:heathen-admin "$RAID_PATH" || { echo "❌ Error: Failed to set ownership on $RAID_PATH"; exit 1; }
        echo "Setting permissions on $RAID_PATH..."
        sudo chmod -v 775 "$RAID_PATH" || { echo "❌ Error: Failed to set permissions on $RAID_PATH"; exit 1; }
    else
        echo "✅ $RAID_PATH already exists—skipping creation"
    fi
    echo "✅ Success: Samba directories set up"
}

configure_shell() {
    echo "🔧 Configuring shell aliases..."
    echo "Appending aliases to ~${SUDO_USER:-$USER}/.bashrc..."
    sudo -u "${SUDO_USER:-$USER}" bash -c "echo -e '# Modern CLI Aliases\nalias ls=\"eza --icons --group-directories-first\"\nalias ll=\"eza -l --icons --group-directories-first\"\nalias la=\"eza -la --icons --group-directories-first\"\nalias cat=\"bat --style=full --paging=never\"\nalias find=\"fd\"\nalias grep=\"rg\"\n# System Monitoring Aliases\nalias htop=\"htop -t\"\nalias diskspace=\"df -h\"\nalias memusage=\"free -h\"\nalias sysinfo=\"neofetch\"\nalias monitor=\"glances\"\nalias netwatch=\"netdata\"\nalias gputop=\"nvtop\"\nalias amdtop=\"amdgpu_top\"\n# Google Drive Alias\nalias gdrive=\"rclone --vfs-cache-mode writes mount googledrive: ~/GoogleDrive\"' >> ~${SUDO_USER:-$USER}/.bashrc" || { echo "❌ Error: Failed to append aliases to .bashrc"; exit 1; }
    echo "✅ Success: Shell aliases configured"
}

setup_monitoring() {
    echo "📊 Setting up monitoring tools..."
    monitoring_tools=(neofetch glances nvtop)
    install_packages "${monitoring_tools[@]}"
    echo "Creating neofetch config directory..."
    sudo -u "${SUDO_USER:-$USER}" mkdir -pv ~${SUDO_USER:-$USER}/.config/neofetch || { echo "❌ Error: Failed to create neofetch config dir"; exit 1; }
    echo "Writing neofetch config..."
    sudo -u "${SUDO_USER:-$USER}" bash -c "echo -e 'print_info() {\n    info title\n    info underline\n    info \"OS\" distro\n    info \"Host\" model\n    info \"Kernel\" kernel\n    info \"Uptime\" uptime\n    info \"Packages\" packages\n    info \"Shell\" shell\n    info \"CPU\" cpu\n    info \"GPU\" gpu\n    info \"Memory\" memory\n    info \"Disk\" disk\n    info \"Local IP\" local_ip\n}' > ~${SUDO_USER:-$USER}/.config/neofetch/config.conf" || { echo "❌ Error: Failed to write neofetch config"; exit 1; }
    echo "✅ Success: Monitoring tools (including nvtop) set up"
}

setup_security() {
    echo "🔒 Setting up security tools..."
    security_tools=(unattended-upgrades apt-listchanges)
    install_packages "${security_tools[@]}"
    echo "Configuring unattended upgrades..."
    echo -e 'APT::Periodic::Update-Package-Lists "1";\nAPT::Periodic::Download-Upgradeable-Packages "1";\nAPT::Periodic::AutocleanInterval "7";\nAPT::Periodic::Unattended-Upgrade "1";' | sudo tee /etc/apt/apt.conf.d/20auto-upgrades > /dev/null || { echo "❌ Error: Failed to configure unattended upgrades"; exit 1; }
    echo "✅ Success: Security tools configured"
}

configure_firewall() {
    echo "🧱 Configuring firewall..."
    install_packages ufw
    echo "Setting default deny incoming..."
    sudo ufw default deny incoming || { echo "❌ Error: Failed to set default deny incoming"; exit 1; }
    echo "Setting default allow outgoing..."
    sudo ufw default allow outgoing || { echo "❌ Error: Failed to set default allow outgoing"; exit 1; }
    echo "Allowing SSH..."
    sudo ufw allow ssh || { echo "❌ Error: Failed to allow SSH"; exit 1; }
    echo "Allowing Samba..."
    sudo ufw allow samba || { echo "❌ Error: Failed to allow Samba"; exit 1; }
    echo "Allowing port 11434/tcp (Ollama)..."
    sudo ufw allow 11434/tcp || { echo "❌ Error: Failed to allow port 11434"; exit 1; }
    echo "Allowing port 3000/tcp (OpenWebUI)..."
    sudo ufw allow 3000/tcp || { echo "❌ Error: Failed to allow port 3000"; exit 1; }
    echo "Allowing port 8080/tcp (OpenWebUI alt)..."
    sudo ufw allow 8080/tcp || { echo "❌ Error: Failed to allow port 8080"; exit 1; }
    echo "Verifying SSH is working before enabling firewall..."
    SSH_SERVICE=$(detect_ssh_service)
    if [ "$SSH_SERVICE" = "UNKNOWN" ]; then
        echo "⚠️ Warning: SSH service not detected, not enabling firewall"
        return
    fi
    if ! systemctl is-active "$SSH_SERVICE" >/dev/null; then
        echo "⚠️ Warning: SSH service not active, attempting to start..."
        sudo systemctl start "$SSH_SERVICE" || {
            echo "❌ Error: Failed to start SSH service, not enabling firewall"
            return
        }
    fi
    echo "SSH service verified as running. Enabling UFW..."
    echo "y" | sudo ufw enable || { echo "❌ Error: Failed to enable UFW"; exit 1; }
    echo "✅ Success: Firewall configured"
}

start_services() {
    echo "🌐 Starting services..."
    SSH_SERVICE=$(detect_ssh_service)
    if [ "$SSH_SERVICE" = "UNKNOWN" ]; then
        echo "❌ Error: Cannot determine SSH service name"
        echo "Attempting to reinstall SSH server..."
        sudo apt-get install --reinstall openssh-server
        SSH_SERVICE=$(detect_ssh_service)
    fi
    if [ "$SSH_SERVICE" != "UNKNOWN" ]; then
        echo "Processing SSH service: $SSH_SERVICE..."
        echo "Stopping $SSH_SERVICE if running..."
        sudo systemctl stop "$SSH_SERVICE" 2>/dev/null || echo "⚠️ $SSH_SERVICE wasn't running"
        echo "Starting $SSH_SERVICE..."
        sudo systemctl start "$SSH_SERVICE" || { 
            echo "❌ Error: Failed to start $SSH_SERVICE"; 
            sudo systemctl status "$SSH_SERVICE" --no-pager; 
            exit 1;
        }
        echo "Enabling $SSH_SERVICE..."
        sudo systemctl enable "$SSH_SERVICE" || { echo "❌ Error: Failed to enable $SSH_SERVICE"; exit 1; }
        echo "✅ Success: $SSH_SERVICE started and enabled"
    else
        echo "❌ Error: SSH service still not found after reinstall attempt"
        exit 1
    fi
    for svc in smbd nmbd; do
        echo "Processing service: $svc..."
        echo "Stopping $svc if running..."
        sudo systemctl stop "$svc" 2>/dev/null || echo "⚠️ $svc wasn't running"
        echo "Starting $svc..."
        sudo systemctl start "$svc" || { echo "❌ Error: Failed to start $svc"; sudo systemctl status "$svc" --no-pager; exit 1; }
        echo "Enabling $svc..."
        sudo systemctl enable "$svc" || { echo "❌ Error: Failed to enable $svc"; exit 1; }
        echo "✅ Success: $svc started and enabled"
    done
    echo "✅ Success: All services processed"
}

verify_ssh_connection() {
    echo "🔍 Verifying SSH connection..."
    SSH_SERVICE=$(detect_ssh_service)
    if [ "$SSH_SERVICE" = "UNKNOWN" ]; then
        echo "❌ Error: SSH service not detected"
        return 1
    fi
    if ! systemctl is-active "$SSH_SERVICE" >/dev/null; then
        echo "❌ Error: SSH service not active"
        return 1
    fi
    echo "Checking for SSH listening port..."
    if ! netstat -tuln | grep -q ":22 "; then
        echo "❌ Error: SSH not listening on port 22"
        return 1
    fi
    echo "✅ Success: SSH appears to be properly configured"
    return 0
}

final_cleanup() {
    echo "🧹 Performing final cleanup..."
    echo "Updating package lists..."
    sudo apt-get update || { echo "❌ Error: Final apt update failed"; exit 1; }
    echo "Upgrading installed packages..."
    sudo apt-get upgrade -y || { echo "❌ Error: Final apt upgrade failed"; exit 1; }
    echo "Running autoremove..."
    sudo apt-get autoremove -y || { echo "❌ Error: Final autoremove failed"; exit 1; }
    echo "Cleaning apt cache..."
    sudo apt-get autoclean || { echo "❌ Error: Final autoclean failed"; exit 1; }
    echo "✅ Success: Final cleanup complete"
}

main() {
    echo "🔧 Entering main function..."
    check_root
    fix_interruptions
    update_system
    install_essentials
    install_go
    install_modern_tools
    create_symlinks
    install_whatsie
    install_amdgpu_top
    install_google_drive
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
    verify_ssh_connection
    configure_firewall
    final_cleanup
    echo "
✨ Ubuntu Setup Complete! ✨
- Essentials, dev tools (including Go $GO_VERSION and GCC 12), and security installed
- Additional tools (cmake, make, jq, tree, rsync, zip, bzip2, sudo, man-db) installed
- Whatsie (WhatsApp client), amdgpu_top, and Google Drive (via rclone) installed
- Network services (SSH, Samba) configured
- Monitoring tools (including nvtop) ready
- Run 'source ~${SUDO_USER:-$USER}/.bashrc' for aliases, Go, and cargo PATH
- Try 'go version', 'gcc --version', 'jq --version', 'tree', 'rsync --version', 'whatsie', 'amdgpu_top', 'gdrive', 'sysinfo', 'monitor', 'netwatch', or 'gputop'
Network Info:"
    ip addr show | grep "inet " || echo "⚠️ Warning: Failed to display network info"
    echo "
📡 SSH Status:"
    SSH_SERVICE=$(detect_ssh_service)
    [ "$SSH_SERVICE" != "UNKNOWN" ] && systemctl status "$SSH_SERVICE" --no-pager || echo "❌ SSH service not found"
    echo "SSH Port:"
    netstat -tuln | grep ":22 " || echo "❌ No service listening on port 22"
    echo "✅ Setup done! Enjoy your system, J!"
}

main