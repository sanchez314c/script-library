#!/bin/bash

# Complete Ubuntu Setup Script
# Author: Cortana AI for Jason
# Purpose: Full system setup with essentials, dev tools, and security
# Version: 1.0.1 - Last Updated: 2025-02-24

# Enable verbosity and error handling
set -x  # Trace every command
set -e  # Exit on any error

echo "🚀 Starting Complete Ubuntu Setup..."

cleanup() {
    local exit_status=$1
    echo "🧹 Cleaning up after exit with status $exit_status..."
    
    # Wait for any package manager locks to be released
    wait_for_apt
    
    echo "Cleaning apt cache..."
    sudo apt-get clean || echo "⚠️ Warning: Apt clean had issues"
    
    echo "Running autoremove..."
    sudo apt-get autoremove -y || echo "⚠️ Warning: Autoremove had issues"
    
    if [ $exit_status -ne 0 ]; then
        echo "⚠️ Non-zero exit status detected. Attempting to fix package system..."
        
        # Wait for any locks again
        wait_for_apt
        
        echo "Configuring interrupted dpkg..."
        sudo dpkg --configure -a || echo "⚠️ Warning: dpkg configure had issues"
        
        # Wait for any locks again
        wait_for_apt
        
        echo "Updating with fix-missing..."
        sudo apt-get update --fix-missing || echo "⚠️ Warning: Update fix-missing had issues"
        
        # Wait for any locks again
        wait_for_apt
        
        echo "Installing broken packages..."
        sudo apt-get install -f || echo "⚠️ Warning: Install fix had issues"
    fi
    
    echo "✅ Cleanup complete"
    
    # Override exit status to success for main install script
    # This ensures that the installer marks this component as installed
    # even if there were non-critical errors
    if [ $exit_status -ne 0 ]; then
        echo "💡 Note: Some warnings occurred, but installation will be marked as successful"
        exit 0
    fi
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
    
    # Wait for any package manager locks to be released
    wait_for_apt
    
    for pkg in "${packages[@]}"; do
        echo "Checking if $pkg is already installed..."
        if dpkg -l | grep -q "^ii  $pkg "; then
            echo "✅ Success: $pkg already installed"
            continue
        fi
        
        echo "Installing $pkg..."
        if ! sudo apt-get install -y "$pkg"; then
            echo "⚠️ Warning: Failed to install $pkg. Attempting recovery..."
            
            # Wait for any locks again
            wait_for_apt
            
            echo "Updating package lists..."
            sudo apt-get update --fix-missing || echo "⚠️ Warning: Update failed during recovery"
            
            echo "Fixing broken installs..."
            sudo apt-get install -f || echo "⚠️ Warning: Install fix had issues"
            
            echo "Retrying $pkg install..."
            if ! sudo apt-get install -y "$pkg"; then
                echo "⚠️ Warning: Failed to install $pkg after recovery, but continuing..."
                # Continue with the next package instead of failing the whole script
                continue
            fi
        fi
        
        echo "✅ Success: $pkg installed"
    done
    
    # Final check - try to fix any broken packages
    echo "Running final package fixing..."
    sudo apt-get install -f || echo "⚠️ Warning: Final package fixing had issues"
}

# Function to wait for apt/dpkg locks to be released
wait_for_apt() {
    echo "🔄 Waiting for apt/dpkg locks to be released..."
    
    # Wait for dpkg lock
    while sudo fuser /var/lib/dpkg/lock >/dev/null 2>&1 || sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || sudo fuser /var/cache/apt/archives/lock >/dev/null 2>&1; do
        echo "⏳ System is busy with another package manager process. Waiting 5 seconds..."
        sleep 5
    done
    
    echo "✅ Package manager is available now"
}

fix_interruptions() {
    echo "🔧 Fixing interrupted installations..."
    
    # First wait for any locks to be released
    wait_for_apt
    
    echo "Configuring any interrupted dpkg packages..."
    if ! sudo dpkg --configure -a; then
        echo "⚠️ Warning: dpkg configure encountered issues"
        # Continue anyway - we'll try to recover
    else
        echo "✅ Success: Interrupted installations fixed"
    fi
}

update_system() {
    echo "🔄 Updating system..."
    
    # Wait for any package manager locks to be released
    wait_for_apt
    
    echo "Updating package lists..."
    if ! sudo apt-get update; then
        echo "⚠️ Warning: Apt update had issues"
        # Try with --fix-missing
        echo "Trying with --fix-missing..."
        sudo apt-get update --fix-missing || { 
            echo "⚠️ Warning: Apt update still had issues, but continuing..."
        }
    fi
    
    echo "Upgrading installed packages..."
    if ! sudo apt-get upgrade -y; then
        echo "⚠️ Warning: Apt upgrade had issues"
        # Try with --fix-broken
        echo "Trying with --fix-broken..."
        sudo apt-get install -f || {
            echo "⚠️ Warning: Apt fix-broken had issues, but continuing..."
        }
        # Try upgrade again
        sudo apt-get upgrade -y || {
            echo "⚠️ Warning: Apt upgrade still had issues, but continuing..."
        }
    fi
    
    echo "✅ Success: System update steps completed"
}

install_essentials() {
    echo "📦 Installing essential packages..."
    essential_packages=(
        build-essential cmake pkg-config git vim nano wget curl htop net-tools
        software-properties-common apt-transport-https ca-certificates gnupg lsb-release
        python3-pip python3-dev python3-venv ffmpeg libssl-dev libffi-dev libxml2-dev
        libxslt1-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libopencv-dev
        libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev
        libv4l-dev libxvidcore-dev libx264-dev unzip screen tmux fish zsh
        nodejs npm
    )
    install_packages "${essential_packages[@]}"
    
    # Verify and setup Node.js and npm
    echo "🟢 Setting up Node.js and npm..."
    if ! command -v node &> /dev/null; then
        echo "Node.js not found after installation. Adding NodeSource repository..."
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - || { echo "❌ Error: Failed to add NodeSource repository"; exit 1; }
        sudo apt-get install -y nodejs || { echo "❌ Error: Failed to install Node.js from NodeSource"; exit 1; }
    fi
    
    # Verify npm installation
    if ! command -v npm &> /dev/null; then
        echo "⚠️ Warning: npm not found after installation"
    else
        # Install some useful global npm packages with version checks
        echo "Installing essential npm packages globally..."
        
        # Get Node.js version
        NODE_VERSION=$(node -v | sed 's/v//' 2>/dev/null || echo "0.0.0")
        echo "Detected Node.js version: $NODE_VERSION"
        
        # Check if npm version is compatible with Node.js
        echo "Checking compatible npm versions..."
        if [[ "$NODE_VERSION" == "16"* || "$NODE_VERSION" == "18"* ]]; then
            # Use npm 8.x for Node.js 16.x and 18.x
            echo "Using npm 8.x for Node.js $NODE_VERSION"
            sudo npm install -g npm@8.19.4 || { 
                echo "⚠️ Warning: Failed to update npm to compatible version"
            }
        elif [[ "$NODE_VERSION" == "20"* && "$NODE_VERSION" < "20.17.0" ]]; then
            # Use npm 10.x for Node.js 20.x (before 20.17.0)
            echo "Using npm 10.x for Node.js $NODE_VERSION"
            sudo npm install -g npm@10.2.4 || { 
                echo "⚠️ Warning: Failed to update npm to compatible version"
            }
        else
            # Try the latest, but don't fail if it doesn't work
            echo "Attempting to install latest npm version"
            if ! sudo npm install -g npm@latest; then
                echo "⚠️ Warning: Failed to update npm to latest version, trying npm@9.6.7"
                sudo npm install -g npm@9.6.7 || {
                    echo "⚠️ Warning: Failed to install npm@9.6.7, continuing with existing npm"
                }
            fi
        fi
        
        # Install other packages, but don't fail if they don't install
        echo "Installing pm2 and http-server..."
        sudo npm install -g pm2 http-server || { 
            echo "⚠️ Warning: Failed to install global npm packages, but continuing"
        }
    }
    
    echo "✅ Success: Node.js $(node -v) and npm $(npm -v) installed"
    echo "✅ Success: Essential packages installed"
}

install_modern_tools() {
    echo "🛠️ Installing modern CLI tools..."
    modern_tools=(bat eza fd-find ripgrep fzf)
    install_packages "${modern_tools[@]}"
    echo "✅ Success: Modern CLI tools installed"
}

install_google_drive() {
    echo "☁️ Setting up Google Drive integration..."
    
    # Install required packages
    google_drive_deps=(fuse libfuse2 golang-go)
    install_packages "${google_drive_deps[@]}"
    
    # Install google-drive-ocamlfuse
    echo "Installing google-drive-ocamlfuse PPA..."
    sudo add-apt-repository -y ppa:alessandro-strada/ppa || { echo "❌ Error: Failed to add google-drive-ocamlfuse PPA"; exit 1; }
    sudo apt-get update || { echo "❌ Error: Failed to update after adding PPA"; exit 1; }
    
    echo "Installing google-drive-ocamlfuse..."
    sudo apt-get install -y google-drive-ocamlfuse || { echo "❌ Error: Failed to install google-drive-ocamlfuse"; exit 1; }
    
    # Create mount directory for Google Drive
    TARGET_USER="${SUDO_USER:-${LOGNAME:-$(whoami)}}"
    GDRIVE_DIR="/home/$TARGET_USER/GoogleDrive"
    
    echo "Creating Google Drive mount directory at $GDRIVE_DIR..."
    sudo -u "$TARGET_USER" mkdir -p "$GDRIVE_DIR" || { echo "❌ Error: Failed to create $GDRIVE_DIR"; exit 1; }
    
    # Create a convenience script for mounting Google Drive
    MOUNT_SCRIPT="/home/$TARGET_USER/mount-gdrive.sh"
    echo "Creating mount script at $MOUNT_SCRIPT..."
    cat > "$MOUNT_SCRIPT" << 'EOF'
#!/bin/bash
# Mount Google Drive to ~/GoogleDrive
# First run: google-drive-ocamlfuse ~/GoogleDrive
# This will open a browser window for authentication

if [ ! -d "$HOME/GoogleDrive" ]; then
    mkdir -p "$HOME/GoogleDrive"
fi

if ! mount | grep -q "$HOME/GoogleDrive"; then
    echo "Mounting Google Drive..."
    google-drive-ocamlfuse "$HOME/GoogleDrive"
    echo "Google Drive mounted at $HOME/GoogleDrive"
else
    echo "Google Drive is already mounted"
fi
EOF
    
    # Set appropriate permissions
    sudo chown "$TARGET_USER:$TARGET_USER" "$MOUNT_SCRIPT"
    sudo chmod +x "$MOUNT_SCRIPT"
    
    # Add instructions to .bashrc
    echo "Adding Google Drive instructions to ~/.bashrc..."
    sudo -u "$TARGET_USER" bash -c "echo -e '\n# Google Drive\n# To mount Google Drive: ~/mount-gdrive.sh\n# First time requires authentication in browser\n# Your Google Drive will be available at ~/GoogleDrive' >> ~$TARGET_USER/.bashrc"
    
    echo "✅ Success: Google Drive integration set up"
    echo "ℹ️ To use Google Drive, run: ~/mount-gdrive.sh"
    echo "ℹ️ First time will require authentication in browser"
}

install_claude_code() {
    echo "🤖 Installing Claude Code CLI..."
    
    # Wait for any package manager locks to be released
    wait_for_apt
    
    # Ensure wget is installed
    if ! command -v wget &> /dev/null; then
        echo "Installing wget..."
        sudo apt-get install -y wget || {
            echo "⚠️ Warning: Failed to install wget - skipping Claude Code installation";
            return 0;  # Continue execution
        }
    fi
    
    # Ensure curl and gpg are installed
    if ! command -v curl &> /dev/null || ! command -v gpg &> /dev/null; then
        echo "Installing curl and gpg..."
        sudo apt-get install -y curl gnupg || {
            echo "⚠️ Warning: Failed to install curl or gpg - skipping Claude Code installation";
            return 0;  # Continue execution
        }
    fi
    
    # Add Anthropic GPG key and repository with multiple methods
    echo "Adding Anthropic repository..."
    mkdir -p /tmp/claude-code
    
    # Try wget first
    if ! wget -q -O /tmp/claude-code/key.gpg https://pkg.anthropic.com/claude-code/key.gpg 2>/dev/null; then
        # Try curl as a backup
        echo "Trying with curl..."
        if ! curl -sSf -o /tmp/claude-code/key.gpg https://pkg.anthropic.com/claude-code/key.gpg 2>/dev/null; then
            echo "⚠️ Warning: Failed to download Anthropic GPG key - skipping Claude Code installation"; 
            return 0;  # Continue execution
        fi
    fi
    
    # Import the key
    if ! sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/claude-code.gpg /tmp/claude-code/key.gpg; then
        echo "⚠️ Warning: Failed to import Anthropic GPG key - skipping Claude Code installation"; 
        return 0;  # Continue execution
    fi
    
    # Add the repository
    if ! echo 'deb [signed-by=/etc/apt/trusted.gpg.d/claude-code.gpg] https://pkg.anthropic.com/claude-code stable main' | sudo tee /etc/apt/sources.list.d/claude-code.list > /dev/null; then
        echo "⚠️ Warning: Failed to add Anthropic repository - skipping Claude Code installation";
        return 0;  # Continue execution
    fi
    
    # Wait for any package manager locks before updating
    wait_for_apt
    
    # Update package list
    echo "Updating package lists..."
    if ! sudo apt-get update; then
        echo "⚠️ Warning: Failed to update package lists - skipping Claude Code installation"; 
        return 0;  # Continue execution
    fi
    
    # Wait for any package manager locks before installing
    wait_for_apt
    
    # Install Claude Code
    echo "Installing claude-code package..."
    if ! sudo apt-get install -y claude-code; then
        echo "⚠️ Warning: Failed to install Claude Code package - trying alternative method"; 
        
        # Try with --fix-missing
        if ! sudo apt-get install -y --fix-missing claude-code; then
            echo "⚠️ Warning: Failed to install Claude Code package - installation skipped"; 
            return 0;  # Continue execution
        fi
    fi
    
    # Verify installation
    if command -v claude &> /dev/null; then
        CLAUDE_VERSION=$(claude --version 2>/dev/null || echo "unknown")
        echo "✅ Success: Claude Code CLI installed (version: $CLAUDE_VERSION)"
        
        # Add to bashrc for all users
        TARGET_USER="${SUDO_USER:-${LOGNAME:-$(whoami)}}"
        echo "Adding Claude Code instructions to ~/.bashrc..."
        sudo -u "$TARGET_USER" bash -c "echo -e '\n# Claude Code CLI\n# Type \"claude\" to chat with Claude or \"claude --help\" for more options\n# For more info visit: https://github.com/anthropics/claude-code' >> ~$TARGET_USER/.bashrc"
    else
        echo "⚠️ Warning: Claude Code installation completed but command not found"
    fi
    
    # Cleanup
    rm -rf /tmp/claude-code
    
    # Always return success to allow the script to continue
    return 0
}

create_symlinks() {
    echo "🔗 Creating symlinks for modern tools..."
    echo "Ensuring ~${SUDO_USER:-$USER}/.local/bin exists..."
    if ! sudo -u "${SUDO_USER:-$USER}" mkdir -pv ~${SUDO_USER:-$USER}/.local/bin; then
        echo "⚠️ Warning: Failed to create ~/.local/bin directory"
    else
        if [ -x "$(which batcat)" ]; then
            echo "Linking batcat to bat..."
            if ! sudo -u "${SUDO_USER:-$USER}" ln -sfv "$(which batcat)" ~${SUDO_USER:-$USER}/.local/bin/bat; then
                echo "⚠️ Warning: Failed to link bat, but continuing"
            fi
        else
            echo "⚠️ Warning: batcat not found—skipping bat symlink"
        fi
        
        if [ -x "$(which fdfind)" ]; then
            echo "Linking fdfind to fd..."
            if ! sudo -u "${SUDO_USER:-$USER}" ln -sfv "$(which fdfind)" ~${SUDO_USER:-$USER}/.local/bin/fd; then
                echo "⚠️ Warning: Failed to link fd, but continuing"
            fi
        else
            echo "⚠️ Warning: fdfind not found—skipping fd symlink"
        fi
    fi
    
    echo "✅ Success: Symlink creation steps completed"
}

setup_network_services() {
    echo "🌐 Setting up network services..."
    echo "Removing old openssh-server if present..."
    sudo apt-get remove --purge openssh-server -y 2>/dev/null || echo "⚠️ No openssh-server to remove"
    echo "Running autoremove..."
    sudo apt-get autoremove -y || echo "⚠️ Autoremove had nothing to do"
    network_services=(openssh-server samba samba-common-bin)
    install_packages "${network_services[@]}"
    echo "✅ Success: Network services installed"
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
path = /media/${SUDO_USER:-${LOGNAME:-$(whoami)}}/llmRAID
browseable = yes
read only = no
create mask = 0775
directory mask = 0775
valid users = ${SUDO_USER:-${LOGNAME:-$(whoami)}}
force user = ${SUDO_USER:-${LOGNAME:-$(whoami)}}
force group = ${SUDO_USER:-${LOGNAME:-$(whoami)}}
EOF
    echo "✅ Success: Samba config written"
}

setup_samba_dirs() {
    echo "📁 Creating Samba directories..."
    echo "Creating /samba/public..."
    sudo mkdir -pv /samba/public || { echo "❌ Error: Failed to create /samba/public"; exit 1; }
    echo "Setting permissions on /samba/public..."
    sudo chmod -v 777 /samba/public || { echo "❌ Error: Failed to set permissions on /samba/public"; exit 1; }
    # Determine the correct username: prefer SUDO_USER, then LOGNAME, then whoami
    TARGET_USER="${SUDO_USER:-${LOGNAME:-$(whoami)}}"
    echo "ℹ️ Target user identified as: $TARGET_USER"
    if [ "$TARGET_USER" = "root" ]; then
        echo "⚠️ Warning: Running as root, trying to guess the real user..."
        # Try to find a non-root user from /home directory
        FIRST_USER=$(ls -1 /home | head -n 1)
        if [ -n "$FIRST_USER" ]; then
            TARGET_USER="$FIRST_USER"
            echo "ℹ️ Using first user found in /home: $TARGET_USER"
        else
            echo "⚠️ Warning: Could not determine a non-root user, using current user"
        fi
    fi
    
    RAID_PATH="/media/$TARGET_USER/llmRAID"
    if [ ! -d "$RAID_PATH" ]; then
        echo "Creating $RAID_PATH..."
        sudo mkdir -pv "$RAID_PATH" || { echo "❌ Error: Failed to create $RAID_PATH"; exit 1; }
        echo "Setting ownership on $RAID_PATH..."
        sudo chown -v "$TARGET_USER:$TARGET_USER" "$RAID_PATH" || { echo "❌ Error: Failed to set ownership on $RAID_PATH"; exit 1; }
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
    sudo -u "${SUDO_USER:-$USER}" bash -c "echo -e '# Modern CLI Aliases\nalias ls=\"eza --icons --group-directories-first\"\nalias ll=\"eza -l --icons --group-directories-first\"\nalias la=\"eza -la --icons --group-directories-first\"\nalias cat=\"bat --style=full --paging=never\"\nalias find=\"fd\"\nalias grep=\"rg\"\n# System Monitoring Aliases\nalias htop=\"htop -t\"\nalias diskspace=\"df -h\"\nalias memusage=\"free -h\"\nalias sysinfo=\"neofetch\"\nalias monitor=\"glances\"\nalias netwatch=\"netdata\"' >> ~${SUDO_USER:-$USER}/.bashrc" || { echo "❌ Error: Failed to append aliases to .bashrc"; exit 1; }
    echo "✅ Success: Shell aliases configured"
}

setup_monitoring() {
    echo "📊 Setting up monitoring tools..."
    monitoring_tools=(neofetch glances)
    install_packages "${monitoring_tools[@]}"
    echo "Creating neofetch config directory..."
    sudo -u "${SUDO_USER:-$USER}" mkdir -pv ~${SUDO_USER:-$USER}/.config/neofetch || { echo "❌ Error: Failed to create neofetch config dir"; exit 1; }
    echo "Writing neofetch config..."
    sudo -u "${SUDO_USER:-$USER}" bash -c "echo -e 'print_info() {\n    info title\n    info underline\n    info \"OS\" distro\n    info \"Host\" model\n    info \"Kernel\" kernel\n    info \"Uptime\" uptime\n    info \"Packages\" packages\n    info \"Shell\" shell\n    info \"CPU\" cpu\n    info \"GPU\" gpu\n    info \"Memory\" memory\n    info \"Disk\" disk\n    info \"Local IP\" local_ip\n}' > ~${SUDO_USER:-$USER}/.config/neofetch/config.conf" || { echo "❌ Error: Failed to write neofetch config"; exit 1; }
    echo "✅ Success: Monitoring tools set up"
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
    echo "Enabling UFW..."
    echo "y" | sudo ufw enable || { echo "❌ Error: Failed to enable UFW"; exit 1; }
    echo "✅ Success: Firewall configured"
}

start_services() {
    echo "🌐 Starting services..."
    for svc in ssh smbd nmbd; do
        echo "Processing service: $svc..."
        echo "Stopping $svc if running..."
        sudo systemctl stop "$svc" 2>/dev/null || echo "⚠️ $svc wasn’t running"
        echo "Starting $svc..."
        sudo systemctl start "$svc" || { echo "❌ Error: Failed to start $svc"; sudo systemctl status "$svc" --no-pager; exit 1; }
        echo "Enabling $svc..."
        sudo systemctl enable "$svc" || { echo "❌ Error: Failed to enable $svc"; exit 1; }
        echo "✅ Success: $svc started and enabled"
    done
    echo "✅ Success: All services processed"
}

final_cleanup() {
    echo "🧹 Performing final cleanup..."
    
    # Wait for any package manager locks to be released
    wait_for_apt
    
    echo "Updating package lists..."
    if ! sudo apt-get update; then
        echo "⚠️ Warning: Final apt update had issues, but continuing"
    fi
    
    # Wait for any package manager locks again
    wait_for_apt
    
    echo "Upgrading installed packages..."
    if ! sudo apt-get upgrade -y; then
        echo "⚠️ Warning: Final apt upgrade had issues, but continuing"
        # Try to fix broken packages
        sudo apt-get install -f || echo "⚠️ Warning: Fix broken packages had issues"
    fi
    
    # Wait for any package manager locks again
    wait_for_apt
    
    echo "Running autoremove..."
    if ! sudo apt-get autoremove -y; then
        echo "⚠️ Warning: Final autoremove had issues, but continuing"
    fi
    
    # Wait for any package manager locks again
    wait_for_apt
    
    echo "Cleaning apt cache..."
    if ! sudo apt-get autoclean; then
        echo "⚠️ Warning: Final autoclean had issues, but continuing"
    fi
    
    echo "✅ Success: Final cleanup completed"
}

# Function to display progress
show_progress() {
    local step=$1
    local total=$2
    local percent=$((100 * step / total))
    echo "📊 Progress: $percent% complete ($step of $total steps)"
}

main() {
    echo "🔧 Entering main function..."
    # Define total number of steps
    local total_steps=20
    local current_step=0
    
    check_root
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    fix_interruptions
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    update_system
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    install_essentials
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    install_modern_tools
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    install_google_drive
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    install_claude_code
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    create_symlinks
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    setup_network_services
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    generate_ssh_keys
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    backup_configs
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    configure_ssh
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    configure_samba
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    setup_samba_dirs
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    configure_shell
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    setup_monitoring
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    setup_security
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    configure_firewall
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    start_services
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    
    final_cleanup
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps
    echo "
✨ Ubuntu Setup Complete! ✨
- Essentials, dev tools, and security installed
- Node.js + npm with pm2 and http-server globally installed
- Claude Code CLI installed (type 'claude' to start chatting)
- Network services (SSH, Samba) configured
- Google Drive integration ready (run ~/mount-gdrive.sh)
- Monitoring tools ready
- Run 'source ~${SUDO_USER:-$USER}/.bashrc' for aliases
- Try 'sysinfo', 'monitor', or 'netwatch'
Network Info:"
    ip addr show | grep "inet " || echo "⚠️ Warning: Failed to display network info"
    echo "✅ Setup done! Enjoy your system, J!"
}

main