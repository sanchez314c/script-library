#!/bin/bash
###############################################################
#    _____ ____  ____  _____ *   * _____ ___ *    *          #
#   | ____/ ___|| ** )| **__| \ | |_   *|* _/ \  | |         #
#   |  *| \*__ \|  * \|  *| |  \| | | |  | / _ \ | |         #
#   | |___ ___) | |_) | |___| |\  | | |  | / ___ \| |___     #
#   |_____|____/|____/|_____|_| \_| |_| |___/_/   \_\____|   #
#                                                             #
###############################################################
# Complete Ubuntu Setup Script
# Version: 3.0.0
# Date: April 15, 2025
# Purpose: Full system setup with essentials, dev tools, security, monitoring, and utilities
# Enable error handling
set -e  # Exit on any error
# Get the real user even when running with sudo
REAL_USER="${SUDO_USER:-$USER}"
# Create log file on desktop
LOG_FILE="/home/${REAL_USER}/Desktop/essentials_setup_$(date +%Y%m%d_%H%M%S).log"
touch "$LOG_FILE"
chown $REAL_USER:$REAL_USER "$LOG_FILE"
chmod 644 "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "🚀 Starting Complete Ubuntu Setup..."
cleanup() {
    local exit_status=$1
    echo "🧹 Cleaning up after exit with status $exit_status..."
    echo "Cleaning apt cache..."
    apt-get clean || echo "⚠️ Warning: Apt clean failed"
    echo "Running autoremove..."
    apt-get autoremove -y || echo "⚠️ Warning: Autoremove failed"
    if [ $exit_status -ne 0 ]; then
        echo "❌ Error detected. Attempting to fix package system..."
        echo "Configuring interrupted dpkg..."
        dpkg --configure -a || echo "⚠️ Warning: dpkg configure failed"
        echo "Updating with fix-missing..."
        apt-get update --fix-missing || echo "⚠️ Warning: Update fix-missing failed"
        echo "Installing broken packages..."
        apt-get install -f || echo "⚠️ Warning: Install fix failed"
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
    
    # Update package database first
    apt-get update || {
        echo "⚠️ Warning: apt-get update failed. Trying with alternate sources..."
        sed -i 's/^# deb-src/deb-src/' /etc/apt/sources.list
        apt-get update || {
            echo "❌ Error: apt-get update still failed. Using --fix-missing option..."
            apt-get update --fix-missing || echo "⚠️ Warning: apt-get update still failing, but proceeding with installations"
        }
    }
    
    # Install packages one by one with error handling
    for pkg in "${packages[@]}"; do
        echo "Checking if $pkg is already installed..."
        if dpkg -l | grep -q "^ii  $pkg "; then
            echo "✅ Success: $pkg already installed"
            continue
        fi
        echo "Installing $pkg..."
        apt-get install -y "$pkg" || {
            echo "⚠️ Warning: Failed to install $pkg. Attempting recovery..."
            apt-get install -f -y
            apt-get install -y "$pkg" || { 
                echo "❌ Error: Failed to install $pkg after recovery"
                # Continue with other packages instead of exiting
                continue
            }
        }
        echo "✅ Success: $pkg installed"
    done
}
fix_interruptions() {
    echo "🔧 Fixing interrupted installations..."
    dpkg --configure -a || { echo "❌ Error: dpkg configure failed"; exit 1; }
    apt-get install -f -y || { echo "❌ Error: apt-get install -f failed"; exit 1; }
    echo "✅ Success: Interrupted installations fixed"
}
update_system() {
    echo "🔄 Updating system..."
    apt-get update || { 
        echo "⚠️ Warning: apt-get update failed. Using --fix-missing option..."
        apt-get update --fix-missing || { 
            echo "❌ Error: apt-get update failed even with --fix-missing"; 
            echo "Continuing with installation but some packages might not be available"
        }
    }
    echo "Upgrading installed packages..."
    apt-get upgrade -y || { 
        echo "⚠️ Warning: apt-get upgrade failed. Using --fix-broken option..."
        apt-get upgrade -y --fix-broken || { 
            echo "❌ Error: apt-get upgrade failed even with --fix-broken"; 
            echo "Continuing with installation but system might not be fully upgraded"
        }
    }
    echo "✅ Success: System updated as much as possible"
}
install_essentials() {
    echo "📦 Installing essential packages including GCC 12, Git LFS, and additional tools..."
    essential_packages=(
        build-essential 
        cmake 
        make 
        pkg-config 
        git 
        git-lfs
        vim 
        nano 
        wget 
        curl 
        htop 
        net-tools
        software-properties-common 
        apt-transport-https 
        ca-certificates 
        gnupg 
        lsb-release
        python3-pip 
        python3-dev 
        python3-venv 
        ffmpeg 
        libssl-dev 
        libffi-dev 
        libxml2-dev
        libxslt1-dev 
        zlib1g-dev 
        libbz2-dev 
        libreadline-dev 
        libsqlite3-dev 
        libopencv-dev
        libjpeg-dev 
        libpng-dev 
        libtiff-dev 
        libavcodec-dev 
        libavformat-dev 
        libswscale-dev
        libv4l-dev 
        libxvidcore-dev 
        libx264-dev 
        unzip 
        screen 
        tmux 
        fish 
        zsh
        gcc-12 
        g++-12
        gcc-11
        g++-11 
        jq 
        tree 
        rsync 
        zip 
        bzip2 
        sudo 
        man-db 
        rustc 
        cargo
        rustup
        npm
        dotnet-sdk-6.0
        dotnet-sdk-9.0
        libstdc++-12-dev 
        libtcmalloc-minimal4 
        nvtop 
        radeontop 
        rovclock 
        libopenmpi3 
        libdnnl-dev 
        ninja-build 
        libopenblas-dev
    )
    install_packages "${essential_packages[@]}"
    
    echo "Setting GCC 12 as default..."
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 || { 
        echo "⚠️ Warning: Failed to set gcc-12 as default"
    }
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 || { 
        echo "⚠️ Warning: Failed to set g++-12 as default"
    }
    update-alternatives --set gcc /usr/bin/gcc-12 || echo "⚠️ Warning: Failed to set gcc-12 as default"
    update-alternatives --set g++ /usr/bin/g++-12 || echo "⚠️ Warning: Failed to set g++-12 as default"
    
    # Also make GCC 11 available as an alternative
    echo "Adding GCC 11 as alternatives..."
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 90 || { 
        echo "⚠️ Warning: Failed to add gcc-11 as alternative"
    }
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 90 || { 
        echo "⚠️ Warning: Failed to add g++-11 as alternative"
    }
    
    # Initialize Git LFS
    echo "Initializing Git LFS..."
    sudo -u "${REAL_USER}" git lfs install || {
        echo "⚠️ Warning: Failed to initialize Git LFS. Trying another approach..."
        sudo -H -u "${REAL_USER}" bash -c "git lfs install" || {
            echo "❌ Error: Failed to initialize Git LFS"
        }
    }
    
    echo "✅ Success: Essential packages, GCC 12, and additional tools installed"
}
install_go() {
    echo "📦 Installing latest Go..."
    GO_VERSION="1.22.0"  # Update to latest if needed
    echo "Downloading Go $GO_VERSION..."
    wget -q https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz -O go.tar.gz || { 
        echo "❌ Error: Go download failed"; 
        return 1;
    }
    echo "Extracting Go to /usr/local..."
    tar -C /usr/local -xzf go.tar.gz || { 
        echo "❌ Error: Go extraction failed"; 
        rm -f go.tar.gz;
        return 1;
    }
    rm -f go.tar.gz
    
    # Add Go to PATH in .bashrc if not already there
    BASHRC_FILE="/home/${REAL_USER}/.bashrc"
    if ! grep -q "export PATH=/usr/local/go/bin" "$BASHRC_FILE"; then
        echo "Adding Go to PATH in .bashrc..."
        echo 'export PATH=/usr/local/go/bin:$PATH' >> "$BASHRC_FILE"
        chown ${REAL_USER}:${REAL_USER} "$BASHRC_FILE"
    fi
    
    echo "Verifying Go installation..."
    if /usr/local/go/bin/go version > /dev/null; then
        GO_VERSION_OUTPUT=$(/usr/local/go/bin/go version)
        echo "✅ Success: Go installed: $GO_VERSION_OUTPUT"
    else
        echo "❌ Error: Go not functional"
        return 1
    fi
}
install_modern_tools() {
    echo "🛠️ Installing modern CLI tools..."
    modern_tools=(bat eza fd-find ripgrep fzf)
    install_packages "${modern_tools[@]}"
    echo "✅ Success: Modern CLI tools installed"
}
create_symlinks() {
    echo "🔗 Creating symlinks for modern tools..."
    echo "Ensuring ~${REAL_USER}/.local/bin exists..."
    sudo -u "${REAL_USER}" mkdir -pv ~${REAL_USER}/.local/bin || { 
        echo "❌ Error: Failed to create ~/.local/bin"; 
        return 1;
    }
    
    if [ -x "$(which batcat)" ]; then
        echo "Linking batcat to bat..."
        sudo -u "${REAL_USER}" ln -sfv "$(which batcat)" ~${REAL_USER}/.local/bin/bat || { 
            echo "❌ Error: Failed to link bat"; 
        }
    else
        echo "⚠️ Warning: batcat not found—skipping bat symlink"
    fi
    
    if [ -x "$(which fdfind)" ]; then
        echo "Linking fdfind to fd..."
        sudo -u "${REAL_USER}" ln -sfv "$(which fdfind)" ~${REAL_USER}/.local/bin/fd || { 
            echo "❌ Error: Failed to link fd"; 
        }
    else
        echo "⚠️ Warning: fdfind not found—skipping fd symlink"
    fi
    
    # Add ~/.local/bin to PATH if not already there
    BASHRC_FILE="/home/${REAL_USER}/.bashrc"
    if ! grep -q "PATH=\"\$HOME/.local/bin:\$PATH\"" "$BASHRC_FILE"; then
        echo "Adding ~/.local/bin to PATH in .bashrc..."
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$BASHRC_FILE"
        chown ${REAL_USER}:${REAL_USER} "$BASHRC_FILE"
    fi
    
    echo "✅ Success: Symlinks created"
}
install_whatsie() {
    echo "📱 Installing Whatsie by Keshav Bhatt..."
    WHATSIE_VERSION="4.14.2"  # Update to latest if needed
    # Try to detect if already installed
    if command -v whatsie >/dev/null 2>&1; then
        INSTALLED_VERSION=$(whatsie --version 2>/dev/null | grep -oP "Version: \K[0-9.]+")
        if [ "$INSTALLED_VERSION" = "$WHATSIE_VERSION" ]; then
            echo "✅ Success: Whatsie $INSTALLED_VERSION already installed"
            return 0
        fi
    fi
    WHATSIE_URL="https://github.com/keshavbhatt/whatsie/releases/download/v${WHATSIE_VERSION}/whatsie_${WHATSIE_VERSION}_amd64.deb"
    echo "Downloading Whatsie v${WHATSIE_VERSION}..."
    wget -q "$WHATSIE_URL" -O whatsie.deb || { 
        echo "❌ Error: Whatsie download failed"; 
        return 1;
    }
    
    echo "Installing Whatsie..."
    dpkg -i whatsie.deb || {
        echo "⚠️ Warning: Whatsie install failed. Fixing dependencies..."
        apt-get install -f -y
        dpkg -i whatsie.deb || { 
            echo "❌ Error: Failed to install Whatsie after fixing dependencies"; 
            rm -f whatsie.deb;
            return 1;
        }
    }
    rm -f whatsie.deb
    
    if command -v whatsie >/dev/null 2>&1; then
        echo "✅ Success: Whatsie installed. Version: $(whatsie --version 2>/dev/null | grep -oP "Version: \K[0-9.]+")"
    else
        echo "❌ Error: Whatsie not found after install"
        return 1
    fi
}
install_amdgpu_top() {
    echo "📊 Installing amdgpu_top..."
    
    # First check if already installed
    if command -v amdgpu_top >/dev/null 2>&1; then
        INSTALLED_VERSION=$(amdgpu_top --version 2>&1 | head -1)
        echo "✅ Success: amdgpu_top already installed: $INSTALLED_VERSION"
        return 0
    fi
    
    # Ensure cargo is available
    if ! command -v cargo >/dev/null 2>&1; then
        echo "⚠️ Warning: cargo not found, installing rustc and cargo..."
        apt-get install -y rustc cargo || {
            echo "❌ Error: Failed to install rustc and cargo";
            return 1;
        }
    fi
    
    echo "Installing amdgpu_top via cargo..."
    sudo -u "${REAL_USER}" cargo install amdgpu_top || { 
        echo "❌ Error: amdgpu_top install failed"; 
        return 1;
    }
    
    # Add cargo bin to PATH if not already there
    BASHRC_FILE="/home/${REAL_USER}/.bashrc"
    if ! grep -q "export PATH=~/.cargo/bin" "$BASHRC_FILE"; then
        echo "Adding cargo bin to PATH in .bashrc..."
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$BASHRC_FILE"
        chown ${REAL_USER}:${REAL_USER} "$BASHRC_FILE"
    fi
    
    echo "Verifying amdgpu_top installation..."
    if /home/${REAL_USER}/.cargo/bin/amdgpu_top --version >/dev/null 2>&1; then
        AMDGPU_VERSION=$(/home/${REAL_USER}/.cargo/bin/amdgpu_top --version 2>&1 | head -1)
        echo "✅ Success: amdgpu_top installed: $AMDGPU_VERSION"
    else
        echo "❌ Error: amdgpu_top not found after install"
        return 1
    fi
}
install_claude_code_cli() {
    echo "🧠 Installing Claude Code CLI..."
    
    # Check if npm is installed
    if ! command -v npm >/dev/null 2>&1; then
        echo "⚠️ Warning: npm not found, installing npm..."
        apt-get install -y npm || {
            echo "❌ Error: Failed to install npm";
            return 1;
        }
    fi
    
    # Install Claude Code CLI globally
    echo "Installing @anthropic-ai/claude-code globally..."
    npm install -g @anthropic-ai/claude-code || {
        echo "❌ Error: Failed to install Claude Code CLI";
        return 1;
    }
    
    # Verify installation
    if command -v claude-code >/dev/null 2>&1; then
        echo "✅ Success: Claude Code CLI installed"
        claude-code --version
    else
        echo "❌ Error: Claude Code CLI not found after install"
        return 1
    fi
}
install_monitoring_tools() {
    echo "📊 Installing monitoring tools..."
    
    # Try both apt and snap for nvtop
    if ! command -v nvtop >/dev/null 2>&1; then
        echo "Installing nvtop via apt..."
        apt-get install -y nvtop || {
            echo "⚠️ Warning: nvtop apt install failed, trying snap..."
            snap install nvtop || {
                echo "❌ Error: Failed to install nvtop via both apt and snap";
            }
        }
    else
        echo "✅ nvtop already installed"
    fi
    
    # Install radeontop
    if ! command -v radeontop >/dev/null 2>&1; then
        echo "Installing radeontop..."
        apt-get install -y radeontop || {
            echo "❌ Error: Failed to install radeontop";
        }
    else
        echo "✅ radeontop already installed"
    fi
    
    # Install rovclock
    if ! command -v rovclock >/dev/null 2>&1; then
        echo "Installing rovclock..."
        apt-get install -y rovclock || {
            echo "❌ Error: Failed to install rovclock";
        }
    else
        echo "✅ rovclock already installed"
    fi
    
    echo "✅ Success: Monitoring tools installed"
}
install_google_drive() {
    echo "💾 Installing Google Drive support via rclone..."
    install_packages rclone
    
    echo "Creating Google Drive mount point..."
    sudo -u "${REAL_USER}" mkdir -pv "/home/${REAL_USER}/GoogleDrive" || { 
        echo "❌ Error: Failed to create GoogleDrive dir"; 
        return 1;
    }
    
    # Add rclone alias to .bashrc if not already there
    BASHRC_FILE="/home/${REAL_USER}/.bashrc"
    GDRIVE_ALIAS="alias gdrive=\"rclone --vfs-cache-mode writes mount googledrive: ~/GoogleDrive\""
    
    if ! grep -q "$GDRIVE_ALIAS" "$BASHRC_FILE"; then
        echo "Adding rclone alias to .bashrc..."
        echo "$GDRIVE_ALIAS" >> "$BASHRC_FILE"
        chown ${REAL_USER}:${REAL_USER} "$BASHRC_FILE"
    fi
    
    echo "✅ Success: rclone installed for Google Drive. Complete setup with 'rclone config'."
    echo "Configuring rclone for Google Drive (manual step required)..."
    echo "Run 'rclone config' after this script to set up Google Drive. Use 'Google Drive' as the storage type and follow the OAuth prompts."
}
setup_network_services() {
    echo "🌐 Setting up network services..."
    network_services=(openssh-server samba samba-common-bin)
    install_packages "${network_services[@]}"
    
    # Check SSH installed correctly
    if ! dpkg -l | grep -q "^ii  openssh-server "; then
        echo "❌ Error: openssh-server failed to install correctly"
        # Try to reinstall
        apt-get install --reinstall -y openssh-server || {
            echo "❌ Error: Failed to reinstall openssh-server";
            return 1;
        }
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
        systemctl daemon-reload
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
    ssh-keygen -A || { 
        echo "❌ Error: Failed to generate SSH host keys"; 
        # Try to reinstall and regenerate
        apt-get install --reinstall -y openssh-server
        ssh-keygen -A || return 1;
    }
    echo "✅ Success: SSH host keys generated"
}
backup_configs() {
    timestamp=$(date +%Y%m%d_%H%M%S)
    echo "💾 Backing up configs with timestamp $timestamp..."
    
    if [ -f /etc/ssh/sshd_config ]; then
        echo "Backing up /etc/ssh/sshd_config..."
        cp -v /etc/ssh/sshd_config "/etc/ssh/sshd_config.bak.$timestamp" || { 
            echo "❌ Error: Failed to backup sshd_config"; 
        }
    fi
    
    if [ -f /etc/samba/smb.conf ]; then
        echo "Backing up /etc/samba/smb.conf..."
        cp -v /etc/samba/smb.conf "/etc/samba/smb.conf.bak.$timestamp" || { 
            echo "❌ Error: Failed to backup smb.conf"; 
        }
    fi
    
    if [ -f "/home/${REAL_USER}/.bashrc" ]; then
        echo "Backing up ~${REAL_USER}/.bashrc..."
        sudo -u "${REAL_USER}" cp -v "/home/${REAL_USER}/.bashrc" "/home/${REAL_USER}/.bashrc.backup.$timestamp" || { 
            echo "❌ Error: Failed to backup .bashrc"; 
        }
    fi
    
    echo "✅ Success: Configs backed up"
}
configure_ssh() {
    echo "🔒 Configuring SSH..."
    echo "Writing SSH config to /etc/ssh/sshd_config..."
    cat > /etc/ssh/sshd_config << 'EOF' || { echo "❌ Error: Failed to write sshd_config"; return 1; }
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
    if sshd -t; then
        echo "✅ Success: SSH config valid"
    else
        echo "❌ Error: SSH config invalid"
        return 1
    fi
    
    # Check if SSH daemon is running
    if ! pgrep -f sshd > /dev/null; then
        echo "⚠️ Warning: SSH daemon not running"
        SSH_SERVICE=$(detect_ssh_service)
        if [ "$SSH_SERVICE" != "UNKNOWN" ]; then
            echo "Restarting SSH service ($SSH_SERVICE)..."
            systemctl restart "$SSH_SERVICE" || {
                echo "❌ Error: Failed to restart SSH";
                return 1;
            }
        else
            echo "❌ Error: Could not determine SSH service to restart"
            return 1
        fi
    fi
    
    echo "✅ Success: SSH configured"
}
configure_samba() {
    echo "📂 Setting up Samba..."
    echo "Writing Samba config to /etc/samba/smb.conf..."
    cat > /etc/samba/smb.conf << 'EOF' || { echo "❌ Error: Failed to write smb.conf"; return 1; }
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
    mkdir -pv /samba/public || { 
        echo "❌ Error: Failed to create /samba/public"; 
        return 1;
    }
    echo "Setting permissions on /samba/public..."
    chmod -v 777 /samba/public || { 
        echo "❌ Error: Failed to set permissions on /samba/public"; 
        return 1;
    }
    
    RAID_PATH="/media/heathen-admin/llmRAID"
    if [ ! -d "$RAID_PATH" ]; then
        echo "Creating $RAID_PATH..."
        mkdir -pv "$RAID_PATH" || { 
            echo "❌ Error: Failed to create $RAID_PATH"; 
            return 1;
        }
        echo "Setting ownership on $RAID_PATH..."
        chown -v heathen-admin:heathen-admin "$RAID_PATH" 2>/dev/null || { 
            echo "⚠️ Warning: Failed to set ownership on $RAID_PATH - username might not exist"; 
        }
        echo "Setting permissions on $RAID_PATH..."
        chmod -v 775 "$RAID_PATH" || { 
            echo "❌ Error: Failed to set permissions on $RAID_PATH"; 
            return 1;
        }
    else
        echo "✅ $RAID_PATH already exists—skipping creation"
    fi
    
    echo "✅ Success: Samba directories set up"
}
configure_shell() {
    echo "🔧 Configuring shell aliases and environment..."
    BASHRC_FILE="/home/${REAL_USER}/.bashrc"
    
    # Check if Conda aliases already exist
    if ! grep -q "# Conda Aliases and Auto-Activation" "$BASHRC_FILE"; then
        echo "Adding Conda aliases to .bashrc..."
        cat >> "$BASHRC_FILE" << 'EOF'
# Conda Aliases and Auto-Activation
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate BASE
fi
alias rocm="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate ROCM"
alias nanorocm="nano $HOME/miniconda3/envs/ROCM/etc/conda/activate.d/env_vars.sh"
alias watchrocm="watch -n 1 rocm-smi"
alias cuda="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate CUDA"
alias nanocuda="nano $HOME/miniconda3/envs/CUDA/etc/conda/activate.d/env_vars.sh"
alias watchcuda="watch -n 1 nvidia-smi"
alias nanobase="nano $HOME/miniconda3/envs/BASE/etc/conda/activate.d/env_vars.sh"
alias base="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate BASE"
EOF
    fi

    # Check if Docker alias exists
    if ! grep -q "alias drun=" "$BASHRC_FILE"; then
        echo "Adding Docker alias to .bashrc..."
        cat >> "$BASHRC_FILE" << 'EOF'
alias drun='docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G -v $HOME/dockerx:/dockerx -w /dockerx'
EOF
    fi

    # Check if Ollama aliases exist
    if ! grep -q "ollamass=" "$BASHRC_FILE"; then
        echo "Adding Ollama aliases to .bashrc..."
        cat >> "$BASHRC_FILE" << 'EOF'
alias ollamass="sudo nano /etc/systemd/system/ollama.service"
alias ollamacfg="nano ~/.ollama/config.json"
alias startollama="sudo systemctl start ollama"
alias stopollama="sudo systemctl stop ollama"
alias restartollama="sudo systemctl restart ollama"
alias ollamastatus="sudo systemctl status ollama-*"
alias ollamaports="sudo netstat -tulpn | grep ollama"
EOF
    fi

    # Check if system management aliases exist
    if ! grep -q "alias nanobash=" "$BASHRC_FILE"; then
        echo "Adding system management aliases to .bashrc..."
        cat >> "$BASHRC_FILE" << 'EOF'
alias nanobash="sudo nano ~/.bashrc"
alias nanoenv="sudo nano /etc/environment"
EOF
    fi

    # Check if Modern CLI aliases exist
    if ! grep -q "# Modern CLI Aliases" "$BASHRC_FILE"; then
        echo "Adding modern CLI aliases to .bashrc..."
        cat >> "$BASHRC_FILE" << 'EOF'
# Modern CLI Aliases
alias ls="eza --icons --group-directories-first"
alias ll="eza -l --icons --group-directories-first"
alias la="eza -la --icons --group-directories-first"
alias cat="bat --style=full --paging=never"
alias find="fd"
alias grep="rg"
# System Monitoring Aliases
alias htop="htop -t"
alias diskspace="df -h"
alias memusage="free -h"
alias sysinfo="neofetch"
alias monitor="glances"
alias netwatch="netstat -tulpn"
alias gputop="nvtop"
alias amdtop="amdgpu_top"
# Google Drive Alias
alias gdrive="rclone --vfs-cache-mode writes mount googledrive: ~/GoogleDrive"
EOF
    fi

    # Check if Claude Code aliases exist
    if ! grep -q "# Claude Code Aliases" "$BASHRC_FILE"; then
        echo "Adding Claude Code aliases to .bashrc..."
        cat >> "$BASHRC_FILE" << 'EOF'
# Claude Code Aliases
alias cc="claude-code"
alias claudec="claude-code"
EOF
    fi

    # Add ROCm and CUDA paths
    if ! grep -q "# ROCm Paths" "$BASHRC_FILE"; then
        echo "Adding ROCm and CUDA paths to .bashrc..."
        cat >> "$BASHRC_FILE" << 'EOF'
# ROCm Paths
export ROCM_PATH=/opt/rocm-6.3.4
export HIP_PATH=/opt/rocm-6.3.4/hip

# CUDA Paths
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda

# Combined PATH
export PATH=$HOME/bin:/usr/local/cuda/bin:/usr/local/go/bin:$ROCM_PATH/bin:$ROCM_PATH/hip/bin:$PATH

# Combined LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$LD_LIBRARY_PATH
EOF
    fi

    # Add home folder recursively to PATH
    if ! grep -q "find \$HOME -type d | tr '\\n' ':'" "$BASHRC_FILE"; then
        echo "Adding home folder recursively to PATH in .bashrc..."
        cat >> "$BASHRC_FILE" << 'EOF'
# Add home folder recursively to PATH
export PATH="$(find $HOME -type d | tr '\n' ':')$PATH"
EOF
    fi

    # Add general environment variables
    if ! grep -q "DEBIAN_FRONTEND=noninteractive" "$BASHRC_FILE"; then
        echo "Adding general environment variables to .bashrc..."
        cat >> "$BASHRC_FILE" << 'EOF'
# General Environment Variables
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PYTHONENCODING=UTF-8
export PIP_ROOT_USER_ACTION=ignore
EOF
    fi

    # Add $HOME/bin to PATH if not already there
    if ! grep -q "PATH=\"\$HOME/bin:\$PATH\"" "$BASHRC_FILE"; then
        echo "Adding bin directory to PATH in .bashrc..."
        cat >> "$BASHRC_FILE" << 'EOF'
# Add $HOME/bin to PATH
export PATH="$HOME/bin:$PATH"
EOF
    fi

    # Fix ownership
    chown "${REAL_USER}:${REAL_USER}" "$BASHRC_FILE"
    echo "✅ Success: Shell aliases and environment variables configured"
}
setup_monitoring() {
    echo "📊 Setting up monitoring tools..."
    monitoring_tools=(neofetch glances)
    install_packages "${monitoring_tools[@]}"
    
    # Create neofetch config directory
    echo "Creating neofetch config directory..."
    NEOFETCH_DIR="/home/${REAL_USER}/.config/neofetch"
    sudo -u "${REAL_USER}" mkdir -pv "$NEOFETCH_DIR" || { 
        echo "❌ Error: Failed to create neofetch config dir"; 
        return 1;
    }
    
    # Customize neofetch configuration
    echo "Writing neofetch config..."
    sudo -u "${REAL_USER}" bash -c "cat > \"$NEOFETCH_DIR/config.conf\"" << 'EOF'
print_info() {
    info title
    info underline
    info "OS" distro
    info "Host" model
    info "Kernel" kernel
    info "Uptime" uptime
    info "Packages" packages
    info "Shell" shell
    info "CPU" cpu
    info "GPU" gpu
    info "Memory" memory
    info "Disk" disk
    info "Local IP" local_ip
}
EOF
    echo "✅ Success: Monitoring tools set up"
}
setup_security() {
    echo "🔒 Setting up security tools..."
    security_tools=(unattended-upgrades apt-listchanges)
    install_packages "${security_tools[@]}"
    
    echo "Configuring unattended upgrades..."
    cat > /etc/apt/apt.conf.d/20auto-upgrades << EOF || { echo "❌ Error: Failed to configure unattended upgrades"; return 1; }
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Download-Upgradeable-Packages "1";
APT::Periodic::AutocleanInterval "7";
APT::Periodic::Unattended-Upgrade "1";
EOF
    echo "✅ Success: Security tools configured"
}
configure_firewall() {
    echo "🧱 Configuring firewall..."
    install_packages ufw
    
    echo "Setting default deny incoming..."
    ufw default deny incoming || { 
        echo "❌ Error: Failed to set default deny incoming"; 
        return 1;
    }
    
    echo "Setting default allow outgoing..."
    ufw default allow outgoing || { 
        echo "❌ Error: Failed to set default allow outgoing"; 
        return 1;
    }
    
    echo "Allowing SSH..."
    ufw allow ssh || { 
        echo "❌ Error: Failed to allow SSH"; 
        return 1;
    }
    
    echo "Allowing Samba..."
    ufw allow samba || { 
        echo "❌ Error: Failed to allow Samba"; 
        return 1;
    }
    
    echo "Allowing port 11434/tcp (Ollama)..."
    ufw allow 11434/tcp || { 
        echo "❌ Error: Failed to allow port 11434"; 
        return 1;
    }
    
    echo "Allowing port 3000/tcp (OpenWebUI)..."
    ufw allow 3000/tcp || { 
        echo "❌ Error: Failed to allow port 3000"; 
        return 1;
    }
    
    echo "Allowing port 8080/tcp (OpenWebUI alt)..."
    ufw allow 8080/tcp || { 
        echo "❌ Error: Failed to allow port 8080"; 
        return 1;
    }
    
    # Verify SSH running before enabling
    echo "Verifying SSH is working before enabling firewall..."
    SSH_SERVICE=$(detect_ssh_service)
    if [ "$SSH_SERVICE" = "UNKNOWN" ]; then
        echo "⚠️ Warning: SSH service not detected, not enabling firewall"
        return 1
    fi
    
    if ! systemctl is-active "$SSH_SERVICE" >/dev/null; then
        echo "⚠️ Warning: SSH service not active, attempting to start..."
        systemctl start "$SSH_SERVICE" || {
            echo "❌ Error: Failed to start SSH service, not enabling firewall"
            return 1
        }
    fi
    
    echo "SSH service verified as running. Enabling UFW..."
    echo "y" | ufw enable || { 
        echo "❌ Error: Failed to enable UFW"; 
        return 1;
    }
    
    echo "✅ Success: Firewall configured"
}
start_services() {
    echo "🌐 Starting services..."
    # Determine SSH service
    SSH_SERVICE=$(detect_ssh_service)
    if [ "$SSH_SERVICE" = "UNKNOWN" ]; then
        echo "❌ Error: Cannot determine SSH service name"
        echo "Attempting to reinstall SSH server..."
        apt-get install --reinstall openssh-server
        SSH_SERVICE=$(detect_ssh_service)
    fi
    
    # Start and enable SSH
    if [ "$SSH_SERVICE" != "UNKNOWN" ]; then
        echo "Processing SSH service: $SSH_SERVICE..."
        echo "Stopping $SSH_SERVICE if running..."
        systemctl stop "$SSH_SERVICE" 2>/dev/null || echo "⚠️ $SSH_SERVICE wasn't running"
        echo "Starting $SSH_SERVICE..."
        systemctl start "$SSH_SERVICE" || { 
            echo "❌ Error: Failed to start $SSH_SERVICE"; 
            systemctl status "$SSH_SERVICE" --no-pager; 
            return 1;
        }
        echo "Enabling $SSH_SERVICE..."
        systemctl enable "$SSH_SERVICE" || { 
            echo "❌ Error: Failed to enable $SSH_SERVICE"; 
            return 1;
        }
        echo "✅ Success: $SSH_SERVICE started and enabled"
    else
        echo "❌ Error: SSH service still not found after reinstall attempt"
        return 1
    fi
    
    # Start and enable Samba
    for svc in smbd nmbd; do
        echo "Processing service: $svc..."
        echo "Stopping $svc if running..."
        systemctl stop "$svc" 2>/dev/null || echo "⚠️ $svc wasn't running"
        echo "Starting $svc..."
        systemctl start "$svc" || { 
            echo "❌ Error: Failed to start $svc"; 
            systemctl status "$svc" --no-pager; 
            return 1;
        }
        echo "Enabling $svc..."
        systemctl enable "$svc" || { 
            echo "❌ Error: Failed to enable $svc"; 
            return 1;
        }
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
    apt-get update || { 
        echo "❌ Error: Final apt update failed"; 
        # Continue despite failure
    }
    
    echo "Upgrading installed packages..."
    apt-get upgrade -y || { 
        echo "❌ Error: Final apt upgrade failed"; 
        # Continue despite failure
    }
    
    echo "Running autoremove..."
    apt-get autoremove -y || { 
        echo "❌ Error: Final autoremove failed"; 
        # Continue despite failure
    }
    
    echo "Cleaning apt cache..."
    apt-get autoclean || { 
        echo "❌ Error: Final autoclean failed"; 
        # Continue despite failure
    }
    
    echo "✅ Success: Final cleanup complete"
}
install_gpu_tools() {
    echo "🎮 Installing GPU monitoring tools..."
    gpu_tools=(nvtop radeontop rovclock)
    install_packages "${gpu_tools[@]}"
    
    # Check for nvtop from Snap if apt install failed
    if ! command -v nvtop >/dev/null 2>&1; then
        echo "Installing nvtop via snap..."
        snap install nvtop || {
            echo "⚠️ Warning: Failed to install nvtop via snap, continuing..."
        }
    fi
    
    # Check for amdgpu_top
    if ! command -v amdgpu_top >/dev/null 2>&1; then
        echo "Installing amdgpu_top..."
        install_amdgpu_top
    fi
    
    echo "✅ Success: GPU monitoring tools installed"
}
install_development_libs() {
    echo "🔧 Installing development libraries..."
    dev_libs=(
        libstdc++-12-dev
        libtcmalloc-minimal4
        libopenmpi3
        libdnnl-dev
        ninja-build
        libopenblas-dev
        libpng-dev
        libjpeg-dev
    )
    install_packages "${dev_libs[@]}"
    
    # Fix tcmalloc library paths
    if [ -f "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" ]; then
        echo "Setting up tcmalloc symlinks..."
        ln -sf /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 /usr/lib/libtcmalloc_minimal.so.4 2>/dev/null || true
        ln -sf /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 /usr/lib/libtcmalloc_minimal.so 2>/dev/null || true
        ldconfig
    fi
    
    echo "✅ Success: Development libraries installed"
}
install_dotnet_sdk() {
    echo "📦 Installing .NET SDK packages..."
    
    # First check if we need to add the Microsoft repository
    if [ ! -f /etc/apt/sources.list.d/microsoft-prod.list ]; then
        echo "Adding Microsoft package repository..."
        wget -q https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
        dpkg -i packages-microsoft-prod.deb
        rm -f packages-microsoft-prod.deb
        apt-get update
    fi
    
    # Install .NET SDK packages
    dotnet_packages=(
        dotnet-sdk-6.0
        dotnet-sdk-9.0
    )
    
    install_packages "${dotnet_packages[@]}"
    
    # Verify installation
    if command -v dotnet >/dev/null 2>&1; then
        echo "✅ .NET SDK installed successfully:"
        dotnet --list-sdks
    else
        echo "❌ Error: .NET SDK not found after installation"
    fi
    
    echo "✅ Success: .NET SDK packages installed"
}
main() {
    echo "🔧 Entering main function..."
    check_root
    fix_interruptions
    update_system
    install_essentials
    install_development_libs
    install_dotnet_sdk
    install_go
    install_monitoring_tools
    install_claude_code_cli
    install_modern_tools
    create_symlinks
    install_whatsie
    install_gpu_tools
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
    
    # Get system information
    CPU_INFO=$(grep "model name" /proc/cpuinfo | head -n 1 | sed 's/model name\s*: //g')
    MEM_INFO=$(free -h | grep Mem | awk '{print $2}')
    DISK_INFO=$(df -h / | grep / | awk '{print $2}')
    
    echo "
✨ Ubuntu Setup Complete! ✨
System Information:
- CPU: $CPU_INFO
- Memory: $MEM_INFO
- Disk: $DISK_INFO
Installed Components:
- Essentials: Git, Git LFS, GCC 12/11, development libraries, build tools
- Programming: .NET SDK 6.0 & 9.0, Node.js (npm), Rust (rustup)
- AI Tools: Claude Code CLI
- GPU Tools: nvtop, radeontop, amdgpu_top, rovclock
- ML Libraries: OpenBLAS, DNNL, OpenMPI, tcmalloc
- Network: SSH, Samba, Firewall (UFW)
- Utilities: Whatsie, Google Drive (rclone), monitoring tools
Next Steps:
- Run 'source ~/.bashrc' to activate new aliases and paths
- For Google Drive setup, run 'rclone config'
- Check 'amdtop', 'gputop', or 'sysinfo' commands
- Use 'claude-code' or 'cc' command to access Claude Code CLI
Log file: $LOG_FILE
"
    # Display network information
    echo "Network Information:"
    ip addr show | grep "inet " || echo "⚠️ Warning: Failed to display network info"
    
    # Display SSH status
    echo "
📡 SSH Status:"
    SSH_SERVICE=$(detect_ssh_service)
    if [ "$SSH_SERVICE" != "UNKNOWN" ]; then
        systemctl status "$SSH_SERVICE" --no-pager || echo "❌ SSH service not found"
    else
        echo "❌ SSH service not found"
    fi
    
    echo "SSH Port:"
    netstat -tuln | grep ":22 " || echo "❌ No service listening on port 22"
    
    echo "✅ Setup complete! System ready for use."
}
main