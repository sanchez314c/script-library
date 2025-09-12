#!/bin/bash
###############################################################
#    _   ___     _____ ____ ___    _      _  _   ____   ___   #
#   | \ | \ \   / /_ _|  _ \_ _|  / \    | || | |  _ \ / _ \  #
#   |  \| |\ \ / / | || | | | |  / _ \   | || |_| | | | | | | #
#   | |\  | \ V /  | || |_| | | / ___ \  |__   _| |_| | |_| | #
#   |_| \_|  \_/  |___|____/___/_/   \_\    |_| |____/ \___/  #
#                                                             #
###############################################################
#
# NVIDIA Driver 470 Installation Script for K80 GPUs
# Version: 2.0.0
# Date: April 15, 2025
# Description: Installs NVIDIA driver 470 with best practices for K80 GPUs

# Enable strict error handling
set -e

# Get the real user even when running with sudo
REAL_USER="${SUDO_USER:-$USER}"

# Create log file on desktop
LOG_FILE="/home/${REAL_USER}/Desktop/nvidia_driver_install_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"
chown $REAL_USER:$REAL_USER "$LOG_FILE"
chmod 644 "$LOG_FILE"

# Redirect output to log file and stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "🚀 Starting NVIDIA Driver 470 Installation..."

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "❌ Error: Requires root privileges. Run with sudo."
        exit 1
    fi
    echo "✅ Running as root"
}

check_gpu() {
    echo "🔍 Checking for NVIDIA GPU hardware..."
    if ! lspci | grep -i nvidia > /dev/null; then
        echo "❌ Error: No NVIDIA GPU hardware detected. Installation cannot proceed."
        exit 1
    fi
    
    # Initialize GPU flags
    IS_K80=false
    IS_RTX=false
    
    # Detect different GPU types
    echo "📋 Detected NVIDIA GPUs:"
    lspci | grep -i nvidia | while read -r line; do
        echo "  - $line"
        if [[ "$line" == *"K80"* || "$line" == *"Tesla K80"* ]]; then
            IS_K80=true
        fi
        if [[ "$line" == *"3090"* || "$line" == *"RTX 3090"* ]]; then
            IS_RTX=true
        fi
    done
    
    # Confirm specific GPU types
    if [ "$IS_K80" = true ]; then
        echo "✅ NVIDIA K80 GPU detected - will configure compute optimizations"
    fi
    
    if [ "$IS_RTX" = true ]; then
        echo "✅ NVIDIA RTX 3090 Ti GPU detected - will configure gaming/graphics optimizations"
    fi
    
    echo "🔧 Configuring driver for multi-GPU setup (K80 + RTX 3090 Ti)"
}

# check_blacklisted_modules() {
#     echo "🔍 Checking for blacklisted NVIDIA modules..."
#     if grep -q "blacklist nvidia" /etc/modprobe.d/* 2>/dev/null; then
#         echo "⚠️ Warning: NVIDIA modules are blacklisted. This may prevent driver loading."
#         echo "Would you like to remove these blacklist entries? (y/N)"
#         read -r REMOVE_BLACKLIST
#         if [[ "$REMOVE_BLACKLIST" =~ ^[Yy]$ ]]; then
#             echo "Removing blacklist entries..."
#             sed -i '/blacklist nvidia/d' /etc/modprobe.d/*
#             echo "✅ Blacklist entries removed"
#         else
#             echo "⚠️ Continuing with blacklisted modules. Driver may not load properly."
#         fi
#     else
#         echo "✅ No blacklisted NVIDIA modules found"
#     fi
# }

backup_existing_config() {
    echo "📦 Backing up existing X11 and NVIDIA configurations..."
    
    # Backup X config
    if [ -f /etc/X11/xorg.conf ]; then
        cp /etc/X11/xorg.conf /etc/X11/xorg.conf.backup.$(date +%Y%m%d_%H%M%S)
        echo "✅ Backed up /etc/X11/xorg.conf"
    fi
    
    # Backup NVIDIA settings
    if [ -f /etc/nvidia/nvidia-settings.rc ]; then
        mkdir -p /etc/nvidia/backup.$(date +%Y%m%d_%H%M%S)
        cp /etc/nvidia/nvidia-settings.rc /etc/nvidia/backup.$(date +%Y%m%d_%H%M%S)/
        echo "✅ Backed up NVIDIA settings"
    fi
    
    # Backup X11 configs in user's home
    if [ -f "/home/${REAL_USER}/.nvidia-settings-rc" ]; then
        cp "/home/${REAL_USER}/.nvidia-settings-rc" "/home/${REAL_USER}/.nvidia-settings-rc.backup.$(date +%Y%m%d_%H%M%S)"
        echo "✅ Backed up user's NVIDIA settings"
    fi
    
    echo "✅ Configuration backup complete"
}

remove_existing_nvidia() {
    echo "🧹 Checking for existing NVIDIA installations..."
    
    # Check if NVIDIA 470 is already installed
    if dpkg -l | grep -q "nvidia-driver-470"; then
        echo "ℹ️ NVIDIA 470 already installed."
        echo "Would you like to reinstall it? (y/N)"
        read -r REINSTALL
        if [[ ! "$REINSTALL" =~ ^[Yy]$ ]]; then
            echo "✅ Keeping existing NVIDIA 470 installation—skipping removal"
            return 0
        fi
        echo "🧹 Proceeding with removal for reinstallation..."
    fi
    
    echo "🧹 Removing existing NVIDIA drivers..."
    
    # Stop services using NVIDIA
    systemctl stop nvidia-persistenced 2>/dev/null || true
    
    # Remove all NVIDIA packages
    apt-get remove --purge -y '^nvidia-.*' 2>/dev/null || true
    apt-get remove --purge -y '^libnvidia-.*' 2>/dev/null || true
    apt-get remove --purge -y 'cuda-.*' 2>/dev/null || true
    
    # Check for and run NVIDIA's own uninstaller
    if command -v nvidia-uninstall &>/dev/null; then
        echo "📥 Removing NVIDIA .run driver installation..."
        nvidia-uninstall --silent || true
    fi
    
    # Clean up the system
    apt-get autoremove -y
    apt-get clean
    
    # Remove NVIDIA module from kernel if loaded
    if lsmod | grep -q nvidia; then
        echo "🧹 Unloading NVIDIA kernel modules..."
        rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia || true
    fi
    
    # Remove NVIDIA files
    rm -rf /etc/nvidia /etc/X11/xorg.conf 2>/dev/null || true
    
    echo "✅ Existing NVIDIA installations removed"
}

install_dependencies() {
    echo "📦 Installing driver-specific dependencies..."
    
    apt-get update || { 
        echo "⚠️ Warning: apt-get update failed. Using --fix-missing option..."
        apt-get update --fix-missing || {
            echo "❌ Error: apt-get update failed. Check your network and apt sources."
            exit 1
        }
    }
    
    # Install essential packages for building drivers
    DEPS=(
        dkms
        build-essential
        libglvnd-dev
        pkg-config
        "linux-headers-$(uname -r)"
    )
    
    for package in "${DEPS[@]}"; do
        echo "📦 Installing $package..."
        apt-get install -y "$package" || {
            echo "⚠️ Warning: Failed to install $package. Trying with --fix-broken..."
            apt-get install -f -y
            apt-get install -y "$package" || {
                echo "❌ Error: Failed to install $package. Installation may not complete successfully."
                # Continue despite failure
            }
        }
    done
    
    echo "✅ Dependencies installed"
}

add_nvidia_repo() {
    echo "🌐 Adding NVIDIA driver repository..."
    
    # Check if graphics-drivers PPA already added
    if ! grep -r "ppa:graphics-drivers/ppa" /etc/apt/sources.list /etc/apt/sources.list.d/ &>/dev/null; then
        # Install dependencies
        apt-get install -y software-properties-common || {
            echo "❌ Error: Failed to install software-properties-common"
            exit 1
        }
        
        # Add the PPA
        add-apt-repository -y ppa:graphics-drivers/ppa || { 
            echo "❌ Error: PPA add failed—check network"
            exit 1
        }
    else
        echo "ℹ️ NVIDIA repository already added"
    fi
    
    apt-get update || { 
        echo "⚠️ Warning: apt-get update failed. Using --fix-missing option..."
        apt-get update --fix-missing || {
            echo "❌ Error: apt-get update failed after adding repository"
            exit 1
        }
    }
    
    echo "✅ NVIDIA PPA added"
}

# disable_nouveau() {
#     echo "🔧 Checking for and disabling Nouveau driver..."
    
#     # Check if Nouveau is loaded
#     if lsmod | grep -q nouveau; then
#         echo "ℹ️ Nouveau driver currently loaded. Creating blacklist file..."
        
#         # Create blacklist file
#         cat > /etc/modprobe.d/blacklist-nouveau.conf << EOF
# blacklist nouveau
# options nouveau modeset=0
# EOF
        
#         # Update initramfs
#         echo "🔄 Updating initramfs to disable Nouveau..."
#         update-initramfs -u || {
#             echo "⚠️ Warning: Failed to update initramfs. Nouveau might still be loaded after reboot."
#         }
        
#         echo "⚠️ Nouveau driver has been blacklisted, but is still running."
#         echo "A system reboot is required before installing NVIDIA drivers."
#         echo "Would you like to reboot now? (y/N)"
#         read -r REBOOT_NOW
#         if [[ "$REBOOT_NOW" =~ ^[Yy]$ ]]; then
#             echo "🔄 Rebooting system in 5 seconds..."
#             sleep 5
#             reboot
#             exit 0
#         else
#             echo "⚠️ Please reboot your system before continuing with the installation."
#             exit 0
#         fi
#     else
#         echo "✅ Nouveau driver not loaded"
#     fi
# }

install_nvidia_driver() {
    echo "📥 Installing NVIDIA driver 470..."
    
    # Install driver packages
    DRIVER_PACKAGES=(
        "nvidia-driver-470"
        "nvidia-dkms-470"
        "nvidia-utils-470"
        "nvidia-settings"
    )
    
    for package in "${DRIVER_PACKAGES[@]}"; do
        echo "📦 Installing $package..."
        apt-get install -y "$package" || {
            echo "⚠️ Warning: Failed to install $package. Trying with --fix-broken..."
            apt-get install -f -y
            apt-get install -y "$package" || {
                echo "❌ Error: Failed to install $package. Installation may not complete successfully."
                # Continue despite failure
            }
        }
    done
    
    # Install additional packages for CUDA/compute
    if [ "$IS_K80" = true ]; then
        echo "📦 Installing additional packages for K80 compute capabilities..."
        COMPUTE_PACKAGES=(
            "nvidia-compute-utils-470"
            "libnvidia-compute-470"
        )
        
        for package in "${COMPUTE_PACKAGES[@]}"; do
            echo "📦 Installing $package..."
            apt-get install -y "$package" || {
                echo "⚠️ Warning: Failed to install $package. Trying with --fix-broken..."
                apt-get install -f -y
                apt-get install -y "$package" || {
                    echo "❌ Error: Failed to install $package. Installation may not complete successfully."
                    # Continue despite failure
                }
            }
        done
    fi
    
    echo "✅ NVIDIA driver 470 installed"
}

configure_nvidia() {
    echo "🔧 Configuring NVIDIA driver for multi-GPU setup..."
    
    # Create nvidia-xconfig if missing
    if ! command -v nvidia-xconfig &> /dev/null; then
        echo "⚠️ Warning: nvidia-xconfig not found, skipping X server configuration"
    else
        echo "🔧 Configuring X server..."
        nvidia-xconfig --no-logo || {
            echo "⚠️ Warning: Failed to configure X server, continuing..."
        }
    fi
    
    # Enable persistence mode
    if command -v nvidia-persistenced &> /dev/null; then
        echo "🔧 Enabling NVIDIA persistence mode..."
        systemctl enable nvidia-persistenced || true
        systemctl start nvidia-persistenced || {
            echo "⚠️ Warning: Failed to start NVIDIA persistence daemon, continuing..."
        }
    fi
    
    # Create configuration directory
    mkdir -p /etc/nvidia
    
    # Create multi-GPU configuration
    echo "🔧 Creating optimizations for multi-GPU setup (K80 + RTX 3090 Ti)..."
    cat > /etc/nvidia/nvidia-multi-gpu.conf << 'EOF'
# NVIDIA Multi-GPU Configuration (K80 + RTX 3090 Ti)

# Enable CUDA operations
options nvidia NVreg_EnableStreamMemOPs=1

# Performance settings for all GPUs
options nvidia NVreg_RegistryDwords="PerfLevelSrc=0x2222; PowerMizerEnable=0x1"

# K80-specific optimizations (compute-focused)
# RTX 3090 Ti optimizations (mixed compute/graphics)
EOF
    
    # Apply the configuration
    cp /etc/nvidia/nvidia-multi-gpu.conf /etc/modprobe.d/
    
    # Create script to configure GPU specific settings
    cat > /usr/local/bin/nvidia-configure-gpus.sh << 'EOF'
#!/bin/bash
# Script to apply specific settings for K80 and RTX 3090 Ti GPUs

# Enable persistence mode on all GPUs
nvidia-smi -pm 1

# Get list of GPUs
GPUS=$(nvidia-smi --query-gpu=index,name --format=csv,noheader)

# Process each GPU
echo "$GPUS" | while IFS=, read -r IDX NAME; do
    # Trim whitespace
    IDX=$(echo "$IDX" | xargs)
    NAME=$(echo "$NAME" | xargs)
    
    echo "Configuring GPU $IDX: $NAME"
    
    # Apply settings based on GPU type
    if [[ "$NAME" == *"K80"* || "$NAME" == *"Tesla K80"* ]]; then
        echo "  Applying K80 compute optimizations..."
        # Set to maximum performance mode
        nvidia-smi -i "$IDX" -ac 2505,875
        # Disable auto boost
        nvidia-smi -i "$IDX" --auto-boost-default=0
        # Set to maximum performance state
        nvidia-smi -i "$IDX" -pl 150
    elif [[ "$NAME" == *"3090"* || "$NAME" == *"RTX 3090"* ]]; then
        echo "  Applying RTX 3090 Ti optimizations..."
        # Set to preferred performance mode for mixed workloads
        nvidia-smi -i "$IDX" -ac 1395,1695
        # Enable auto boost
        nvidia-smi -i "$IDX" --auto-boost-default=1
        # Use a slightly conservative power limit to maintain stability
        nvidia-smi -i "$IDX" -pl 350
    else
        echo "  Using default settings for unknown GPU type"
    fi
done

# Apply general multi-GPU settings
echo "Applying multi-GPU settings..."
# Disable GPU compute mode exclusivity for better sharing
nvidia-smi --compute-mode=0

echo "GPU configuration complete."
EOF
    chmod +x /usr/local/bin/nvidia-configure-gpus.sh
    
    # Run the configuration script
    /usr/local/bin/nvidia-configure-gpus.sh || {
        echo "⚠️ Warning: Failed to apply GPU-specific settings, will try again after verification"
    }
    
    # Create systemd service to apply settings on boot
    cat > /etc/systemd/system/nvidia-config.service << 'EOF'
[Unit]
Description=NVIDIA GPU Configuration
After=nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/nvidia-configure-gpus.sh
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
EOF
    
    # Enable the service
    systemctl enable nvidia-config.service
    
    echo "✅ NVIDIA driver multi-GPU configuration complete"
}

verify_installation() {
    echo "🔍 Verifying NVIDIA driver installation..."
    
    # Check for NVIDIA kernel module
    echo "🔄 Loading NVIDIA kernel module..."
    modprobe nvidia || {
        echo "⚠️ Warning: Failed to load NVIDIA kernel module. This might be resolved after a reboot."
    }
    
    # Check driver status with nvidia-smi
    echo "🔍 Running nvidia-smi to verify driver..."
    if nvidia-smi > /dev/null; then
        echo "✅ NVIDIA driver verified successfully!"
        echo "Driver details:"
        nvidia-smi
    else
        echo "❌ Error: nvidia-smi failed. Driver may not be loaded properly."
        echo "This issue can often be resolved by rebooting the system."
        echo "Would you like to reboot now? (y/N)"
        read -r REBOOT_NOW
        if [[ "$REBOOT_NOW" =~ ^[Yy]$ ]]; then
            echo "🔄 Rebooting system in 5 seconds..."
            sleep 5
            reboot
            exit 0
        else
            echo "⚠️ Please reboot your system to complete the installation."
        fi
    fi
}

create_gpu_utils() {
    echo "🛠️ Creating useful NVIDIA GPU utilities..."
    
    # Create a directory for scripts if it doesn't exist
    SCRIPT_DIR="/usr/local/bin"
    mkdir -p "$SCRIPT_DIR"
    
    # Create GPU monitoring script
    cat > "$SCRIPT_DIR/nvidia-monitor.sh" << 'EOF'
#!/bin/bash
# NVIDIA GPU Monitoring Script
# Shows GPU status continuously with refresh

while true; do
    clear
    echo "NVIDIA GPU Status Monitor"
    echo "-------------------------"
    echo "Press Ctrl+C to exit"
    echo ""
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw,clocks.current.graphics --format=csv,noheader
    sleep 3
done
EOF
    chmod +x "$SCRIPT_DIR/nvidia-monitor.sh"
    
    # Create GPU info script
    cat > "$SCRIPT_DIR/nvidia-info.sh" << 'EOF'
#!/bin/bash
# NVIDIA GPU Information Script
# Shows detailed GPU information

echo "NVIDIA GPU Detailed Information"
echo "-----------------------------"

# Basic information
echo "GPU Device Information:"
nvidia-smi -L

echo -e "\nGPU Driver Information:"
nvidia-smi | grep "Driver Version"

echo -e "\nCUDA Version:"
nvidia-smi | grep "CUDA Version"

echo -e "\nDetailed GPU Information:"
nvidia-smi -q

echo -e "\nGPU Topology:"
nvidia-smi topo -m

echo -e "\nGPU Performance State:"
nvidia-smi --query-gpu=name,pstate,clocks.gr,clocks.mem --format=csv

echo -e "\nGPU Power State:"
nvidia-smi --query-gpu=name,power.draw,power.limit --format=csv
EOF
    chmod +x "$SCRIPT_DIR/nvidia-info.sh"
    
    # Create GPU reset script
    cat > "$SCRIPT_DIR/nvidia-reset.sh" << 'EOF'
#!/bin/bash
# NVIDIA GPU Reset Script
# Use with caution - requires root privileges

if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run with sudo" >&2
    exit 1
fi

echo "Attempting to reset NVIDIA GPU modules..."

# Unload NVIDIA modules in the correct order
echo "Unloading NVIDIA kernel modules..."
rmmod nvidia_drm 2>/dev/null || true
rmmod nvidia_modeset 2>/dev/null || true
rmmod nvidia_uvm 2>/dev/null || true
rmmod nvidia 2>/dev/null || true

# Wait a moment
sleep 2

# Reload NVIDIA modules in the correct order
echo "Reloading NVIDIA kernel modules..."
modprobe nvidia
modprobe nvidia_uvm
modprobe nvidia_modeset
modprobe nvidia_drm

# Verify modules loaded
echo "Verifying modules loaded correctly:"
lsmod | grep nvidia

echo "Attempting to restart NVIDIA services..."
systemctl restart nvidia-persistenced

echo "Reset complete. Testing NVIDIA with nvidia-smi:"
nvidia-smi
EOF
    chmod +x "$SCRIPT_DIR/nvidia-reset.sh"
    
    # Create multi-GPU management script
    cat > "$SCRIPT_DIR/nvidia-mgr.sh" << 'EOF'
#!/bin/bash
# NVIDIA GPU Manager for Multi-GPU Systems (K80 and 3090 Ti)
# Use with caution - requires root privileges for some operations

if [ "$1" = "reset" ] || [ "$1" = "restart" ]; then
    if [ "$(id -u)" -ne 0 ]; then
        echo "Reset operation requires root privileges. Please run with sudo."
        exit 1
    fi
fi

# Function to show help
show_help() {
    echo "NVIDIA GPU Manager for Multi-GPU Systems"
    echo "----------------------------------------"
    echo "Usage: $(basename $0) [command]"
    echo ""
    echo "Commands:"
    echo "  status      - Show status of all GPUs"
    echo "  info        - Show detailed information for all GPUs"
    echo "  perf        - Show performance metrics"
    echo "  reset       - Reset all GPU drivers (requires root)"
    echo "  compute     - Set all GPUs to compute mode (requires root)"
    echo "  graphics    - Set all GPUs to graphics mode (requires root)"
    echo "  optimize    - Apply optimal settings for K80+3090Ti setup (requires root)"
    echo "  monitor     - Start continuous monitoring"
    echo "  help        - Show this help"
}

# Function to show GPU status
show_status() {
    echo "Current GPU Status:"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=table
}

# Function to show detailed GPU information
show_info() {
    nvidia-info.sh
}

# Function to show performance metrics
show_perf() {
    echo "GPU Performance Metrics:"
    echo "----------------------"
    nvidia-smi --query-gpu=index,name,pstate,clocks.current.graphics,clocks.current.memory --format=table
    
    echo -e "\nGPU Power and Thermal:"
    echo "---------------------"
    nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit,fan.speed --format=table
    
    echo -e "\nGPU Utilization:"
    echo "---------------"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,utilization.encoder,utilization.decoder --format=table
    
    echo -e "\nGPU Memory:"
    echo "-----------"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=table
}

# Function to reset GPU drivers
reset_gpu() {
    echo "Resetting NVIDIA GPU drivers..."
    nvidia-reset.sh
}

# Function to set all GPUs to compute mode
set_compute_mode() {
    echo "Setting all GPUs to compute mode..."
    
    # Get list of GPUs
    GPUS=$(nvidia-smi --query-gpu=index,name --format=csv,noheader)
    
    # Process each GPU
    echo "$GPUS" | while IFS=, read -r IDX NAME; do
        # Trim whitespace
        IDX=$(echo "$IDX" | xargs)
        NAME=$(echo "$NAME" | xargs)
        
        echo "Configuring GPU $IDX: $NAME for compute..."
        
        # Apply settings based on GPU type
        if [[ "$NAME" == *"K80"* || "$NAME" == *"Tesla K80"* ]]; then
            # K80 compute settings
            nvidia-smi -i "$IDX" -ac 2505,875
            nvidia-smi -i "$IDX" --auto-boost-default=0
            nvidia-smi -i "$IDX" -pl 150
        elif [[ "$NAME" == *"3090"* || "$NAME" == *"RTX 3090"* ]]; then
            # 3090 Ti compute settings
            nvidia-smi -i "$IDX" -ac 1395,1590
            nvidia-smi -i "$IDX" --auto-boost-default=0
            nvidia-smi -i "$IDX" -pl 350
        fi
    done
    
    # Set compute mode
    nvidia-smi --compute-mode=0
    
    echo "All GPUs set to compute mode."
}

# Function to set all GPUs to graphics mode
set_graphics_mode() {
    echo "Setting all GPUs to graphics mode..."
    
    # Get list of GPUs
    GPUS=$(nvidia-smi --query-gpu=index,name --format=csv,noheader)
    
    # Process each GPU
    echo "$GPUS" | while IFS=, read -r IDX NAME; do
        # Trim whitespace
        IDX=$(echo "$IDX" | xargs)
        NAME=$(echo "$NAME" | xargs)
        
        echo "Configuring GPU $IDX: $NAME for graphics..."
        
        # Reset application clocks to default
        nvidia-smi -i "$IDX" -rac
        
        # Apply settings based on GPU type
        if [[ "$NAME" == *"K80"* || "$NAME" == *"Tesla K80"* ]]; then
            # K80 doesn't need graphics optimizations
            nvidia-smi -i "$IDX" --auto-boost-default=0
            nvidia-smi -i "$IDX" -pl 150
        elif [[ "$NAME" == *"3090"* || "$NAME" == *"RTX 3090"* ]]; then
            # 3090 Ti graphics settings - enable auto boost
            nvidia-smi -i "$IDX" --auto-boost-default=1
            nvidia-smi -i "$IDX" -pl 400
        fi
    done
    
    # Make sure we're not in exclusive compute mode
    nvidia-smi --compute-mode=0
    
    echo "All GPUs set to graphics mode."
}

# Function to apply optimal settings for K80+3090Ti setup
optimize_multi_gpu() {
    echo "Applying optimal settings for K80+3090Ti multi-GPU setup..."
    
    # Get list of GPUs
    GPUS=$(nvidia-smi --query-gpu=index,name --format=csv,noheader)
    
    # Process each GPU
    echo "$GPUS" | while IFS=, read -r IDX NAME; do
        # Trim whitespace
        IDX=$(echo "$IDX" | xargs)
        NAME=$(echo "$NAME" | xargs)
        
        echo "Optimizing GPU $IDX: $NAME..."
        
        # Apply settings based on GPU type
        if [[ "$NAME" == *"K80"* || "$NAME" == *"Tesla K80"* ]]; then
            echo "  Applying K80 compute-optimized settings..."
            # Set to maximum performance mode for compute
            nvidia-smi -i "$IDX" -ac 2505,875
            nvidia-smi -i "$IDX" --auto-boost-default=0
            nvidia-smi -i "$IDX" -pl 150
        elif [[ "$NAME" == *"3090"* || "$NAME" == *"RTX 3090"* ]]; then
            echo "  Applying RTX 3090 Ti balanced settings..."
            # Set balanced mode for 3090 Ti 
            nvidia-smi -i "$IDX" --auto-boost-default=1
            # Apply moderate power limit
            nvidia-smi -i "$IDX" -pl 380
        fi
    done
    
    # Set overall compute mode
    nvidia-smi --compute-mode=0
    
    echo "Optimal settings applied for multi-GPU setup."
}

# Function to start continuous monitoring
start_monitor() {
    nvidia-monitor.sh
}

# Main command processing
case "$1" in
    status|"")
        show_status
        ;;
    info)
        show_info
        ;;
    perf)
        show_perf
        ;;
    reset|restart)
        reset_gpu
        ;;
    compute)
        if [ "$(id -u)" -ne 0 ]; then
            echo "This operation requires root privileges. Please run with sudo."
            exit 1
        fi
        set_compute_mode
        ;;
    graphics)
        if [ "$(id -u)" -ne 0 ]; then
            echo "This operation requires root privileges. Please run with sudo."
            exit 1
        fi
        set_graphics_mode
        ;;
    optimize)
        if [ "$(id -u)" -ne 0 ]; then
            echo "This operation requires root privileges. Please run with sudo."
            exit 1
        fi
        optimize_multi_gpu
        ;;
    monitor)
        start_monitor
        ;;
    help|-h|--help)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
EOF
    chmod +x "$SCRIPT_DIR/nvidia-mgr.sh"
    
    # Create symbolic links for easier access
    ln -sf "$SCRIPT_DIR/nvidia-mgr.sh" "$SCRIPT_DIR/gpumgr"
    
    echo "✅ NVIDIA GPU utilities created in $SCRIPT_DIR"
    echo "  - nvidia-monitor.sh: Real-time monitoring"
    echo "  - nvidia-info.sh: Detailed information"
    echo "  - nvidia-reset.sh: Reset GPU modules"
    echo "  - nvidia-mgr.sh (gpumgr): Multi-GPU management tool"
}

main() {
    check_root
    check_gpu
 #  check_blacklisted_modules
    backup_existing_config
    remove_existing_nvidia
    install_dependencies
    add_nvidia_repo
#   disable_nouveau
    install_nvidia_driver
    configure_nvidia
    verify_installation
    create_gpu_utils
    
    echo "
✨ NVIDIA Driver 470 Installation Complete! ✨

🔧 Installation Summary:
- Driver: NVIDIA 470.x for multi-GPU system (K80 + RTX 3090 Ti)
- Config: Optimized for both compute and graphics workloads
- Utils: Advanced GPU management and monitoring tools installed

📦 Utilities Available:
- nvidia-monitor.sh: Real-time GPU monitoring
- nvidia-info.sh: Detailed GPU information
- nvidia-reset.sh: Reset GPU modules if issues occur
- nvidia-mgr.sh (gpumgr): Multi-GPU management tool
  - Use 'gpumgr help' to see all available commands
  - 'gpumgr optimize' will apply optimal settings for your K80 + 3090 Ti setup
  - 'gpumgr compute' will optimize for compute workloads
  - 'gpumgr graphics' will optimize for graphics performance

⚠️ Important Notes:
- A system reboot is recommended to ensure the driver is properly loaded
- For CUDA installation, this driver is compatible with CUDA 11.4
- The multi-GPU management tool will auto-detect and apply appropriate settings for each GPU
- Check the installation log at $LOG_FILE for any issues

Would you like to reboot now? (y/N)"
    read -r FINAL_REBOOT
    if [[ "$FINAL_REBOOT" =~ ^[Yy]$ ]]; then
        echo "🔄 Rebooting system in 5 seconds..."
        sleep 5
        reboot
    else
        echo "✅ Installation complete! Remember to reboot your system to fully load the driver."
    fi
}

# Execute main function
main