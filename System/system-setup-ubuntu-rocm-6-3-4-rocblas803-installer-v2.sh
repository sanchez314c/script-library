#!/bin/bash
#########################################################################
#     ____   ___   ____  __  __     __   _____  _  _  ____             #
#    |  _ \ / _ \ / ___||  \/  |   / _| | ____|| \| ||  _ \            #
#    | |_) | | | | |    | |\/| |  | |_  |  _|  | .` || | | |           #
#    |  _ <| |_| | |___ | |  | |  |  _| | |___ | |\  || |_| |          #
#    |_| \_\\___/ \____||_|  |_|  |_|   |_____||_| \_||____/           #
#                                                                       #
#########################################################################
#
# 🚀 Premium ROCm 6.3.4 Installation Script for RX580 GPUs
# Version: 5.0.0
# Date: April 15, 2025
# Author: Cortana AI
# Description: Ultimate ROCm setup for RX580 with optimized gfx803 support
#              Includes built-from-source rocBLAS and auto-fixes for common issues
#
# Usage: sudo bash ./ubuntu-rocm-6-3-4-installer-ENHANCED.sh [uninstall]
#
#########################################################################

# Strict mode
set -e

# Create a timestamp for backup and log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Control verbosity with DEBUG variable
DEBUG=${DEBUG:-false}
if [ "$DEBUG" = "true" ]; then
    set -x  # Show commands when executed
fi

# Identify the correct user even when run with sudo
TARGET_USER=$(logname 2>/dev/null || echo $SUDO_USER || echo $USER)
if [ "$TARGET_USER" = "root" ] && [ -n "$SUDO_USER" ]; then
    TARGET_USER="$SUDO_USER"
fi

# Create directories
BACKUP_DIR="/home/${TARGET_USER}/rocm-backups/${TIMESTAMP}"
LOG_DIR="/home/${TARGET_USER}/logs"
mkdir -p "$BACKUP_DIR" "$LOG_DIR"
chmod -R 755 "$BACKUP_DIR" "$LOG_DIR"
chown -R ${TARGET_USER}:${TARGET_USER} "$BACKUP_DIR" "$LOG_DIR" 2>/dev/null || true

# Set up logging
LOG_FILE="${LOG_DIR}/rocm_install_${TIMESTAMP}.log"
SCRIPT_NAME=$(basename "$0")
exec &> >(tee -a "$LOG_FILE")

# Print colored messages
print_green() { echo -e "\e[32m$1\e[0m"; }
print_yellow() { echo -e "\e[33m$1\e[0m"; }
print_red() { echo -e "\e[31m$1\e[0m"; }
print_blue() { echo -e "\e[34m$1\e[0m"; }

# Print banners and section headers
print_banner() {
    local text="$1"
    local width=70
    local padding=$(( (width - ${#text}) / 2 ))
    local line=$(printf '═%.0s' $(seq 1 $width))
    
    echo ""
    print_blue "$line"
    printf "\e[34m%${padding}s$text%${padding}s\e[0m\n" "" ""
    print_blue "$line"
    echo ""
}

print_section() {
    local text="$1"
    echo ""
    print_blue "▓▒░ $text ░▒▓"
}

print_banner "ROCm 6.3.4 Installation - RX580 Optimized"
echo "📋 Script: $SCRIPT_NAME"
echo "📅 Date: $(date)"
echo "👤 User: $TARGET_USER"
echo "📝 Log: $LOG_FILE"
echo ""

check_root() {
    print_section "🔍 Checking for root privileges"
    
    if [ "$EUID" -ne 0 ]; then
        print_red "❌ Error: This script requires root privileges."
        print_red "   Please run with: sudo bash $SCRIPT_NAME"
        exit 1
    fi
    
    print_green "✅ Running with root privileges"
}

detect_gpu() {
    print_section "🔍 Detecting AMD GPU hardware"
    
    if ! command -v lspci &>/dev/null; then
        print_yellow "⚠️ lspci not found, installing pciutils..."
        apt-get update -qq && apt-get install -y pciutils
    fi
    
    AMD_GPU_INFO=$(lspci | grep -i "AMD\|ATI\|Radeon" | grep -i "RX\|Graphics")
    if [ -z "$AMD_GPU_INFO" ]; then
        print_yellow "⚠️ Warning: No AMD GPU detected with lspci."
        print_yellow "   This script is optimized for RX580/gfx803 GPUs."
        read -p "Continue anyway? (y/N): " CONTINUE
        if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            print_red "❌ Installation aborted by user."
            exit 1
        fi
    else
        print_green "✅ AMD GPU detected:"
        echo "$AMD_GPU_INFO"
        
        if echo "$AMD_GPU_INFO" | grep -q "RX 580"; then
            print_green "✅ RX580 GPU confirmed - perfect for this installer!"
        else
            print_yellow "⚠️ This script is optimized for RX580/gfx803 GPUs."
            print_yellow "   Your GPU may require different optimizations."
            read -p "Continue anyway? (y/N): " CONTINUE
            if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
                print_red "❌ Installation aborted by user."
                exit 1
            fi
        fi
    fi
}

backup_rocblas() {
    print_section "💾 Backing up any existing rocBLAS packages"
    
    ROCBLAS_BACKUP_DIR="${BACKUP_DIR}/rocblas"
    mkdir -p "$ROCBLAS_BACKUP_DIR"
    
    # First check for system rocBLAS packages
    if dpkg -l | grep -q rocblas; then
        print_yellow "🔍 Found system rocBLAS packages, backing up details..."
        dpkg -l | grep rocblas > "${ROCBLAS_BACKUP_DIR}/rocblas_packages.list"
        dpkg -l | grep rocblas | awk '{print $2}' | xargs -I{} dpkg-query -s {} | grep "Package\|Status\|Version" > "${ROCBLAS_BACKUP_DIR}/packages_info.txt"
    else
        echo "No system rocBLAS packages currently installed." > "${ROCBLAS_BACKUP_DIR}/rocblas_packages.list"
    fi
    
    # Check for previously built rocBLAS packages
    if [ -d "/home/${TARGET_USER}/rocBLAS-build/build/release" ]; then
        print_yellow "🔍 Found previous rocBLAS build, backing up packages..."
        cp -v /home/${TARGET_USER}/rocBLAS-build/build/release/rocblas_*.deb "${ROCBLAS_BACKUP_DIR}/" 2>/dev/null || true
        cp -v /home/${TARGET_USER}/rocBLAS-build/build/release/rocblas-dev_*.deb "${ROCBLAS_BACKUP_DIR}/" 2>/dev/null || true
    else
        echo "No previous rocBLAS build found." > "${ROCBLAS_BACKUP_DIR}/build_status.txt"
    fi
    
    # Ensure proper ownership
    chown -R ${TARGET_USER}:${TARGET_USER} "$ROCBLAS_BACKUP_DIR" 2>/dev/null || true
    
    print_green "✅ Backup completed to ${ROCBLAS_BACKUP_DIR}"
}

# clean_existing_rocm() {
#     print_section "🧹 Cleaning existing ROCm installation"
    
#     # Check if there are running processes using ROCm
#     if [ -d "/opt/rocm" ] || [ -d "/opt/rocm-6.3.4" ]; then
#         print_yellow "🔍 Checking for processes using ROCm libraries..."
#         ROCM_PROCESSES=$(fuser -v /opt/rocm*/lib/lib* 2>/dev/null || true)
        
#         if [ -n "$ROCM_PROCESSES" ]; then
#             print_yellow "⚠️ Found processes using ROCm libraries:"
#             echo "$ROCM_PROCESSES"
#             print_yellow "🔪 Terminating these processes..."
#             fuser -k /opt/rocm*/lib/lib* 2>/dev/null || true
#             sleep 2
#         fi
#     fi
    
#     print_yellow "🧹 Removing any existing ROCm packages..."
    
#     # Handle problematic packages first with force options
#     pkgs_to_force_remove=(
#         "rocm-llvm"
#         "rocm-hip-sdk"
#         "rocm-hip-libraries"
#         "rocm-libs"
#         "amdgpu-install"
#     )
    
#     for pkg in "${pkgs_to_force_remove[@]}"; do
#         if dpkg -l | grep -q "^ii.*$pkg"; then
#             print_yellow "  ↪ Force removing $pkg..."
#             dpkg --force-all -P "$pkg" 2>/dev/null || true
#         fi
#     done
    
#     # Main package patterns to remove
#     pkg_patterns=(
#         "rocm*"
#         "hip*"
#         "comgr*"
#         "hsa*"
#         "rocblas*"
#         "amdgpu*"
#         "libdrm-amdgpu*"
#         "libgl1-amdgpu-mesa*"
#         "libglapi-amdgpu-mesa*"
#         "libllvm*-amdgpu"
#         "openmp-extras*"
#         "roctracer*"
#     )
    
#     # Try to remove packages with dpkg first
#     for pattern in "${pkg_patterns[@]}"; do
#         print_yellow "  ↪ Removing packages matching $pattern..."
#         pkgs=$(dpkg -l | grep "^ii" | grep "$pattern" | awk '{print $2}' || true)
#         if [ -n "$pkgs" ]; then
#             echo "$pkgs" | xargs -r dpkg --force-all -P 2>/dev/null || true
#         fi
#     done
    
#     # Try apt purge as fallback
#     print_yellow "  ↪ Running apt purge on remaining packages..."
#     apt-get purge -y "${pkg_patterns[@]}" 2>/dev/null || true
#     apt-get autoremove -y 2>/dev/null || true
    
#     # Remove repositories and keys
#     print_yellow "🧹 Removing ROCm repositories and keys..."
#     rm -f /etc/apt/sources.list.d/amdgpu.list
#     rm -f /etc/apt/sources.list.d/rocm*.list
#     rm -f /etc/apt/trusted.gpg.d/rocm-*
    
#     # Clean APT cache
#     print_yellow "🧹 Cleaning APT cache..."
#     apt-get clean
#     apt-get update -qq || {
#         print_yellow "⚠️ apt-get update failed, but continuing..."
#     }
    
#     # Remove ROCm directories
#     print_yellow "🧹 Removing ROCm directories..."
#     for dir in /opt/rocm*; do
#         if [ -d "$dir" ]; then
#             print_yellow "  ↪ Removing $dir..."
#             rm -rf "$dir" || {
#                 print_yellow "⚠️ Could not fully remove $dir, using fallback method..."
#                 find "$dir" -type f -delete 2>/dev/null || true
#                 find "$dir" -type l -delete 2>/dev/null || true
#                 find "$dir" -type d -empty -delete 2>/dev/null || true
#             }
#         fi
#     done
    
#     # Remove ldconfig entries
#     print_yellow "🧹 Removing ROCm ldconfig entries..."
#     rm -f /etc/ld.so.conf.d/rocm*.conf
#     ldconfig
    
#     print_green "✅ Cleanup completed"
# }

install_dependencies() {
    print_section "📦 Installing essential dependencies"
    
    # Update package index
    print_yellow "🔄 Updating package index..."
    apt-get update -qq || {
        print_yellow "⚠️ apt-get update failed, attempting to continue..."
    }
    
    # List of essential dependencies
    DEPS=(
        "linux-headers-$(uname -r)"
        "linux-modules-extra-$(uname -r)"
        wget
        gnupg
        software-properties-common
        build-essential
        git
        python3-dev
        python3-pip
        cmake
        ninja-build
        libopenmpi3
        libstdc++-12-dev
        libdnnl-dev
        libopenblas-dev
        libpng-dev
        libjpeg-dev
    )
    
    # Install dependencies
    print_yellow "📦 Installing packages..."
    for pkg in "${DEPS[@]}"; do
        if dpkg -l | grep -q "^ii.*$(echo $pkg | cut -d'=' -f1)"; then
            echo "  ✓ $pkg already installed"
        else
            print_yellow "  ↪ Installing $pkg..."
            apt-get install -y "$pkg" || {
                print_yellow "⚠️ Failed to install $pkg, continuing..."
            }
        fi
    done
    
    print_green "✅ Dependencies installed"
}

download_amdgpu_installer() {
    print_section "📥 Downloading and installing AMD GPU installer"
    
    cd /tmp
    rm -f amdgpu-install_6.3.60304-1_all.deb*
    
    print_yellow "📥 Downloading AMDGPU installer..."
    INSTALLER_URL="https://repo.radeon.com/amdgpu-install/6.3.4/ubuntu/noble/amdgpu-install_6.3.60304-1_all.deb"
    
    wget -q --show-progress "$INSTALLER_URL" || {
        print_red "❌ Download failed. Checking connectivity to repo.radeon.com..."
        if ping -c 1 repo.radeon.com &>/dev/null; then
            print_yellow "⚠️ Can ping repo.radeon.com but download failed."
            print_yellow "⚠️ Trying alternative download method..."
            
            # Fallback to curl
            if command -v curl &>/dev/null; then
                curl -L "$INSTALLER_URL" -o amdgpu-install_6.3.60304-1_all.deb || {
                    print_red "❌ Alternative download method failed."
                    exit 1
                }
            else
                apt-get install -y curl
                curl -L "$INSTALLER_URL" -o amdgpu-install_6.3.60304-1_all.deb || {
                    print_red "❌ Alternative download method failed."
                    exit 1
                }
            fi
        else
            print_red "❌ Cannot reach repo.radeon.com. Check your internet connection."
            exit 1
        fi
    }
    
    print_yellow "🔧 Installing AMDGPU installer package..."
    dpkg -i amdgpu-install_6.3.60304-1_all.deb || {
        print_yellow "⚠️ dpkg install failed, trying apt fix..."
        apt-get install -f -y
        dpkg -i amdgpu-install_6.3.60304-1_all.deb || {
            print_red "❌ Failed to install AMDGPU installer package even after fix."
            exit 1
        }
    }
    
    print_green "✅ AMDGPU installer package installed"
}

install_rocm() {
    print_section "🚀 Installing ROCm 6.3.4"
    
    print_yellow "🔄 Updating package lists..."
    apt-get update -qq || {
        print_yellow "⚠️ apt-get update failed, attempting to continue..."
    }
    
    print_yellow "🚀 Running AMDGPU installer for ROCm..."
    print_yellow "   This may take several minutes..."
    
    amdgpu-install --usecase=graphics,rocm,rocmdev,rocmdevtools,opencl,openclsdk,hip,hiplibsdk,mllib,mlsdk -y || {
        print_red "❌ ROCm installation failed"
        print_yellow "🔍 Checking for detailed diagnostic information..."
        
        # Check GPU detection
        if ! lspci | grep -i "RX 580" &>/dev/null; then
            print_red "❌ No RX580 GPU detected. This could be the reason for installation failure."
        fi
        
        # Check for specific errors in apt logs
        if grep -q "Broken packages" /var/log/apt/term.log 2>/dev/null; then
            print_yellow "⚠️ Detected broken packages issue. Attempting recovery..."
            apt-get update
            apt-get install -f -y --no-install-recommends
            print_yellow "🔄 Retrying ROCm installation with minimal components..."
            amdgpu-install --usecase=rocm -y || {
                print_red "❌ Minimal ROCm installation also failed."
                print_red "   Please check the log file for details: $LOG_FILE"
                exit 1
            }
        else
            print_red "❌ Installation failed without known recovery method."
            print_red "   Please check the log file for details: $LOG_FILE"
            exit 1
        fi
    }
    
    print_yellow "🔧 Ensuring core components are present..."
    apt-get install -y hip-dev hip-runtime-amd rocm-dev rocm-libs || {
        print_yellow "⚠️ Some core packages failed to install, attempting fix..."
        apt-get install -f -y
        apt-get install -y hip-dev hip-runtime-amd rocm-dev rocm-libs || {
            print_red "❌ Failed to install core ROCm packages."
            exit 1
        }
    }
    
    print_green "✅ ROCm 6.3.4 installed successfully"
}

build_rocblas() {
    print_section "🔨 Building optimized rocBLAS for RX580 (gfx803)"
    
    # Set up environment variables for the build
    export ROCM_PATH=/opt/rocm-6.3.4
    export HIP_PLATFORM=amd
    export PATH=$ROCM_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
    
    print_yellow "📂 Setting up build directory..."
    # Create build directory with proper permissions
    BUILD_DIR="/home/${TARGET_USER}/rocBLAS-build"
    mkdir -p "$BUILD_DIR"
    chown -R ${TARGET_USER}:${TARGET_USER} "$BUILD_DIR"
    
    # Remove old build if it exists
    if [ -d "${BUILD_DIR}/rocBLAS" ]; then
        print_yellow "🧹 Removing previous rocBLAS source directory..."
        rm -rf "${BUILD_DIR}/rocBLAS"
    fi
    
    # Run the build as the target user
    print_yellow "🔨 Building rocBLAS from source (this may take 15-30 minutes)..."
    sudo -u "$TARGET_USER" bash -c "
        set -e
        cd '$BUILD_DIR'
        
        echo '📥 Cloning rocBLAS repository...'
        git clone https://github.com/ROCmSoftwarePlatform/rocBLAS.git
        cd rocBLAS
        
        echo '🔄 Switching to rocm-6.3.0 branch...'
        git fetch --tags
        git checkout rocm-6.3.0 -b rocm-6.3.0-custom
        
        echo '🔧 Accelerating build with parallel make...'
        sed -i 's/\"make\"/\"make -j\$(nproc)\"/' rmake.py
        
        echo '🔧 Setting up compiler environment...'
        export CMAKE_PREFIX_PATH='$ROCM_PATH'
        export CXX='$ROCM_PATH/llvm/bin/clang++'
        export CC='$ROCM_PATH/llvm/bin/clang'
        
        echo '🚀 Starting build specifically for gfx803 architecture...'
        ./install.sh -ida gfx803
    " || {
        print_red "❌ rocBLAS build failed."
        print_yellow "⚠️ Check the log file for details: $LOG_FILE"
        exit 1
    }
    
    print_green "✅ rocBLAS built successfully for gfx803 architecture"
}

install_custom_rocblas() {
    print_section "📦 Installing custom rocBLAS packages"
    
    ROCBLAS_BUILD_DIR="/home/${TARGET_USER}/rocBLAS-build/rocBLAS/build/release"
    
    if [ ! -d "$ROCBLAS_BUILD_DIR" ]; then
        print_red "❌ Custom rocBLAS build directory not found: $ROCBLAS_BUILD_DIR"
        print_yellow "⚠️ Continuing with system rocBLAS, performance may not be optimal for RX580."
        return 1
    fi
    
    if ! ls ${ROCBLAS_BUILD_DIR}/rocblas_*.deb ${ROCBLAS_BUILD_DIR}/rocblas-dev_*.deb &>/dev/null; then
        print_red "❌ Custom rocBLAS packages not found in: $ROCBLAS_BUILD_DIR"
        print_yellow "⚠️ Continuing with system rocBLAS, performance may not be optimal for RX580."
        return 1
    fi
    
    # Backup these custom packages
    print_yellow "💾 Backing up custom rocBLAS packages..."
    CUSTOM_ROCBLAS_BACKUP="/home/${TARGET_USER}/rocblas-custom-packages"
    mkdir -p "$CUSTOM_ROCBLAS_BACKUP"
    cp -v ${ROCBLAS_BUILD_DIR}/rocblas_*.deb ${ROCBLAS_BUILD_DIR}/rocblas-dev_*.deb "$CUSTOM_ROCBLAS_BACKUP/" 2>/dev/null || true
    chown -R ${TARGET_USER}:${TARGET_USER} "$CUSTOM_ROCBLAS_BACKUP" 2>/dev/null || true
    
    # Remove system rocBLAS first to avoid conflicts
    print_yellow "🧹 Removing system rocBLAS packages..."
    apt-get remove -y rocblas rocblas-dev 2>/dev/null || true
    
    # Install our custom packages
    print_yellow "📦 Installing custom rocBLAS packages..."
    cd "$ROCBLAS_BUILD_DIR"
    dpkg -i rocblas_*.deb rocblas-dev_*.deb || {
        print_yellow "⚠️ Initial installation failed, fixing dependencies..."
        apt-get install -f -y
        dpkg -i rocblas_*.deb rocblas-dev_*.deb || {
            print_red "❌ Failed to install custom rocBLAS packages"
            print_yellow "🔄 Restoring system packages..."
            apt-get install --reinstall -y rocblas rocblas-dev
            apt-get install -f -y
            return 1
        }
    }
    
    # Create helper scripts for reinstallation and system restore
    print_yellow "📝 Creating helper scripts for future maintenance..."
    
    # Script to reinstall custom rocBLAS
    cat > "/home/${TARGET_USER}/reinstall-custom-rocblas.sh" << EOF
#!/bin/bash
# Helper script to reinstall custom rocBLAS packages
set -e

BACKUP_DIR="/home/${TARGET_USER}/rocblas-custom-packages"
if [ ! -d "\$BACKUP_DIR" ]; then
    echo "⚠️ Backup directory not found!"
    exit 1
fi

echo "🔧 Removing system rocBLAS packages..."
sudo apt-get remove -y rocblas rocblas-dev || true

echo "🔧 Reinstalling custom rocBLAS packages..."
sudo dpkg -i \$BACKUP_DIR/rocblas_*.deb \$BACKUP_DIR/rocblas-dev_*.deb
sudo apt-get install -f -y
echo "✅ Reinstallation complete!"
EOF
    chmod +x "/home/${TARGET_USER}/reinstall-custom-rocblas.sh"
    chown ${TARGET_USER}:${TARGET_USER} "/home/${TARGET_USER}/reinstall-custom-rocblas.sh"
    
    # Script to restore system rocBLAS
    cat > "/home/${TARGET_USER}/restore-system-rocblas.sh" << EOF
#!/bin/bash
echo "Restoring system rocBLAS packages..."
sudo apt-get remove -y rocblas rocblas-dev || true
sudo apt-get install --reinstall rocblas rocblas-dev
sudo apt-get install -f -y
echo "System rocBLAS packages restored"
EOF
    chmod +x "/home/${TARGET_USER}/restore-system-rocblas.sh"
    chown ${TARGET_USER}:${TARGET_USER} "/home/${TARGET_USER}/restore-system-rocblas.sh"
    
    print_green "✅ Custom rocBLAS installed with helper scripts"
    return 0
}

configure_system() {
    print_section "🔧 Configuring system for ROCm"
    
    ROCM_PATH="/opt/rocm-6.3.4"
    
    print_yellow "🔧 Setting up ROCm environment in /etc/environment..."
    # Backup the original /etc/environment file
    cp /etc/environment "${BACKUP_DIR}/environment.bak"
    
    # Update environment variables
    ENV_UPDATES=(
        "ROCM_PATH=${ROCM_PATH}"
        "HIP_PLATFORM=amd"
        "PATH=${ROCM_PATH}/bin:\$PATH"
        "LD_LIBRARY_PATH=${ROCM_PATH}/lib:\$LD_LIBRARY_PATH"
    )
    
    for env_var in "${ENV_UPDATES[@]}"; do
        var_name=$(echo "$env_var" | cut -d'=' -f1)
        var_value=$(echo "$env_var" | cut -d'=' -f2-)
        
        if grep -q "^${var_name}=" /etc/environment; then
            # Variable exists, update it
            sed -i "s|^${var_name}=.*|${var_name}=${var_value}|" /etc/environment
        else
            # Variable doesn't exist, add it
            echo "${var_name}=${var_value}" >> /etc/environment
        fi
    done
    
    print_yellow "🔧 Setting up rocm.conf for ldconfig..."
    echo -e "${ROCM_PATH}/lib\n${ROCM_PATH}/lib64" > /etc/ld.so.conf.d/rocm.conf
    ldconfig
    
    print_yellow "🔧 Applying HIP directory fix..."
    mkdir -p "${ROCM_PATH}/hip"
    ln -sf "${ROCM_PATH}/include/hip" "${ROCM_PATH}/hip/include"
    ln -sf "${ROCM_PATH}/lib" "${ROCM_PATH}/hip/lib"
    ln -sf "${ROCM_PATH}/hip" "/opt/rocm/hip"
    
    print_yellow "🔧 Adding user to video and render groups..."
    usermod -a -G video,render "$TARGET_USER"
    
    print_yellow "🔧 Setting up device permissions..."
    # Set up kfd device permissions if it exists
    if [ -e /dev/kfd ]; then
        chmod 660 /dev/kfd
        chown root:render /dev/kfd
    fi
    
    # Set up DRI devices permissions
    for dev in /dev/dri/card* /dev/dri/render*; do
        if [ -e "$dev" ]; then
            chmod 660 "$dev"
            chown root:render "$dev"
        fi
    done
    
    print_yellow "🔧 Reloading udev rules..."
    if command -v udevadm &>/dev/null; then
        udevadm control --reload-rules
        udevadm trigger
    fi
    
    print_green "✅ System configured for ROCm"
}

create_helper_scripts() {
    print_section "📝 Creating helpful utility scripts"
    
    # Create script directory
    SCRIPTS_DIR="/home/${TARGET_USER}/rocm-scripts"
    mkdir -p "$SCRIPTS_DIR"
    
    # Create verification script
    cat > "${SCRIPTS_DIR}/verify-rocm.sh" << 'EOF'
#!/bin/bash
# ROCm Verification Script

echo "ROCm Verification Tool"
echo "======================"
echo ""

# Check ROCm binaries
echo "1. Checking ROCm executables..."
for cmd in rocminfo rocm-smi hipcc hipconfig; do
    echo -n "   $cmd: "
    if command -v $cmd &>/dev/null; then
        echo "✓ Found - $($cmd --version 2>/dev/null || echo 'Version info not available')"
    else
        echo "✗ Not found"
    fi
done

# Check environment variables
echo ""
echo "2. Checking environment variables..."
for var in ROCM_PATH HIP_PLATFORM; do
    echo -n "   $var: "
    if [ -n "${!var}" ]; then
        echo "✓ Set to ${!var}"
    else
        echo "✗ Not set"
    fi
done

# Check libraries
echo ""
echo "3. Checking ROCm libraries..."
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
libs=("libamdhip64.so" "librocblas.so" "libhsa-runtime64.so")
for lib in "${libs[@]}"; do
    echo -n "   $lib: "
    if ldconfig -p | grep -q "$lib"; then
        echo "✓ Found in library path"
    elif [ -f "$ROCM_PATH/lib/$lib" ]; then
        echo "! Found in $ROCM_PATH/lib but not in ldconfig"
    else
        echo "✗ Not found"
    fi
done

# Check GPU detection
echo ""
echo "4. Checking GPU detection..."
if command -v rocminfo &>/dev/null; then
    if rocminfo 2>/dev/null | grep -q "AMD"; then
        echo "   ✓ ROCm detects AMD GPU"
        rocminfo 2>/dev/null | grep -A 2 "Name:" | head -3
    else
        echo "   ✗ No AMD GPU detected by ROCm"
    fi
else
    echo "   ✗ rocminfo not available"
fi

# Check rocBLAS
echo ""
echo "5. Checking rocBLAS..."
if dpkg -l | grep -q rocblas; then
    echo "   ✓ rocBLAS installed"
    ver=$(dpkg -l | grep rocblas | head -1 | awk '{print $3}')
    echo "   Version: $ver"
    if [[ "$ver" == *"8ebd6c11"* ]] || [[ "$ver" == *"custom"* ]]; then
        echo "   ✓ Custom rocBLAS detected (optimized for RX580/gfx803)"
    else
        echo "   ! System rocBLAS detected (not optimized for RX580/gfx803)"
    fi
else
    echo "   ✗ rocBLAS not installed"
fi

echo ""
echo "Verification complete!"
EOF
    chmod +x "${SCRIPTS_DIR}/verify-rocm.sh"
    
    # Create benchmark script
    cat > "${SCRIPTS_DIR}/benchmark-rocm.sh" << 'EOF'
#!/bin/bash
# Simple ROCm Benchmarking Tool

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "ROCm Simple Benchmark"
echo "===================="
echo ""

# Create a simple benchmarking program
cat > benchmark.cpp << 'END'
#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

#define CHECK_HIP(cmd)                                                   \
{                                                                        \
    hipError_t error = cmd;                                              \
    if (error != hipSuccess) {                                           \
        std::cerr << "HIP error: " << hipGetErrorString(error) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl;           \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
}

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Print device info
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    std::cout << "Device: " << devProp.name << std::endl;
    std::cout << "Compute capability: " << devProp.major << "." << devProp.minor << std::endl;
    
    // Vector size
    const int N = 100000000;
    const size_t size = N * sizeof(float);
    
    // Host vectors
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N);
    
    // Device vectors
    float *d_A, *d_B, *d_C;
    CHECK_HIP(hipMalloc(&d_A, size));
    CHECK_HIP(hipMalloc(&d_B, size));
    CHECK_HIP(hipMalloc(&d_C, size));
    
    // Measure memory transfer time (host to device)
    auto start = std::chrono::high_resolution_clock::now();
    CHECK_HIP(hipMemcpy(d_A, h_A.data(), size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B.data(), size, hipMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> h2d_time = end - start;
    
    // Launch kernel and measure time
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {  // Run multiple iterations for more stable timing
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        CHECK_HIP(hipDeviceSynchronize());
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> kernel_time = end - start;
    
    // Measure memory transfer time (device to host)
    start = std::chrono::high_resolution_clock::now();
    CHECK_HIP(hipMemcpy(h_C.data(), d_C, size, hipMemcpyDeviceToHost));
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> d2h_time = end - start;
    
    // Verify result
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - 3.0f) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    // Print results
    std::cout << "Vector size: " << N << " elements" << std::endl;
    std::cout << "Transfers:" << std::endl;
    std::cout << "  Host to device: " << h2d_time.count() << " seconds" << std::endl;
    std::cout << "  Device to host: " << d2h_time.count() << " seconds" << std::endl;
    std::cout << "Kernel execution (10 iterations): " << kernel_time.count() << " seconds" << std::endl;
    std::cout << "Result verification: " << (correct ? "PASSED" : "FAILED") << std::endl;
    
    // Free memory
    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));
    CHECK_HIP(hipFree(d_C));
    
    return 0;
}
END

echo "Compiling benchmark..."
$ROCM_PATH/bin/hipcc -o benchmark benchmark.cpp

if [ $? -ne 0 ]; then
    echo "Failed to compile benchmark. Check if HIP SDK is properly installed."
    exit 1
fi

echo -e "\nRunning benchmark...\n"
./benchmark

echo -e "\nBenchmark complete"
echo -e "Temporary files in $TEMP_DIR\n"
EOF
    chmod +x "${SCRIPTS_DIR}/benchmark-rocm.sh"
    
    # Create system info script
    cat > "${SCRIPTS_DIR}/system-info.sh" << 'EOF'
#!/bin/bash
# ROCm System Information Tool

echo "ROCm System Information"
echo "======================"
echo ""

echo "System Information:"
echo "------------------"
echo "Hostname: $(hostname)"
echo "Kernel: $(uname -r)"
echo "OS: $(lsb_release -d 2>/dev/null | cut -f2- || cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo ""

echo "CPU Information:"
echo "--------------"
lscpu | grep "Model name\|Socket(s)\|Core(s) per socket\|CPU MHz\|CPU max MHz"
echo ""

echo "Memory Information:"
echo "----------------"
free -h
echo ""

echo "GPU Information:"
echo "--------------"
if command -v lspci &>/dev/null; then
    lspci | grep -i "VGA\|Display\|3D\|AMD\|ATI\|Radeon"
else
    echo "lspci not available"
fi
echo ""

if command -v rocm-smi &>/dev/null; then
    echo "ROCm-SMI Output:"
    echo "--------------"
    rocm-smi
else
    echo "rocm-smi not available"
fi
echo ""

# Check environment variables
echo "ROCm Environment Variables:"
echo "-------------------------"
for var in ROCM_PATH HIP_PLATFORM PATH LD_LIBRARY_PATH; do
    echo "$var=${!var}"
done
echo ""

echo "Installed ROCm Packages:"
echo "---------------------"
dpkg -l | grep -i "rocm\|hip\|hsa\|amdgpu" | awk '{print $2 "\t" $3}'
echo ""

echo "System information gathering complete!"
EOF
    chmod +x "${SCRIPTS_DIR}/system-info.sh"
    
    # Create a README file explaining the scripts
    cat > "${SCRIPTS_DIR}/README.md" << 'EOF'
# ROCm Utility Scripts

This directory contains useful scripts for managing and monitoring your ROCm installation.

## Available Scripts

- **verify-rocm.sh**: Verifies your ROCm installation, checking components, libraries, and GPU detection
- **benchmark-rocm.sh**: Runs a simple HIP vector addition benchmark to test basic GPU compute performance
- **system-info.sh**: Gathers system information useful for debugging or reporting issues

## Usage

Each script can be run directly:

```bash
./verify-rocm.sh
./benchmark-rocm.sh
./system-info.sh
```

## Maintenance Scripts

Additional scripts in your home directory:

- **~/reinstall-custom-rocblas.sh**: Reinstalls the custom rocBLAS packages optimized for RX580/gfx803
- **~/restore-system-rocblas.sh**: Reverts to standard system rocBLAS packages

## Common Issues

If you encounter problems after system updates:

1. Run `sudo ldconfig` to update the library cache
2. Verify your installation with `./verify-rocm.sh`
3. If rocBLAS issues occur, run `~/reinstall-custom-rocblas.sh`
EOF
    
    # Set correct ownership
    chown -R ${TARGET_USER}:${TARGET_USER} "$SCRIPTS_DIR"
    
    print_green "✅ Utility scripts created in ${SCRIPTS_DIR}"
}

verify_installation() {
    print_section "🧪 Verifying ROCm installation"
    
    # Load environment variables to ensure everything's available
    export ROCM_PATH=/opt/rocm-6.3.4
    export HIP_PLATFORM=amd
    export PATH=$ROCM_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
    
    # Check for critical binaries
    print_yellow "🔍 Checking for critical ROCm binaries..."
    MISSING_BINARIES=0
    
    for binary in rocminfo rocm-smi hipcc hipconfig; do
        if ! command -v $binary &>/dev/null; then
            print_red "❌ Missing binary: $binary"
            MISSING_BINARIES=1
        else
            print_green "✅ Found $binary: $(which $binary)"
            # Show version if available
            $binary --version &>/dev/null && $binary --version | head -1
        fi
    done
    
    if [ $MISSING_BINARIES -eq 1 ]; then
        print_yellow "⚠️ Some critical binaries are missing. Installation may be incomplete."
    fi
    
    # Check for AMD GPU detection
    print_yellow "🔍 Checking AMD GPU detection with rocminfo..."
    if command -v rocminfo &>/dev/null; then
        if rocminfo 2>/dev/null | grep -q "AMD"; then
            print_green "✅ ROCm detects AMD GPU:"
            rocminfo 2>/dev/null | grep -A 2 "Name:" | head -3
        else
            print_red "❌ No AMD GPU detected by ROCm. This could indicate a driver issue."
        fi
    else
        print_red "❌ rocminfo not available, skipping GPU detection check."
    fi
    
    # Check file structure
    print_yellow "🔍 Checking ROCm directory structure..."
    if [ -d "/opt/rocm-6.3.4" ]; then
        print_green "✅ ROCm 6.3.4 installation directory exists"
    else
        print_red "❌ ROCm 6.3.4 installation directory missing"
    fi
    
    if [ -d "/opt/rocm-6.3.4/hip" ]; then
        print_green "✅ HIP directory exists"
    else
        print_red "❌ HIP directory missing - HIP fixes may not have been applied"
    fi
    
    if [ -L "/opt/rocm/hip" ]; then
        print_green "✅ HIP symlink exists"
    else
        print_red "❌ HIP symlink missing - HIP fixes may not have been applied"
    fi
    
    # Check rocBLAS
    print_yellow "🔍 Checking rocBLAS installation..."
    if dpkg -l | grep -q rocblas; then
        print_green "✅ rocBLAS packages installed:"
        dpkg -l | grep rocblas | grep ^ii
        
        # Check if these are custom packages
        rocblas_version=$(dpkg -l | grep rocblas | head -1 | awk '{print $3}')
        if [[ "$rocblas_version" == *"8ebd6c11"* ]] || [[ "$rocblas_version" == *"custom"* ]]; then
            print_green "✅ Custom rocBLAS detected (optimized for RX580/gfx803)"
        else
            print_yellow "ℹ️  System rocBLAS detected (not optimized for RX580/gfx803)"
        fi
    else
        print_red "❌ rocBLAS not installed"
    fi
    
    # Check user groups
    print_yellow "🔍 Checking user group membership..."
    if id -nG "$TARGET_USER" | grep -qw "video"; then
        print_green "✅ User is in video group"
    else
        print_red "❌ User is not in video group"
    fi
    
    if id -nG "$TARGET_USER" | grep -qw "render"; then
        print_green "✅ User is in render group"
    else
        print_red "❌ User is not in render group"
    fi
    
    print_green "✅ ROCm verification complete"
}

# Uninstall function
uninstall_all() {
    print_banner "ROCm 6.3.4 Uninstallation"
    
    check_root
    
    print_yellow "Creating backup before uninstallation..."
    backup_rocblas
    
    print_red "🧹 Uninstalling ROCm completely..."
    clean_existing_rocm
    
    print_green "✅ ROCm has been completely uninstalled."
    print_yellow "The backup of your settings and packages is available at: ${BACKUP_DIR}"
    
    echo ""
    print_yellow "To reinstall, run this script without the 'uninstall' parameter."
    exit 0
}

print_completion_message() {
    print_banner "ROCm 6.3.4 Installation Complete!"
    
    cat << EOF
$(print_green "✅ ROCm 6.3.4 has been successfully installed with RX580/gfx803 optimizations.")

📋 $(print_blue "IMPORTANT NOTES:")
$(print_yellow "• Custom rocBLAS packages optimized for RX580/gfx803 have been installed")
$(print_yellow "• Helper scripts are available in /home/${TARGET_USER}/rocm-scripts/")
$(print_yellow "• Maintenance scripts:")
  - /home/${TARGET_USER}/reinstall-custom-rocblas.sh - Reinstall custom packages
  - /home/${TARGET_USER}/restore-system-rocblas.sh - Restore system packages

🔄 $(print_blue "AFTER REBOOTING:")
$(print_yellow "• Log out and log back in for group changes to take effect")
$(print_yellow "• Run 'sudo ldconfig' to ensure library paths are updated")
$(print_yellow "• Verify your installation with: /home/${TARGET_USER}/rocm-scripts/verify-rocm.sh")

📝 $(print_blue "MORE INFORMATION:")
$(print_yellow "• Installation log: ${LOG_FILE}")
$(print_yellow "• Backups saved to: ${BACKUP_DIR}")

$(print_blue "Thank you for using ROCm 6.3.4 Enhanced Installer!")
EOF
}

main() {
    # Check if we're uninstalling
    if [ "$1" = "uninstall" ]; then
        uninstall_all
        exit 0
    fi
    
    # Perform installation
    check_root
    detect_gpu
    backup_rocblas
#   clean_existing_rocm
    install_dependencies
    download_amdgpu_installer
    install_rocm
    build_rocblas
    install_custom_rocblas
    configure_system
    create_helper_scripts
    verify_installation
    print_completion_message
}

# Run the main function with all arguments
main "$@"
