#!/bin/bash
# ollama-rx580-gfx803-installer.sh
# Consolidated script for installing Ollama with ROCm support for GFX803 (AMD RX 580)
# Created by Claude on 2025-03-04

# Use error trapping instead of immediate exit
set +e

# Define error handler
error_handler() {
  local LINE=$1
  local ERR_CODE=$2
  echo -e "${RED}Error on line $LINE: Command exited with status $ERR_CODE${NC}"
  echo -e "${YELLOW}Check the log file $LOG_FILE for details${NC}"
}

# Set up error trap
trap 'error_handler ${LINENO} $?' ERR

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
LOG_FILE="ollama-rocm-install.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo -e "${BLUE}=== Ollama ROCm GFX803 Installer ===${NC}"
echo -e "${BLUE}This script will install Ollama with ROCm support for AMD RX 580 (GFX803)${NC}"
echo "Installation log will be saved to $LOG_FILE"
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run as root${NC}"
  exit 1
fi

# Check AMD GPU presence
if ! lspci | grep -i amd > /dev/null; then
  echo -e "${RED}No AMD GPU detected. This script is intended for AMD GPUs.${NC}"
  exit 1
fi

# Detect ROCm version or install if not present
if [ -d "/opt/rocm" ]; then
  ROCM_VERSION=$(ls -1 /opt/rocm 2>/dev/null | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n1)
  
  if [ -z "$ROCM_VERSION" ]; then
    ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null || echo "Unknown")
  fi
  
  echo -e "${GREEN}ROCm $ROCM_VERSION detected${NC}"
else
  echo -e "${YELLOW}ROCm not detected. Installing ROCm 6.3.3...${NC}"
  
  # Add ROCm apt repository
  wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
  echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.3.3 ubuntu main" | tee /etc/apt/sources.list.d/rocm.list
  
  # Update and install ROCm
  apt-get update -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true || echo -e "${YELLOW}Warning: apt update completed with errors but continuing...${NC}"
  apt-get install -y rocm-dev rocm-libs rocminfo rocm-cmake --no-install-recommends
  
  ROCM_VERSION="6.3.3"
  echo -e "${GREEN}ROCm $ROCM_VERSION installed${NC}"
fi

# Check if Go is installed
if ! command -v go &> /dev/null; then
  echo -e "${YELLOW}Go not found. Installing Go 1.23.4...${NC}"
  
  # Download and install Go
  wget -q https://go.dev/dl/go1.23.4.linux-amd64.tar.gz
  rm -rf /usr/local/go && tar -C /usr/local -xzf go1.23.4.linux-amd64.tar.gz
  rm go1.23.4.linux-amd64.tar.gz
  
  # Add Go to PATH
  echo 'export PATH=$PATH:/usr/local/go/bin' > /etc/profile.d/go.sh
  source /etc/profile.d/go.sh
  
  echo -e "${GREEN}Go 1.23.4 installed${NC}"
else
  GO_VERSION=$(go version | cut -d " " -f 3 | sed 's/go//')
  echo -e "${GREEN}Go $GO_VERSION detected${NC}"
  
  # Check if Go version is sufficient
  if [ "$(echo -e "$GO_VERSION\n1.21.0" | sort -V | head -n1)" != "1.21.0" ]; then
    echo -e "${YELLOW}Go version $GO_VERSION may be too old. Consider upgrading to 1.21.0+${NC}"
  fi
fi

# Install build dependencies
echo -e "${BLUE}Installing build dependencies...${NC}"
# Update with error handling for problematic repositories
apt-get update -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true || echo -e "${YELLOW}Warning: apt update completed with errors but continuing...${NC}"
apt-get install -y build-essential cmake git clang ninja-build --no-install-recommends

# Clone Ollama repository
echo -e "${BLUE}Cloning Ollama repository...${NC}"
INSTALL_DIR="/home/heathen-admin/Ollama-ROCm"
mkdir -p "$INSTALL_DIR"
chown heathen-admin:heathen-admin "$INSTALL_DIR"
cd "$INSTALL_DIR"
git clone https://github.com/ollama/ollama.git
cd ollama

# Get latest release tag
LATEST_TAG=$(git describe --tags --abbrev=0)
echo -e "${GREEN}Latest Ollama version: $LATEST_TAG${NC}"
echo -e "${YELLOW}Note: We're building from source to enable GFX803 support${NC}"

# Checkout the latest release
git checkout $LATEST_TAG

# Patch Ollama to support GFX803
echo -e "${BLUE}Patching Ollama for GFX803 support...${NC}"

# Find GPU detection file
echo -e "${YELLOW}Searching for GPU detection file...${NC}"
GPU_DETECT_FILE=$(find . -name "gpu.go" -path "*/discover/*" | head -n 1)
if [ -z "$GPU_DETECT_FILE" ]; then
  echo -e "${RED}Could not find GPU detection file. Aborting.${NC}"
  exit 1
fi

echo -e "${BLUE}============= PATCHING FILES =============${NC}"
echo -e "${GREEN}Found GPU detection file: $GPU_DETECT_FILE${NC}"

# Show the content before patching
echo -e "${YELLOW}Original GPU detection code:${NC}"
grep -n "RocmComputeMajorMin" "$GPU_DETECT_FILE" --context=3
echo

# Patch the file
echo -e "${YELLOW}Patching $GPU_DETECT_FILE to enable GFX803 support...${NC}"
sed -i 's/RocmComputeMajorMin = "9"/RocmComputeMajorMin = "8"/g' "$GPU_DETECT_FILE"

# Show the content after patching
echo -e "${GREEN}Patched GPU detection code:${NC}"
grep -n "RocmComputeMajorMin" "$GPU_DETECT_FILE" --context=3
echo

# Check for and patch CMake files for GFX803
CMAKE_PRESETS_FILE="llm/generate/builtins/CMakePresets.json"
if [ -f "$CMAKE_PRESETS_FILE" ]; then
  echo -e "${YELLOW}Original CMake Presets:${NC}"
  grep -n "gfx900" "$CMAKE_PRESETS_FILE" --context=1
  echo
  
  echo -e "${YELLOW}Patching $CMAKE_PRESETS_FILE to enable GFX803...${NC}"
  sed -i 's/"gfx900"/"gfx803", "gfx900"/g' "$CMAKE_PRESETS_FILE"
  
  echo -e "${GREEN}Patched CMake Presets:${NC}"
  grep -n "gfx803" "$CMAKE_PRESETS_FILE" --context=1
  echo
fi

CMAKE_LISTS_FILE="llm/generate/builtins/CMakeLists.txt"
if [ -f "$CMAKE_LISTS_FILE" ]; then
  echo -e "${YELLOW}Original CMakeLists:${NC}"
  grep -n "AMDGPU_TARGETS" "$CMAKE_LISTS_FILE" --context=1
  echo
  
  echo -e "${YELLOW}Patching $CMAKE_LISTS_FILE to enable GFX803...${NC}"
  sed -i 's/AMDGPU_TARGETS "gfx900"/AMDGPU_TARGETS "gfx803;gfx900"/g' "$CMAKE_LISTS_FILE"
  
  echo -e "${GREEN}Patched CMakeLists:${NC}"
  grep -n "AMDGPU_TARGETS" "$CMAKE_LISTS_FILE" --context=1
  echo
fi
echo -e "${BLUE}=========================================${NC}"

# Create the get_compiler shim to work around HIP linker issues
echo -e "${BLUE}Creating compiler shim for ROCm compatibility...${NC}"
cat > get_compiler.sh << 'EOF'
#!/bin/bash
cc="$1"
[ "$cc" = "cc" ] && exec /usr/bin/gcc "$@"
[ "$cc" = "c++" ] && exec /usr/bin/g++ "$@"
exec /usr/bin/ld "$@"
EOF
chmod +x get_compiler.sh

# Build Ollama with ROCm support
echo -e "${BLUE}Building Ollama with ROCm support...${NC}"

# Set environment variables for ROCm and GFX803 support
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export ROC_ENABLE_PRE_VEGA=1
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export OLLAMA_HOST=0.0.0.0:11435
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Display environment variables for debugging
echo -e "${BLUE}============= ENVIRONMENT VARIABLES =============${NC}"
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "ROC_ENABLE_PRE_VEGA=$ROC_ENABLE_PRE_VEGA"
echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "OLLAMA_HOST=$OLLAMA_HOST"
echo "CC=$CC"
echo "CXX=$CXX"
echo -e "${BLUE}=================================================${NC}"

# Check if ROCm is accessible
echo -e "${BLUE}ROCm Installation Check:${NC}"
if command -v rocminfo &>/dev/null; then
    echo -e "${GREEN}rocminfo found at $(which rocminfo)${NC}"
    echo -e "${YELLOW}GPU information (brief):${NC}"
    rocminfo | grep -A 5 "GPU Device" | grep -E 'Name|Marketing|Architecture'
else
    echo -e "${RED}rocminfo command not found. Check ROCm installation.${NC}"
fi
echo -e "${BLUE}=================================================${NC}"

# Build with HIP support - full verbose output
echo -e "${YELLOW}Building Ollama with full verbose output...${NC}"
echo -e "${BLUE}============= BUILD COMMAND =============${NC}"
echo -e "go build -v -x -ldflags=\"-X=github.com/ollama/ollama/version.Version=$LATEST_TAG-rocm\" -tags=\"hip,rocm,experimental\" -o ollama-rocm"
echo -e "${BLUE}=========================================${NC}"

# Use verbose build flags to show all commands and compilation steps
if ! GODEBUG=asyncpreemptoff=1 go build -v -x -ldflags="-X=github.com/ollama/ollama/version.Version=$LATEST_TAG-rocm" -tags="hip,rocm,experimental" -o ollama-rocm; then
  echo -e "${RED}Failed to build Ollama. See error messages above.${NC}"
  echo -e "${YELLOW}You may need to try again with a newer ROCm version or examine the log for specific errors.${NC}"
  exit 1
fi
echo -e "${GREEN}Build successful!${NC}"

# Install the binary
echo -e "${BLUE}Installing Ollama with ROCm support...${NC}"
cp ollama-rocm /usr/local/bin/

# Create systemd service
echo -e "${BLUE}Creating systemd service...${NC}"
cat > /etc/systemd/system/ollama-rocm.service << EOF
[Unit]
Description=Ollama with ROCm Support for GFX803
After=network-online.target

[Service]
Environment="PATH=/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:\${LD_LIBRARY_PATH}"
Environment="ROC_ENABLE_PRE_VEGA=1"
Environment="HSA_OVERRIDE_GFX_VERSION=8.0.3"
# Set host to listen on all interfaces
Environment="OLLAMA_HOST=0.0.0.0:11435"
# Add this critical environment variable for clients
Environment="OLLAMA_CLIENT_HOST=localhost:11435"
ExecStart=/usr/local/bin/ollama-rocm serve
Restart=always
RestartSec=5
User=heathen-admin
Group=heathen-admin
WorkingDirectory=/home/heathen-admin/Ollama-ROCm/data

[Install]
WantedBy=multi-user.target
EOF

# Create separate data directory for ollama-rocm
mkdir -p /home/heathen-admin/Ollama-ROCm/data
chown -R heathen-admin:heathen-admin /home/heathen-admin/Ollama-ROCm/data
chmod 755 /home/heathen-admin/Ollama-ROCm/data

# Reload systemd and enable service
systemctl daemon-reload
systemctl enable ollama-rocm
systemctl start ollama-rocm

# Install the binary directly - no wrappers needed
echo -e "${BLUE}Installing Ollama with ROCm support...${NC}"
cp ollama-rocm /usr/local/bin/

# Create verification script
echo -e "${BLUE}Creating verification script...${NC}"
cat > /usr/local/bin/verify-ollama-rocm << 'EOF'
#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Verifying Ollama ROCm Installation${NC}"

# Check service status
echo -e "\n${YELLOW}Checking ollama-rocm service status:${NC}"
systemctl status ollama-rocm --no-pager

# Check if service is listening on port
echo -e "\n${YELLOW}Checking if service is listening on port 11435:${NC}"
if netstat -tuln | grep 11435 > /dev/null; then
  echo -e "${GREEN}Service is listening on port 11435${NC}"
else
  echo -e "${RED}Service is NOT listening on port 11435${NC}"
fi

# Check GPU detection
echo -e "\n${YELLOW}Running GPU detection check:${NC}"
ollama-rocm devices

# Check rocminfo
echo -e "\n${YELLOW}Running rocminfo:${NC}"
rocminfo | grep -A 5 "GPU Device" | grep -E 'Name|Marketing|Architecture'

# Check environment variables
echo -e "\n${YELLOW}Checking environment variables:${NC}"
systemctl show ollama-rocm | grep Environment

echo -e "\n${BLUE}Verification complete${NC}"
EOF
chmod +x /usr/local/bin/verify-ollama-rocm

# Create global system-wide environment settings
echo -e "${BLUE}Setting up system-wide environment...${NC}"

# Add to /etc/environment to ensure variable is set system-wide
if ! grep -q "OLLAMA_HOST" /etc/environment; then
  echo 'OLLAMA_HOST=localhost:11435' >> /etc/environment
  echo -e "${GREEN}Added OLLAMA_HOST to /etc/environment${NC}"
fi

# Create profile script for aliases
cat > /etc/profile.d/ollama-rocm.sh << 'EOF'
# Set OLLAMA_HOST for all shells
export OLLAMA_HOST=localhost:11435

# Alias for convenience
alias or='ollama-rocm'
EOF

# Print completion message
echo -e "\n${GREEN}Ollama with ROCm support for GFX803 (AMD RX 580) has been installed!${NC}"
echo -e "${YELLOW}Service is running on port 11435${NC}"
echo -e "${YELLOW}You can use the 'or' command to interact with it:${NC}"
echo -e "  or pull llama3"
echo -e "  or run llama3 \"Hello, how are you?\""
echo -e "${YELLOW}To verify the installation, run:${NC}"
echo -e "  verify-ollama-rocm"
echo -e "${YELLOW}To see the service status:${NC}"
echo -e "  systemctl status ollama-rocm"
echo -e "\n${GREEN}Installation complete!${NC}"