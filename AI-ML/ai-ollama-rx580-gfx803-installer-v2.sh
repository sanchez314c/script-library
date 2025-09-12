#!/bin/bash
# ollama-rx580-gfx803-installer-2.sh
# Installer for Ollama with ROCm support for AMD RX 580 (GFX803)

# Enable trace and add error handling without immediate exit
set -x

# Define error handler function
error_handler() {
  local LINE=$1
  local STATUS=$2
  echo -e "${RED}Error on line $LINE: Command exited with status $STATUS${NC}"
  echo -e "${YELLOW}Continuing script execution...${NC}"
}

# Set up error trap to catch but not exit
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

# Directory settings
OLLAMA_DIR="/home/heathen-admin/Ollama-ROCm"
OLLAMA_MODELS_DIR="$OLLAMA_DIR/models"
OLLAMA_DATA_DIR="$OLLAMA_DIR/data"
BINDIR="/usr/local/bin"
ROCM_PATH="/opt/rocm"

# Check AMD GPU presence
if ! lspci | grep -i amd > /dev/null; then
  echo -e "${RED}No AMD GPU detected. This script is intended for AMD GPUs.${NC}"
  exit 1
fi

# Detect ROCm version
if [ -d "/opt/rocm" ]; then
  ROCM_VERSION=$(ls -1 /opt/rocm 2>/dev/null | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n1)
  
  if [ -z "$ROCM_VERSION" ]; then
    ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null || echo "Unknown")
  fi
  
  echo -e "${GREEN}ROCm $ROCM_VERSION detected${NC}"
else
  echo -e "${RED}ROCm not detected. Please install ROCm before running this script.${NC}"
  echo -e "${YELLOW}Recommended version: ROCm 6.3.3${NC}"
  exit 1
fi

# Check if Go is installed and get the latest version
echo -e "${BLUE}Checking for Go installation...${NC}"

# Get the latest Go version from the website
echo -e "${YELLOW}Fetching latest Go version...${NC}"
LATEST_GO_VERSION=$(curl -s https://go.dev/VERSION?m=text | head -n 1)
echo -e "${GREEN}Latest Go version is $LATEST_GO_VERSION${NC}"

# Remove "go" prefix if present
LATEST_GO_VERSION=${LATEST_GO_VERSION#go}

install_latest_go() {
  echo -e "${YELLOW}Installing Go $LATEST_GO_VERSION...${NC}"
  
  # Download and install Go
  wget -q "https://go.dev/dl/go${LATEST_GO_VERSION}.linux-amd64.tar.gz"
  rm -rf /usr/local/go && tar -C /usr/local -xzf "go${LATEST_GO_VERSION}.linux-amd64.tar.gz"
  rm "go${LATEST_GO_VERSION}.linux-amd64.tar.gz"
  
  # Add Go to PATH
  echo 'export PATH=$PATH:/usr/local/go/bin' > /etc/profile.d/go.sh
  source /etc/profile.d/go.sh
  
  echo -e "${GREEN}Go $LATEST_GO_VERSION installed${NC}"
}

if ! command -v go &> /dev/null; then
  # Go is not installed, install latest version
  install_latest_go
else
  # Go is installed, check version
  CURRENT_GO_VERSION=$(go version | cut -d " " -f 3 | sed 's/go//')
  echo -e "${GREEN}Go $CURRENT_GO_VERSION detected${NC}"
  
  # Compare versions (simple string comparison)
  if [ "$CURRENT_GO_VERSION" != "$LATEST_GO_VERSION" ]; then
    echo -e "${YELLOW}Updating Go from $CURRENT_GO_VERSION to $LATEST_GO_VERSION${NC}"
    install_latest_go
  else
    echo -e "${GREEN}Go is already at the latest version${NC}"
  fi
fi

# Verify Go installation
go version

# Install build dependencies
echo -e "${BLUE}Installing build dependencies...${NC}"
apt-get update -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true || echo -e "${YELLOW}Warning: apt update completed with errors but continuing...${NC}"
apt-get install -y build-essential cmake git clang ninja-build pkg-config lsof

# Check for existing service and remove it
echo -e "${BLUE}Checking for existing services...${NC}"
if systemctl is-active --quiet ollama-rocm; then
  echo -e "${YELLOW}Stopping existing ollama-rocm service...${NC}"
  systemctl stop ollama-rocm
  # Wait to ensure service is fully stopped
  sleep 3
fi

if systemctl is-enabled --quiet ollama-rocm; then
  echo -e "${YELLOW}Disabling existing ollama-rocm service...${NC}"
  systemctl disable ollama-rocm
fi

if [ -f "/etc/systemd/system/ollama-rocm.service" ]; then
  echo -e "${YELLOW}Removing existing service file...${NC}"
  rm -f /etc/systemd/system/ollama-rocm.service
fi

# Kill any remaining processes still using the binary
if [ -f "$BINDIR/ollama-rocm" ]; then
  echo -e "${YELLOW}Checking for and killing processes using ollama-rocm...${NC}"
  pids=$(lsof -t "$BINDIR/ollama-rocm" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo -e "${YELLOW}Found processes: $pids - killing them...${NC}"
    kill -9 $pids 2>/dev/null || true
    sleep 2
  else
    echo -e "${GREEN}No processes found using ollama-rocm${NC}"
  fi
  
  echo -e "${YELLOW}Removing existing ollama-rocm binary...${NC}"
  rm -f "$BINDIR/ollama-rocm"
fi

# Clean up any other related processes
echo -e "${YELLOW}Killing any other ollama processes...${NC}"
pkill -f "ollama" 2>/dev/null || echo -e "${GREEN}No ollama processes found to kill${NC}"
sleep 2

# Add debug pause to see if the script exits
echo -e "${BLUE}Debug checkpoint 1 - Continuing with installation...${NC}"

# Check for existing installation and remove it if found
if [ -d "$OLLAMA_DIR" ]; then
  echo -e "${YELLOW}Existing installation found at $OLLAMA_DIR${NC}"
  echo -e "${YELLOW}Removing existing installation...${NC}"
  rm -rf "$OLLAMA_DIR"
  echo -e "${GREEN}Removed existing installation${NC}"
fi

# Create fresh directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p "$OLLAMA_DIR" "$OLLAMA_MODELS_DIR" "$OLLAMA_DATA_DIR"
chown -R heathen-admin:heathen-admin "$OLLAMA_DIR"

# Clone Ollama
echo -e "${BLUE}Cloning Ollama repository...${NC}"
cd "$OLLAMA_DIR" || { echo -e "${RED}Failed to cd to $OLLAMA_DIR${NC}"; mkdir -p "$OLLAMA_DIR"; cd "$OLLAMA_DIR" || exit 1; }
git clone https://github.com/ollama/ollama.git || { echo -e "${RED}Failed to clone repository${NC}"; exit 1; }
cd ollama || { echo -e "${RED}Failed to cd to ollama directory${NC}"; exit 1; }

# Add debug checkpoint
echo -e "${BLUE}Debug checkpoint 2 - Successfully cloned repository...${NC}"

# Get latest release tag
LATEST_TAG=$(git describe --tags --abbrev=0)
echo -e "${GREEN}Latest Ollama version: $LATEST_TAG${NC}"

# Checkout the latest release
git checkout $LATEST_TAG

# Patch Ollama for GFX803 support
echo -e "${BLUE}Patching Ollama for GFX803 support...${NC}"
echo -e "${YELLOW}Searching for GPU detection file...${NC}"

# Find GPU detection files
GPU_DETECT_FILE=$(find . -name "gpu.go" -path "*/discover/*" | head -n 1)
if [ -z "$GPU_DETECT_FILE" ]; then
  echo -e "${RED}Could not find GPU detection file. Aborting.${NC}"
  exit 1
fi

echo -e "${GREEN}Found GPU detection file: $GPU_DETECT_FILE${NC}"
# Show original
echo -e "${YELLOW}Original GPU detection code:${NC}"
grep -n "RocmComputeMajorMin" "$GPU_DETECT_FILE" --context=2 || echo "Pattern not found"
# Patch for gfx803
sed -i 's/RocmComputeMajorMin = "9"/RocmComputeMajorMin = "8"/g' "$GPU_DETECT_FILE"
# Show patched
echo -e "${GREEN}Patched GPU detection code:${NC}"
grep -n "RocmComputeMajorMin" "$GPU_DETECT_FILE" --context=2 || echo "Pattern not found"

# Look for other files that need patching
echo -e "${BLUE}Checking for CMake files...${NC}"
CMAKE_PRESETS_FILE="llm/generate/builtins/CMakePresets.json"
if [ -f "$CMAKE_PRESETS_FILE" ]; then
  echo -e "${YELLOW}Original CMake Presets:${NC}"
  grep -n "gfx900" "$CMAKE_PRESETS_FILE" --context=1 || echo "Pattern not found"
  
  echo -e "${YELLOW}Patching $CMAKE_PRESETS_FILE to enable GFX803...${NC}"
  sed -i 's/"gfx900"/"gfx803", "gfx900"/g' "$CMAKE_PRESETS_FILE"
  
  echo -e "${GREEN}Patched CMake Presets:${NC}"
  grep -n "gfx803" "$CMAKE_PRESETS_FILE" --context=1 || echo "Pattern not found"
fi

CMAKE_LISTS_FILE="llm/generate/builtins/CMakeLists.txt"
if [ -f "$CMAKE_LISTS_FILE" ]; then
  echo -e "${YELLOW}Original CMakeLists:${NC}"
  grep -n "AMDGPU_TARGETS" "$CMAKE_LISTS_FILE" --context=1 || echo "Pattern not found"
  
  echo -e "${YELLOW}Patching $CMAKE_LISTS_FILE to enable GFX803...${NC}"
  sed -i 's/AMDGPU_TARGETS "gfx900"/AMDGPU_TARGETS "gfx803;gfx900"/g' "$CMAKE_LISTS_FILE"
  
  echo -e "${GREEN}Patched CMakeLists:${NC}"
  grep -n "AMDGPU_TARGETS" "$CMAKE_LISTS_FILE" --context=1 || echo "Pattern not found"
fi

# Set environment variables for build
echo -e "${BLUE}Setting up environment variables...${NC}"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH"
export ROC_ENABLE_PRE_VEGA=1
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export OLLAMA_HOST=127.0.0.1:11435

# Debug info
echo -e "${YELLOW}Environment variables:${NC}"
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "ROC_ENABLE_PRE_VEGA=$ROC_ENABLE_PRE_VEGA"
echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "OLLAMA_HOST=$OLLAMA_HOST"

# Debug checkpoint before build
echo -e "${BLUE}Debug checkpoint 3 - About to start build...${NC}"

# Build Ollama
echo -e "${BLUE}Building Ollama with ROCm support...${NC}"
echo -e "${YELLOW}This will take several minutes...${NC}"

# Set CGO variable explicitly
export CGO_ENABLED=1

# Build with error handling
if ! GODEBUG=asyncpreemptoff=1 go build -v -ldflags="-X=github.com/ollama/ollama/version.Version=$LATEST_TAG-rocm" -tags="hip,rocm,experimental" -o ollama-rocm; then
  echo -e "${RED}Build failed. See error messages above.${NC}"
  # Don't exit, try to continue
  echo -e "${YELLOW}Attempting to continue despite build errors...${NC}"
else
  echo -e "${GREEN}Build command completed successfully${NC}"
fi

# Debug checkpoint after build
echo -e "${BLUE}Debug checkpoint 4 - Build attempt completed...${NC}"

# Check if build was successful
if [ ! -f "ollama-rocm" ]; then
  echo -e "${RED}Build failed. Check the log for errors.${NC}"
  exit 1
fi

echo -e "${GREEN}Ollama built successfully!${NC}"

# Install the binary
echo -e "${BLUE}Installing Ollama binary...${NC}"
cp -v ollama-rocm $BINDIR/

# Double-check for any lingering processes before service creation
echo -e "${BLUE}Final check for any ollama processes...${NC}"
pkill -f "ollama" 2>/dev/null || true
sleep 1

# Create systemd service
echo -e "${BLUE}Creating new systemd service...${NC}"
cat > /etc/systemd/system/ollama-rocm.service << EOF
[Unit]
Description=Ollama with ROCm Support for GFX803
After=network-online.target

[Service]
User=heathen-admin
Group=heathen-admin
WorkingDirectory=$OLLAMA_DATA_DIR
ExecStart=$BINDIR/ollama-rocm serve
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=127.0.0.1:11435"
Environment="OLLAMA_MODELS=$OLLAMA_MODELS_DIR"
Environment="ROCM_PATH=$ROCM_PATH"
Environment="LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib"
Environment="ROC_ENABLE_PRE_VEGA=1"
Environment="HSA_OVERRIDE_GFX_VERSION=8.0.3"

[Install]
WantedBy=multi-user.target
EOF

# Clean up existing environment settings
echo -e "${BLUE}Cleaning up existing environment settings...${NC}"
if [ -f "/etc/profile.d/ollama-rocm.sh" ]; then
  echo -e "${YELLOW}Removing existing profile script...${NC}"
  rm -f /etc/profile.d/ollama-rocm.sh
fi

# Remove existing environment variable if present
if grep -q "OLLAMA_HOST" /etc/environment; then
  echo -e "${YELLOW}Removing OLLAMA_HOST from /etc/environment...${NC}"
  sed -i '/OLLAMA_HOST/d' /etc/environment
fi

# Set up system environment
echo -e "${BLUE}Setting up system environment...${NC}"
cat > /etc/profile.d/ollama-rocm.sh << 'EOF'
export OLLAMA_HOST=127.0.0.1:11435
alias or='ollama-rocm'
EOF

# Create a global environment variable
echo "OLLAMA_HOST=127.0.0.1:11435" >> /etc/environment

# Debug checkpoint before service operations
echo -e "${BLUE}Debug checkpoint 5 - About to manage systemd service...${NC}"

# Enable and start service with error checking
echo -e "${BLUE}Starting Ollama service...${NC}"
systemctl daemon-reload || echo -e "${YELLOW}Warning: daemon-reload had issues but continuing...${NC}"
systemctl enable ollama-rocm || echo -e "${YELLOW}Warning: Failed to enable service but continuing...${NC}"
systemctl start ollama-rocm || echo -e "${YELLOW}Warning: Failed to start service but continuing...${NC}"

# Debug checkpoint after service operations
echo -e "${BLUE}Debug checkpoint 6 - Service operations completed...${NC}"

# Wait for service to start
echo -e "${YELLOW}Waiting for service to start...${NC}"
sleep 5

# Print status
echo -e "${BLUE}Checking service status...${NC}"
systemctl status ollama-rocm --no-pager

# Verify ROCm detection
echo -e "${BLUE}Verifying ROCm detection...${NC}"
$BINDIR/ollama-rocm devices

echo -e "\n${GREEN}Installation complete!${NC}"
echo -e "${YELLOW}You can now use Ollama with ROCm support:${NC}"
echo -e "  ollama-rocm run llama3.2:3b \"Hello world\""
echo -e "  ollama-rocm list"
echo -e "${YELLOW}To view logs:${NC}"
echo -e "  journalctl -u ollama-rocm -f"