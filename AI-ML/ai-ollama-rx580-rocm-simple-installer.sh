#!/bin/bash
# Simple installer for Ollama with ROCm support for RX 580

# Set directories
OLLAMA_DIR="/home/heathen-admin/Ollama-ROCm"
BINDIR="/usr/local/bin"
ROCM_PATH="/opt/rocm"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Simple Ollama RX 580 Installer ===${NC}"

# Stop any existing service
echo -e "${YELLOW}Stopping any existing ollama-rocm service...${NC}"
systemctl stop ollama-rocm 2>/dev/null || true
systemctl disable ollama-rocm 2>/dev/null || true
rm -f /etc/systemd/system/ollama-rocm.service 2>/dev/null || true

# Remove binary
echo -e "${YELLOW}Removing any existing ollama-rocm binary...${NC}"
rm -f "$BINDIR/ollama-rocm" 2>/dev/null || true

# Setup directory
echo -e "${BLUE}Setting up directories...${NC}"
rm -rf "$OLLAMA_DIR" 2>/dev/null || true
mkdir -p "$OLLAMA_DIR/models" "$OLLAMA_DIR/data"
chown -R heathen-admin:heathen-admin "$OLLAMA_DIR"

# Clone and build
echo -e "${BLUE}Cloning Ollama...${NC}"
cd "$OLLAMA_DIR"
git clone https://github.com/ollama/ollama.git
cd ollama

# Get latest tag
LATEST_TAG=$(git describe --tags --abbrev=0)
echo -e "${GREEN}Latest tag: $LATEST_TAG${NC}"
git checkout $LATEST_TAG

# Patch for GFX803
echo -e "${BLUE}Patching for GFX803 support...${NC}"
# Find GPU detection file
GPU_DETECT_FILE=$(find . -name "gpu.go" -path "*/discover/*" | head -n 1)
if [ -n "$GPU_DETECT_FILE" ]; then
  sed -i 's/RocmComputeMajorMin = "9"/RocmComputeMajorMin = "8"/g' "$GPU_DETECT_FILE"
  echo -e "${GREEN}Patched $GPU_DETECT_FILE${NC}"
fi

# Patch CMake files
CMAKE_PRESETS_FILE="llm/generate/builtins/CMakePresets.json"
if [ -f "$CMAKE_PRESETS_FILE" ]; then
  sed -i 's/"gfx900"/"gfx803", "gfx900"/g' "$CMAKE_PRESETS_FILE"
  echo -e "${GREEN}Patched $CMAKE_PRESETS_FILE${NC}"
fi

CMAKE_LISTS_FILE="llm/generate/builtins/CMakeLists.txt"
if [ -f "$CMAKE_LISTS_FILE" ]; then
  sed -i 's/AMDGPU_TARGETS "gfx900"/AMDGPU_TARGETS "gfx803;gfx900"/g' "$CMAKE_LISTS_FILE"
  echo -e "${GREEN}Patched $CMAKE_LISTS_FILE${NC}"
fi

# Set environment variables
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH"
export ROC_ENABLE_PRE_VEGA=1
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export CGO_ENABLED=1

# Build
echo -e "${BLUE}Building Ollama with ROCm support...${NC}"
echo -e "${YELLOW}This will take several minutes...${NC}"
GODEBUG=asyncpreemptoff=1 go build -v -ldflags="-X=github.com/ollama/ollama/version.Version=$LATEST_TAG-rocm" -tags="hip,rocm,experimental" -o ollama-rocm

# Install
echo -e "${BLUE}Installing binary...${NC}"
cp ollama-rocm "$BINDIR/"

# Create systemd service
echo -e "${BLUE}Creating systemd service...${NC}"
cat > /etc/systemd/system/ollama-rocm.service << EOF
[Unit]
Description=Ollama with ROCm Support for GFX803
After=network-online.target

[Service]
User=heathen-admin
Group=heathen-admin
WorkingDirectory=$OLLAMA_DIR/data
ExecStart=$BINDIR/ollama-rocm serve
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=127.0.0.1:11435"
Environment="OLLAMA_MODELS=$OLLAMA_DIR/models"
Environment="ROCM_PATH=$ROCM_PATH"
Environment="LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib"
Environment="ROC_ENABLE_PRE_VEGA=1"
Environment="HSA_OVERRIDE_GFX_VERSION=8.0.3"

[Install]
WantedBy=multi-user.target
EOF

# Set up system environment
echo -e "${BLUE}Setting up environment...${NC}"
echo "OLLAMA_HOST=127.0.0.1:11435" > /etc/profile.d/ollama-rocm.sh
echo "export OLLAMA_HOST=127.0.0.1:11435" >> /etc/profile.d/ollama-rocm.sh
echo "OLLAMA_HOST=127.0.0.1:11435" >> /etc/environment

# Enable and start service
echo -e "${BLUE}Starting service...${NC}"
systemctl daemon-reload
systemctl enable ollama-rocm
systemctl start ollama-rocm

echo -e "\n${GREEN}Installation complete!${NC}"
echo -e "${YELLOW}You can now use Ollama with:${NC}"
echo -e "  ollama-rocm run llama3.2:3b \"Hello world\""