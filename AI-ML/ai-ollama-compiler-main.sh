#!/bin/bash
# Ollama Custom Compiler for Multiple GPU Types
# Part of DARKPOOL AI/ML Installer
# Date: March 2, 2025
set -e

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}=== DARKPOOL Ollama Multi-GPU Compiler ===${NC}"
echo "This script will compile Ollama for ROCm (RX580) and CUDA (K80) GPUs"

# Ensure script is run as root
if [ "$(id -u)" != "0" ]; then
    echo -e "${RED}${BOLD}This script must be run as root${NC}"
    exit 1
fi

# Determine the correct username: prefer SUDO_USER, then LOGNAME, then whoami
TARGET_USER="${SUDO_USER:-${LOGNAME:-$(whoami)}}"
if [ "$TARGET_USER" = "root" ]; then
    echo "⚠️ Warning: Running as root, trying to guess the real user..."
    # Try to find a non-root user from /home directory
    FIRST_USER=$(ls -1 /home | head -n 1)
    if [ -n "$FIRST_USER" ]; then
        TARGET_USER="$FIRST_USER"
        echo "ℹ️ Using first user found in /home: $TARGET_USER"
    fi
fi

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check if ROCm is installed
    if [ ! -d "/opt/rocm" ]; then
        echo -e "${RED}Error: ROCm not found in /opt/rocm${NC}"
        echo "Please install ROCm 6.3.3 first using 02-rocm-6-3-3-installer.sh"
        exit 1
    fi
    
    # Check if CUDA is installed
    if [ ! -d "/usr/local/cuda" ]; then
        echo -e "${RED}Error: CUDA not found in /usr/local/cuda${NC}"
        echo "Please install CUDA 11.8 first using 01-cuda-11-8-installer.sh"
        exit 1
    fi
    
    # Check if Go is installed (needed for Ollama compilation)
    if ! command -v go &> /dev/null; then
        echo -e "${YELLOW}Go is not installed. Installing Go...${NC}"
        apt update
        apt install -y golang-go
    fi
    
    # Check if required build tools are installed
    if ! command -v gcc &> /dev/null || ! command -v cmake &> /dev/null; then
        echo -e "${YELLOW}Installing required build tools...${NC}"
        apt update
        apt install -y build-essential cmake git
    fi
    
    echo -e "${GREEN}Prerequisites check passed${NC}"
}

# Create build directories
create_build_dirs() {
    echo -e "${YELLOW}Creating build directories...${NC}"
    mkdir -p "/home/$TARGET_USER/AI/Ollama-build"
    sudo chown -R "$TARGET_USER:$TARGET_USER" "/home/$TARGET_USER/AI/Ollama-build"
    echo -e "${GREEN}Build directories created${NC}"
}

# Compile Ollama for ROCm (RX580/gfx803)
compile_ollama_rocm() {
    echo -e "${BLUE}${BOLD}=== Building Ollama for ROCm (RX580/gfx803) ===${NC}"
    
    # Create working directory
    BUILD_DIR="/home/$TARGET_USER/AI/Ollama-build/rocm"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    sudo chown -R "$TARGET_USER:$TARGET_USER" "$BUILD_DIR"
    
    # Prepare environment variables
    echo -e "${YELLOW}Setting up ROCm environment variables...${NC}"
    export HSA_OVERRIDE_GFX_VERSION=8.0.3
    export ROC_ENABLE_PRE_VEGA=1
    export PYTORCH_ROCM_ARCH=gfx803
    export HIP_VISIBLE_DEVICES=0
    export HSA_ENABLE_SDMA=0
    export HCC_AMDGPU_TARGET=gfx803
    export TORCH_BLAS_PREFER_HIPBLASLT=0
    export ROCM_ARCH=gfx803
    export DAMDGPU_TARGETS=gfx803
    export OLLAMA_LLM_LIBRARY=rocm_v60002
    
    # Ensure the ROCm symlink exists
    if [ ! -L "/opt/rocm" ] || [ ! -d "/opt/rocm" ]; then
        echo -e "${YELLOW}Creating symlink for /opt/rocm...${NC}"
        ln -sf $(ls -d /opt/rocm-* | sort -V | tail -n 1) /opt/rocm
    fi
    
    # Download and build rocBLAS for gfx803 if needed
    if [ ! -f "/opt/rocm/lib/librocblas.so" ]; then
        echo -e "${YELLOW}Building rocBLAS for gfx803...${NC}"
        cd "$BUILD_DIR"
        git clone --depth=1 https://github.com/ROCmSoftwarePlatform/rocBLAS.git
        cd rocBLAS
        ./install.sh -ida gfx803
    fi
    
    # Clone Ollama repository
    echo -e "${YELLOW}Cloning Ollama repository...${NC}"
    cd "$BUILD_DIR"
    if [ ! -d "ollama" ]; then
        sudo -u "$TARGET_USER" git clone https://github.com/ollama/ollama.git
    fi
    cd ollama
    sudo -u "$TARGET_USER" git fetch --tags
    LATEST_TAG=$(git tag | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n 1)
    sudo -u "$TARGET_USER" git checkout $LATEST_TAG
    
    # Modify Ollama code to support gfx803
    echo -e "${YELLOW}Modifying Ollama source for gfx803 support...${NC}"
    
    # Modify gpu.go to support gfx803
    sudo -u "$TARGET_USER" sed -i 's/var RocmComputeMajorMin = "9"/var RocmComputeMajorMin = "8"/' discover/gpu.go
    
    # Modify CMake files to include gfx803
    sudo -u "$TARGET_USER" sed -i 's/"gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-"/"gfx803;gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-"/g' CMakePresets.json
    sudo -u "$TARGET_USER" sed -i 's/list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(900|94[012]|101[02]|1030|110[012])$")/list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(803|900|94[012]|101[02]|1030|110[012])$")/g' CMakeLists.txt
    
    # Compile Ollama backend for gfx803
    echo -e "${YELLOW}Compiling Ollama backend for gfx803...${NC}"
    cd "$BUILD_DIR/ollama"
    sudo -u "$TARGET_USER" cmake -B build -DAMDGPU_TARGETS=gfx803 -DLLAMA_HIPBLAS=ON
    sudo -u "$TARGET_USER" cmake --build build -j$(nproc)
    
    # Compile Ollama frontend
    echo -e "${YELLOW}Compiling Ollama frontend...${NC}"
    sudo -u "$TARGET_USER" bash -c "cd $BUILD_DIR/ollama && go generate ./... && go build ."
    
    # Install the binary
    echo -e "${YELLOW}Installing Ollama binary for ROCm...${NC}"
    cp "$BUILD_DIR/ollama/ollama" /usr/local/bin/ollama-rocm
    chmod +x /usr/local/bin/ollama-rocm
    
    # Create necessary directories
    mkdir -p /usr/share/ollama-rocm/.ollama/models
    chown -R "$TARGET_USER:$TARGET_USER" /usr/share/ollama-rocm
    
    # Create systemd service for ROCm Ollama
    echo -e "${YELLOW}Creating systemd service for ROCm Ollama...${NC}"
    cat > /etc/systemd/system/ollama-rocm.service << EOF
[Unit]
Description=Ollama ROCm Instance (RX580)
After=network.target

[Service]
User=$TARGET_USER
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_MODELS=/usr/share/ollama-rocm/.ollama/models"
Environment="HSA_OVERRIDE_GFX_VERSION=8.0.3"
Environment="ROC_ENABLE_PRE_VEGA=1"
Environment="PYTORCH_ROCM_ARCH=gfx803"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="HSA_ENABLE_SDMA=0"
Environment="HCC_AMDGPU_TARGET=gfx803"
Environment="TORCH_BLAS_PREFER_HIPBLASLT=0"
Environment="ROCM_ARCH=gfx803"
Environment="DAMDGPU_TARGETS=gfx803"
Environment="OLLAMA_LLM_LIBRARY=rocm_v60002"
ExecStart=/usr/local/bin/ollama-rocm serve
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

    echo -e "${GREEN}Ollama for ROCm (RX580/gfx803) built successfully!${NC}"
}

# Compile Ollama for CUDA (K80) - GPU0
compile_ollama_cuda0() {
    echo -e "${BLUE}${BOLD}=== Building Ollama for CUDA K80 (GPU 0) ===${NC}"
    
    # Create working directory
    BUILD_DIR="/home/$TARGET_USER/AI/Ollama-build/cuda0"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    sudo chown -R "$TARGET_USER:$TARGET_USER" "$BUILD_DIR"
    
    # Set CUDA environment variables
    echo -e "${YELLOW}Setting up CUDA environment variables...${NC}"
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_ARCH="35 37"
    export CUDA_COMPUTE_MAJOR=3
    export CUDA_COMPUTE_MINOR=7
    
    # Clone Ollama repository
    echo -e "${YELLOW}Cloning Ollama repository...${NC}"
    cd "$BUILD_DIR"
    if [ ! -d "ollama" ]; then
        sudo -u "$TARGET_USER" git clone https://github.com/ollama/ollama.git
    fi
    cd ollama
    sudo -u "$TARGET_USER" git fetch --tags
    LATEST_TAG=$(git tag | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n 1)
    sudo -u "$TARGET_USER" git checkout $LATEST_TAG
    
    # Compile Ollama backend for CUDA
    echo -e "${YELLOW}Compiling Ollama backend for CUDA...${NC}"
    cd "$BUILD_DIR/ollama"
    sudo -u "$TARGET_USER" cmake -B build -DCMAKE_CUDA_ARCHITECTURES="35;37" -DLLAMA_CUBLAS=ON
    sudo -u "$TARGET_USER" cmake --build build -j$(nproc)
    
    # Compile Ollama frontend
    echo -e "${YELLOW}Compiling Ollama frontend...${NC}"
    sudo -u "$TARGET_USER" bash -c "cd $BUILD_DIR/ollama && go generate ./... && go build ."
    
    # Install the binary
    echo -e "${YELLOW}Installing Ollama binary for CUDA GPU 0...${NC}"
    cp "$BUILD_DIR/ollama/ollama" /usr/local/bin/ollama-k80-gpu0
    chmod +x /usr/local/bin/ollama-k80-gpu0
    
    # Create necessary directories
    mkdir -p /usr/share/ollama-k80-gpu0/.ollama/models
    chown -R "$TARGET_USER:$TARGET_USER" /usr/share/ollama-k80-gpu0
    
    # Create systemd service for CUDA GPU 0
    echo -e "${YELLOW}Creating systemd service for CUDA GPU 0...${NC}"
    cat > /etc/systemd/system/ollama-k80-gpu0.service << EOF
[Unit]
Description=Ollama CUDA Instance (K80 GPU0)
After=network.target

[Service]
User=$TARGET_USER
Environment="OLLAMA_HOST=0.0.0.0:11435"
Environment="OLLAMA_MODELS=/usr/share/ollama-k80-gpu0/.ollama/models"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="CUDA_ARCH=35 37"
Environment="CUDA_COMPUTE_MAJOR=3"
Environment="CUDA_COMPUTE_MINOR=7"
ExecStart=/usr/local/bin/ollama-k80-gpu0 serve
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

    echo -e "${GREEN}Ollama for CUDA K80 (GPU 0) built successfully!${NC}"
}

# Compile Ollama for CUDA (K80) - GPU1
compile_ollama_cuda1() {
    echo -e "${BLUE}${BOLD}=== Building Ollama for CUDA K80 (GPU 1) ===${NC}"
    
    # Create working directory
    BUILD_DIR="/home/$TARGET_USER/AI/Ollama-build/cuda1"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    sudo chown -R "$TARGET_USER:$TARGET_USER" "$BUILD_DIR"
    
    # Set CUDA environment variables
    echo -e "${YELLOW}Setting up CUDA environment variables...${NC}"
    export CUDA_VISIBLE_DEVICES=1
    export CUDA_ARCH="35 37"
    export CUDA_COMPUTE_MAJOR=3
    export CUDA_COMPUTE_MINOR=7
    
    # Clone Ollama repository
    echo -e "${YELLOW}Cloning Ollama repository...${NC}"
    cd "$BUILD_DIR"
    if [ ! -d "ollama" ]; then
        sudo -u "$TARGET_USER" git clone https://github.com/ollama/ollama.git
    fi
    cd ollama
    sudo -u "$TARGET_USER" git fetch --tags
    LATEST_TAG=$(git tag | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n 1)
    sudo -u "$TARGET_USER" git checkout $LATEST_TAG
    
    # Compile Ollama backend for CUDA
    echo -e "${YELLOW}Compiling Ollama backend for CUDA...${NC}"
    cd "$BUILD_DIR/ollama"
    sudo -u "$TARGET_USER" cmake -B build -DCMAKE_CUDA_ARCHITECTURES="35;37" -DLLAMA_CUBLAS=ON
    sudo -u "$TARGET_USER" cmake --build build -j$(nproc)
    
    # Compile Ollama frontend
    echo -e "${YELLOW}Compiling Ollama frontend...${NC}"
    sudo -u "$TARGET_USER" bash -c "cd $BUILD_DIR/ollama && go generate ./... && go build ."
    
    # Install the binary
    echo -e "${YELLOW}Installing Ollama binary for CUDA GPU 1...${NC}"
    cp "$BUILD_DIR/ollama/ollama" /usr/local/bin/ollama-k80-gpu1
    chmod +x /usr/local/bin/ollama-k80-gpu1
    
    # Create necessary directories
    mkdir -p /usr/share/ollama-k80-gpu1/.ollama/models
    chown -R "$TARGET_USER:$TARGET_USER" /usr/share/ollama-k80-gpu1
    
    # Create systemd service for CUDA GPU 1
    echo -e "${YELLOW}Creating systemd service for CUDA GPU 1...${NC}"
    cat > /etc/systemd/system/ollama-k80-gpu1.service << EOF
[Unit]
Description=Ollama CUDA Instance (K80 GPU1)
After=network.target

[Service]
User=$TARGET_USER
Environment="OLLAMA_HOST=0.0.0.0:11436"
Environment="OLLAMA_MODELS=/usr/share/ollama-k80-gpu1/.ollama/models"
Environment="CUDA_VISIBLE_DEVICES=1"
Environment="CUDA_ARCH=35 37"
Environment="CUDA_COMPUTE_MAJOR=3"
Environment="CUDA_COMPUTE_MINOR=7"
ExecStart=/usr/local/bin/ollama-k80-gpu1 serve
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

    echo -e "${GREEN}Ollama for CUDA K80 (GPU 1) built successfully!${NC}"
}

# Configure systemd services
configure_services() {
    echo -e "${YELLOW}Reloading systemd and enabling services...${NC}"
    systemctl daemon-reload
    systemctl enable ollama-rocm.service
    systemctl enable ollama-k80-gpu0.service
    systemctl enable ollama-k80-gpu1.service
    echo -e "${GREEN}Services configured successfully!${NC}"
}

# Main function
main() {
    check_prerequisites
    create_build_dirs
    
    # Ask which GPUs to build for
    echo -e "${BLUE}${BOLD}Which Ollama instances would you like to build?${NC}"
    echo "1) All instances (ROCm/RX580 and CUDA/K80 x2)"
    echo "2) ROCm (RX580) only"
    echo "3) CUDA K80 (both GPUs) only"
    echo "4) CUDA K80 (GPU 0) only"
    echo "5) CUDA K80 (GPU 1) only"
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            compile_ollama_rocm
            compile_ollama_cuda0
            compile_ollama_cuda1
            ;;
        2)
            compile_ollama_rocm
            ;;
        3)
            compile_ollama_cuda0
            compile_ollama_cuda1
            ;;
        4)
            compile_ollama_cuda0
            ;;
        5)
            compile_ollama_cuda1
            ;;
        *)
            echo -e "${RED}Invalid choice. Exiting.${NC}"
            exit 1
            ;;
    esac
    
    configure_services
    
    echo -e "${GREEN}${BOLD}=== Ollama Multi-GPU Build Complete! ===${NC}"
    echo "To start the services, run:"
    echo "  sudo systemctl start ollama-rocm"
    echo "  sudo systemctl start ollama-k80-gpu0"
    echo "  sudo systemctl start ollama-k80-gpu1"
    echo ""
    echo "To check service status, run:"
    echo "  sudo systemctl status ollama-rocm"
    echo "  sudo systemctl status ollama-k80-gpu0"
    echo "  sudo systemctl status ollama-k80-gpu1"
    echo ""
    echo "The services are now configured to start automatically on boot."
}

# Run main function
main