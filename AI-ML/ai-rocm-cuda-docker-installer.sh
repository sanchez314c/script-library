#!/bin/bash

# Multi-GPU Docker Environment Setup Script
# Author: Cortana
# Version: 1.3
set -e

# Color coding for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Starting Multi-GPU Docker Environment Setup...${NC}"

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success: $1${NC}"
    else
        echo -e "${RED}✗ Failed: $1${NC}"
        exit 1
    fi
}

# GPU Detection and Validation
detect_gpus() {
    echo -e "${BLUE}🔍 Detecting GPU hardware...${NC}"
    
    # Check for NVIDIA K80
    if nvidia-smi | grep -i "Tesla K80" > /dev/null; then
        echo -e "${GREEN}✓ NVIDIA Tesla K80 detected${NC}"
        NVIDIA_GPU=true
        # Get number of GPU dies
        K80_DIES=$(nvidia-smi -L | grep -c "Tesla K80")
        echo -e "${GREEN}✓ Found $K80_DIES K80 GPU dies${NC}"
    else
        echo -e "${RED}✗ NVIDIA Tesla K80 not found${NC}"
        NVIDIA_GPU=false
    fi
}

# Docker Installation
install_docker() {
    echo -e "${BLUE}📦 Installing Docker...${NC}"
    
    # Install prerequisites
    sudo apt-get update
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    check_status "Installing prerequisites"

    # Add Docker's official GPG key
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    check_status "Adding Docker GPG key"

    # Set up the repository
    echo \
        "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    check_status "Adding Docker repository"

    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    check_status "Installing Docker Engine"

    # Add user to docker group
    sudo usermod -aG docker $USER
    check_status "Adding user to docker group"
}

# Setup NVIDIA Container Toolkit
setup_nvidia_toolkit() {
    if [ "$NVIDIA_GPU" = true ]; then
        echo -e "${BLUE}🛠 Installing NVIDIA Container Toolkit...${NC}"
        
        # First, let's install the signing key
        curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

        # Add the repository for Ubuntu Noble (24.04)
        curl -fsSL https://nvidia.github.io/nvidia-docker/ubuntu24.04/nvidia-docker.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-docker.list

        # Update package listing
        sudo apt-get update

        # Install the NVIDIA Docker packages
        sudo apt-get install -y nvidia-container-toolkit nvidia-docker2

        # Configure the Docker daemon
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        check_status "Installing NVIDIA Container Toolkit"
    fi
}

# Create Container Networks
create_networks() {
    echo -e "${BLUE}🌐 Creating Docker networks...${NC}"
    docker network create gpu-net || true
    check_status "Creating Docker network"
}

# Create and Configure Base Containers
setup_containers() {
    echo -e "${BLUE}🐋 Creating Docker containers...${NC}"
    
    # Create data directories
    mkdir -p ${PWD}/cuda0-data ${PWD}/cuda1-data
    
    # CUDA Containers for K80
    if [ "$NVIDIA_GPU" = true ]; then
        for i in $(seq 0 $(($K80_DIES-1))); do
            echo "Creating CUDA container $i..."
            docker run -d \
                --name cuda-container-$i \
                --network gpu-net \
                --gpus "\"device=$i\"" \
                -v ${PWD}/cuda$i-data:/data \
                nvidia/cuda:11.8.0-base-ubuntu20.04
            check_status "Creating CUDA container $i"
        done
    fi
}

# Create Environment Check Script
create_check_script() {
    cat << 'EOF' | sudo tee /usr/local/bin/check-gpu-containers
#!/bin/bash
echo "Checking container status..."
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo -e "\nChecking GPU assignments..."
if command -v nvidia-smi &> /dev/null; then
    for i in $(seq 0 $(($(nvidia-smi -L | grep -c "Tesla K80")-1))); do
        echo "cuda-container-$i -> K80 (Die $i)"
        docker exec cuda-container-$i nvidia-smi 2>/dev/null || echo "CUDA container $i not responding"
    done
fi
EOF
    sudo chmod +x /usr/local/bin/check-gpu-containers
    check_status "Creating check script"
}

# Main Execution
main() {
    detect_gpus
    install_docker
    setup_nvidia_toolkit
    create_networks
    setup_containers
    create_check_script
    
    echo -e "${GREEN}✨ Setup Complete! ✨${NC}"
    echo -e "${BLUE}Please run 'check-gpu-containers' to verify installation.${NC}"
    
    # Refresh group membership
    exec sudo su -l $USER
}

main