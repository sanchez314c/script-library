#!/bin/bash
# Docker Environment Setup Script with NVIDIA CUDA and ROCm Support
# Version: 2.9.30
# Date: March 24, 2025
# Author: Grok 3 (xAI) for Jason
# Description: Sets up Docker with NVIDIA CUDA, ROCm PyTorch, and Ollama for RX580/K80

set -e
set -x

# Check if script is run with sudo
if [ "$(id -u)" -ne 0 ]; then
    echo "❌ This script must be run with sudo" >&2
    exit 1
fi

# Properly detect the real user even when run with sudo
REAL_USER="${SUDO_USER:-$USER}"
if [ "$REAL_USER" = "root" ] && [ -n "$SUDO_USER" ]; then
    REAL_USER="$SUDO_USER"
fi

LOG_FILE="/home/${REAL_USER}/docker_setup.log"
echo "🚀 Starting Docker Environment Setup..." | tee -a "$LOG_FILE"

clean_docker() {
    echo "🧹 Cleaning up Docker..." | tee -a "$LOG_FILE"
    if docker --version &>/dev/null; then
        echo "✅ Docker already installed—skipping cleanup" | tee -a "$LOG_FILE"
        return
    fi
    systemctl stop docker docker.socket containerd 2>/dev/null || true
    pkill -f dockerd 2>/dev/null || true
    apt-get purge -y docker.io docker-ce docker-ce-cli containerd docker-compose nvidia-container-toolkit || true
    rm -rf /var/lib/docker /etc/docker /run/docker.sock /var/run/docker.sock /home/${REAL_USER}/.docker
    systemctl daemon-reload
    gpasswd -d "${REAL_USER}" docker 2>/dev/null || true
    echo "✅ Docker cleaned" | tee -a "$LOG_FILE"
}

detect_gpu() {
    echo "🔍 Detecting GPUs..." | tee -a "$LOG_FILE"
    
    # Check for NVIDIA GPU
    if [ -x "/usr/bin/nvidia-smi" ] && nvidia-smi &>/dev/null; then
        NVIDIA_GPU=true
        echo "✅ NVIDIA GPU detected with working drivers" | tee -a "$LOG_FILE"
    else
        NVIDIA_GPU=false
        echo "⚠️ No NVIDIA GPU detected with working drivers" | tee -a "$LOG_FILE"
    fi
    
    # Check for ROCm using common installation paths and verify hardware
    if [ -x "/opt/rocm/bin/rocminfo" ] && /opt/rocm/bin/rocminfo &>/dev/null; then
        AMD_GPU=true
        echo "✅ ROCm installation detected with working drivers" | tee -a "$LOG_FILE"
    elif [ -x "/opt/rocm-6.3.4/bin/rocminfo" ] && /opt/rocm-6.3.4/bin/rocminfo &>/dev/null; then
        AMD_GPU=true
        echo "✅ ROCm 6.3.4 installation detected with working drivers" | tee -a "$LOG_FILE"
    elif [ -d "/opt/rocm" ]; then
        # ROCm is installed but we couldn't verify it's working
        AMD_GPU=true
        echo "⚠️ ROCm installation detected but couldn't verify functionality" | tee -a "$LOG_FILE"
    else
        AMD_GPU=false
        echo "⚠️ No ROCm installation detected" | tee -a "$LOG_FILE"
    fi
    
    echo "NVIDIA: $NVIDIA_GPU, AMD: $AMD_GPU" | tee -a "$LOG_FILE"
}

install_docker() {
    echo "🐳 Installing Docker..." | tee -a "$LOG_FILE"
    apt-get update || { echo "❌ Apt update failed" | tee -a "$LOG_FILE"; exit 1; }
    apt-get install -y docker.io containerd runc || { echo "❌ Docker install failed" | tee -a "$LOG_FILE"; exit 1; }
    
    echo "🐳 Installing Docker Compose..." | tee -a "$LOG_FILE"
    apt-get install -y docker-compose || { echo "❌ Docker Compose install failed" | tee -a "$LOG_FILE"; exit 1; }
    /usr/bin/docker-compose --version || { echo "⚠️ Docker Compose not in PATH yet—reboot may fix" | tee -a "$LOG_FILE"; }
    
    systemctl daemon-reload
    systemctl start containerd || { echo "❌ Containerd start failed" | tee -a "$LOG_FILE"; exit 1; }
    systemctl enable containerd
    sleep 2
    systemctl start docker || { echo "❌ Docker start failed, trying manual..." | tee -a "$LOG_FILE"; dockerd --debug > /tmp/dockerd-debug.log 2>&1 & sleep 5; pkill -f dockerd; systemctl start docker || { echo "❌ Docker still failed" | tee -a "$LOG_FILE"; cat /tmp/dockerd-debug.log | tee -a "$LOG_FILE"; exit 1; }; }
    systemctl enable docker
    usermod -aG docker "${REAL_USER}"
    systemctl reset-failed docker.service docker.socket containerd.service 2>/dev/null || true
    echo "✅ Docker and Docker Compose installed" | tee -a "$LOG_FILE"
}

configure_gpu_support() {
    echo "🔧 Configuring GPU support for Docker..." | tee -a "$LOG_FILE"

    # NVIDIA Container Toolkit
    if [ "$NVIDIA_GPU" = true ]; then
        echo "📦 Installing NVIDIA Container Toolkit..." | tee -a "$LOG_FILE"
        if ! dpkg -l | grep -q nvidia-container-toolkit; then
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            apt-get update
            apt-get install -y nvidia-container-toolkit || { echo "❌ NVIDIA Container Toolkit install failed" | tee -a "$LOG_FILE"; exit 1; }
            nvidia-ctk runtime configure --runtime=docker
        fi
        echo "✅ NVIDIA Container Toolkit configured" | tee -a "$LOG_FILE"
    fi

    # ROCm Docker Support
    if [ "$AMD_GPU" = true ]; then
        echo "🔧 Configuring ROCm support for Docker..." | tee -a "$LOG_FILE"
        
        # Create Docker daemon.json if it doesn't exist
        mkdir -p /etc/docker
        
        # Handle daemon.json properly - preserve existing content if present
        if [ ! -f "/etc/docker/daemon.json" ] || [ ! -s "/etc/docker/daemon.json" ]; then
            echo '{}' > /etc/docker/daemon.json
        fi
        
        echo "✅ ROCm will use device pass-through (--device=/dev/kfd --device=/dev/dri)" | tee -a "$LOG_FILE"
    fi

    if [ "$NVIDIA_GPU" = true ] || [ "$AMD_GPU" = true ]; then
        systemctl restart docker || { echo "❌ Docker restart failed" | tee -a "$LOG_FILE"; exit 1; }
        echo "✅ Docker restarted with GPU support" | tee -a "$LOG_FILE"
    fi
}

configure_docker_containers() {
    echo "🐳 Configuring Docker containers..." | tee -a "$LOG_FILE"
    if [ "$AMD_GPU" = true ]; then
        echo "📥 Setting up ROCm containers..." | tee -a "$LOG_FILE"
        # First make sure git is installed
        apt-get install -y git || { echo "❌ Git install failed" | tee -a "$LOG_FILE"; }
        
        # Create temporary directory manually
        mkdir -p /tmp/rocm-docker
        ROCM_DOCKER_DIR="/tmp/rocm-docker"
        echo "🔧 Working in directory: $ROCM_DOCKER_DIR" | tee -a "$LOG_FILE"
        
        # Create Dockerfile directly
        echo "📥 Creating ROCm PyTorch Dockerfile..." | tee -a "$LOG_FILE"
        
        cat > "$ROCM_DOCKER_DIR/Dockerfile" << 'EOL'
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    build-essential \
    git \
    python3 \
    python3-pip \
    python3-dev \
    cmake
# Install ROCm dependencies for PyTorch
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
# Add test utilities
RUN pip3 install numpy matplotlib pandas scikit-learn
WORKDIR /workspace
CMD ["/bin/bash"]
EOL

        # Create Ollama Dockerfile
        echo "📥 Creating Ollama Dockerfile..." | tee -a "$LOG_FILE"
        
        cat > "$ROCM_DOCKER_DIR/Dockerfile_rocm63_ollama" << 'EOL'
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    build-essential \
    git \
    python3 \
    python3-pip \
    curl
# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh
# Install web UI dependencies
RUN pip3 install fastapi uvicorn jinja2
WORKDIR /workspace
EXPOSE 11434
CMD ["ollama", "serve"]
EOL

        # Build Docker images
        echo "🔧 Building ROCm PyTorch container..." | tee -a "$LOG_FILE"
        docker build -f "$ROCM_DOCKER_DIR/Dockerfile" "$ROCM_DOCKER_DIR" -t rocm63_pt25:latest || { 
            echo "❌ ROCm PyTorch build failed" | tee -a "$LOG_FILE"
            exit 1
        }
        
        docker tag rocm63_pt25:latest rocm
        
        echo "🔧 Building Ollama/OpenWebUI container..." | tee -a "$LOG_FILE"
        docker build -f "$ROCM_DOCKER_DIR/Dockerfile_rocm63_ollama" "$ROCM_DOCKER_DIR" -t rocm63_ollama:latest || { 
            echo "❌ Ollama build failed" | tee -a "$LOG_FILE"
            exit 1
        }
        
        docker tag rocm63_ollama:latest ollama
        
        # Clean up
        rm -rf "$ROCM_DOCKER_DIR"
        echo "✅ ROCm containers built successfully" | tee -a "$LOG_FILE"
    fi
    
    if [ "$NVIDIA_GPU" = true ]; then
        echo "📥 Pulling NVIDIA CUDA containers..." | tee -a "$LOG_FILE"
        docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 && \
            docker tag nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 cuda0 && \
            docker tag nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 cuda1
        echo "✅ NVIDIA CUDA containers pulled" | tee -a "$LOG_FILE"
    fi
    
    # Add convenience alias for ROCm containers
    if [ "$AMD_GPU" = true ]; then
        BASHRC_FILE="/home/${REAL_USER}/.bashrc"
        if [ -f "$BASHRC_FILE" ] && ! grep -q "alias drun" "$BASHRC_FILE"; then
            DRUN_ALIAS="alias drun='docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G -v \$HOME/dockerx:/dockerx -w /dockerx'"
            su - "${REAL_USER}" -c "echo \"$DRUN_ALIAS\" >> $BASHRC_FILE"
            echo "✅ Added 'drun' alias for ROCm containers" | tee -a "$LOG_FILE"
        fi
    fi
    
    echo "✅ Docker containers configured" | tee -a "$LOG_FILE"
}

verify_docker() {
    echo "🔍 Verifying Docker setup..." | tee -a "$LOG_FILE"
    docker --version | tee -a "$LOG_FILE" || echo "❌ Docker verification failed" | tee -a "$LOG_FILE"
    docker-compose --version | tee -a "$LOG_FILE" || echo "❌ Docker Compose verification failed" | tee -a "$LOG_FILE"
    
    if [ "$NVIDIA_GPU" = true ]; then
        echo "Testing NVIDIA GPU access..." | tee -a "$LOG_FILE"
        docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 nvidia-smi | tee -a "$LOG_FILE" || echo "⚠️ NVIDIA GPU test failed" | tee -a "$LOG_FILE"
    fi
    
    if [ "$AMD_GPU" = true ]; then
        echo "Testing AMD GPU access..." | tee -a "$LOG_FILE"
        docker run --rm --device=/dev/kfd --device=/dev/dri --group-add=video ubuntu:20.04 ls -la /dev/dri | tee -a "$LOG_FILE" || echo "⚠️ AMD GPU test failed" | tee -a "$LOG_FILE"
    fi
    
    echo "✅ Docker verification complete" | tee -a "$LOG_FILE"
}

main() {
    clean_docker
    detect_gpu
    install_docker
    configure_gpu_support
    configure_docker_containers
    verify_docker
    echo "
✨ Docker Setup Complete!
- Docker: rocm (PyTorch), cuda0, cuda1, ollama (OpenWebUI)
- Docker Compose is now installed
- GPU Support: NVIDIA Container Toolkit ($( [ "$NVIDIA_GPU" = true ] && echo "enabled" || echo "disabled")), ROCm ($( [ "$AMD_GPU" = true ] && echo "enabled" || echo "disabled"))
- Alias: drun (for ROCm container)
- Note: Reboot may be needed for Docker group
- Log: $LOG_FILE

IMPORTANT: To use docker-compose or Ollama right away, run:
  su - ${REAL_USER}
Or simply reboot your system.
" | tee -a "$LOG_FILE"
}

main