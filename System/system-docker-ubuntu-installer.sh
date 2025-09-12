#!/bin/bash
# Docker Environment Setup Script with NVIDIA CUDA and ROCm Support
# Version: 2.9.29
# Date: March 24, 2025
# Author: Grok 3 (xAI) for Jason
# Description: Sets up Docker with NVIDIA CUDA, ROCm PyTorch, and Ollama for RX580/K80

set -e
set -x

LOG_FILE="/home/${SUDO_USER:-$USER}/docker_setup.log"
echo "🚀 Starting Docker Environment Setup..." | tee -a "$LOG_FILE"

clean_docker() {
    echo "🧹 Cleaning up Docker..." | tee -a "$LOG_FILE"
    if docker --version &>/dev/null; then
        echo "✅ Docker already installed—skipping cleanup" | tee -a "$LOG_FILE"
        return
    fi
    sudo systemctl stop docker docker.socket containerd 2>/dev/null || true
    sudo pkill -f dockerd 2>/dev/null || true
    sudo apt-get purge -y docker.io docker-ce docker-ce-cli containerd docker-compose nvidia-container-toolkit || true
    sudo rm -rf /var/lib/docker /etc/docker /run/docker.sock /var/run/docker.sock /home/${SUDO_USER:-$USER}/.docker
    sudo systemctl daemon-reload
    sudo gpasswd -d "${SUDO_USER:-$USER}" docker 2>/dev/null || true
    echo "✅ Docker cleaned" | tee -a "$LOG_FILE"
}

detect_gpu() {
    echo "🔍 Detecting GPUs..." | tee -a "$LOG_FILE"
    NVIDIA_GPU=$([ -x "/usr/bin/nvidia-smi" ] && echo true || echo false)
    
    # Check for ROCm using common installation paths
    if [ -x "/opt/rocm/bin/rocminfo" ] || [ -x "/opt/rocm-6.3.4/bin/rocminfo" ] || [ -d "/opt/rocm" ]; then
        AMD_GPU=true
        echo "✅ ROCm installation detected" | tee -a "$LOG_FILE"
    else
        AMD_GPU=false
        echo "⚠️ No ROCm installation detected" | tee -a "$LOG_FILE"
    fi
    
    echo "NVIDIA: $NVIDIA_GPU, AMD: $AMD_GPU" | tee -a "$LOG_FILE"
}

install_docker() {
    echo "🐳 Installing Docker..." | tee -a "$LOG_FILE"
    sudo apt-get update || { echo "❌ Apt update failed" | tee -a "$LOG_FILE"; exit 1; }
    sudo apt-get install -y docker.io containerd runc || { echo "❌ Docker install failed" | tee -a "$LOG_FILE"; exit 1; }
    
    echo "🐳 Installing Docker Compose..." | tee -a "$LOG_FILE"
    sudo apt-get install -y docker-compose || { echo "❌ Docker Compose install failed" | tee -a "$LOG_FILE"; exit 1; }
    /usr/bin/docker-compose --version || { echo "⚠️ Docker Compose not in PATH yet—reboot may fix" | tee -a "$LOG_FILE"; }
    
    sudo systemctl daemon-reload
    sudo systemctl start containerd || { echo "❌ Containerd start failed" | tee -a "$LOG_FILE"; exit 1; }
    sudo systemctl enable containerd
    sleep 2
    sudo systemctl start docker || { echo "❌ Docker start failed, trying manual..." | tee -a "$LOG_FILE"; sudo dockerd --debug > /tmp/dockerd-debug.log 2>&1 & sleep 5; sudo pkill -f dockerd; sudo systemctl start docker || { echo "❌ Docker still failed" | tee -a "$LOG_FILE"; cat /tmp/dockerd-debug.log | tee -a "$LOG_FILE"; exit 1; }; }
    sudo systemctl enable docker
    sudo usermod -aG docker "${SUDO_USER:-$USER}"
    sudo systemctl reset-failed docker.service docker.socket containerd.service 2>/dev/null || true
    echo "✅ Docker and Docker Compose installed" | tee -a "$LOG_FILE"
}

configure_gpu_support() {
    echo "🔧 Configuring GPU support for Docker..." | tee -a "$LOG_FILE"

    # NVIDIA Container Toolkit
    if [ "$NVIDIA_GPU" = true ]; then
        echo "📦 Installing NVIDIA Container Toolkit..." | tee -a "$LOG_FILE"
        if ! dpkg -l | grep -q nvidia-container-toolkit; then
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            sudo apt-get update
            sudo apt-get install -y nvidia-container-toolkit || { echo "❌ NVIDIA Container Toolkit install failed" | tee -a "$LOG_FILE"; exit 1; }
            sudo nvidia-ctk runtime configure --runtime=docker
        fi
        echo "✅ NVIDIA Container Toolkit configured" | tee -a "$LOG_FILE"
    fi

    # ROCm Docker Support
    if [ "$AMD_GPU" = true ]; then
        echo "🔧 Configuring ROCm support for Docker..." | tee -a "$LOG_FILE"
        
        # We'll use device passthrough for ROCm, no special daemon config needed
        sudo mkdir -p /etc/docker
        if [ ! -f "/etc/docker/daemon.json" ] || [ ! -s "/etc/docker/daemon.json" ]; then
            echo '{}' | sudo tee /etc/docker/daemon.json
        fi
        
        echo "✅ ROCm will use device pass-through (--device=/dev/kfd --device=/dev/dri)" | tee -a "$LOG_FILE"
    fi

    if [ "$NVIDIA_GPU" = true ] || [ "$AMD_GPU" = true ]; then
        sudo systemctl restart docker || { echo "❌ Docker restart failed" | tee -a "$LOG_FILE"; exit 1; }
        echo "✅ Docker restarted with GPU support" | tee -a "$LOG_FILE"
    fi
}

configure_docker_containers() {
    echo "🐳 Configuring Docker containers..." | tee -a "$LOG_FILE"
    if [ "$AMD_GPU" = true ]; then
        echo "📥 Cloning robertrosenbusch/gfx803_rocm for ROCm PyTorch..." | tee -a "$LOG_FILE"
        # First make sure git is installed
        sudo apt-get install -y git || { echo "❌ Git install failed" | tee -a "$LOG_FILE"; }
        
        # Remove the directory if it already exists
        if [ -d "/tmp/gfx803_rocm" ]; then
            echo "🧹 Removing existing /tmp/gfx803_rocm directory..." | tee -a "$LOG_FILE"
            sudo rm -rf /tmp/gfx803_rocm
        fi
        
        # Clone the repository with the correct URL
        sudo git clone https://github.com/robertrosenbusch/gfx803_rocm.git /tmp/gfx803_rocm || { echo "❌ Git clone failed" | tee -a "$LOG_FILE"; exit 1; }
        cd /tmp/gfx803_rocm
        
        # Check if Dockerfile is a symlink pointing to a non-existent location
        if [ -L "Dockerfile" ]; then
            echo "🔧 Fixing symlinked Dockerfile..." | tee -a "$LOG_FILE"
            # Get the target of the symlink
            target=$(readlink "Dockerfile")
            
            # If the target doesn't exist, create a new Dockerfile
            if [ ! -f "$target" ]; then
                echo "Creating new Dockerfile from base image..." | tee -a "$LOG_FILE"
                echo 'FROM ubuntu:20.04

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
CMD ["/bin/bash"]' | sudo tee Dockerfile
            fi
        fi
        
        # Now build the Docker image
        sudo docker build -f Dockerfile . -t rocm63_pt25:latest || { echo "❌ ROCm PyTorch build failed" | tee -a "$LOG_FILE"; exit 1; }
        sudo docker tag rocm63_pt25:latest rocm
        echo "📥 Building Ollama/OpenWebUI container..." | tee -a "$LOG_FILE"
        sudo docker build -f Dockerfile_rocm63_ollama . -t rocm63_ollama:latest || { echo "❌ Ollama build failed" | tee -a "$LOG_FILE"; exit 1; }
        sudo docker tag rocm63_ollama:latest ollama
        cd -
        sudo rm -rf /tmp/gfx803_rocmm/Dockerfile ./Dockerfile
        sudo rm -rf /tmp/gfx803_rocm

        echo "📥 Building ROCm PyTorch container..." | tee -a "$LOG_FILE"
        cd /tmp/rocm_docker
        sudo docker build -t rocm63_pt25:latest . || { echo "❌ ROCm PyTorch build failed, but continuing..." | tee -a "$LOG_FILE"; }
        sudo docker tag rocm63_pt25:latest rocm || true
        
        echo "📥 Building Ollama/OpenWebUI container..." | tee -a "$LOG_FILE"
        sudo docker build -f Dockerfile_rocm63_ollama . -t rocm63_ollama:latest || { echo "❌ Ollama build failed, but continuing..." | tee -a "$LOG_FILE"; }
        sudo docker tag rocm63_ollama:latest ollama || true
        
        cd -
        sudo rm -rf /tmp/rocm_docker
    fi
    
    if [ "$NVIDIA_GPU" = true ]; then
        sudo docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 && \
            sudo docker tag nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 cuda0 && \
            sudo docker tag nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 cuda1
    fi
    
    # Add convenience alias for ROCm containers
    if [ "$AMD_GPU" = true ] && ! sudo -u "${SUDO_USER:-$USER}" grep -q "alias drun" "/home/${SUDO_USER:-$USER}/.bashrc"; then
        sudo -u "${SUDO_USER:-$USER}" bash -c "echo \"alias drun='docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G -v \$HOME/dockerx:/dockerx -w /dockerx'\" >> /home/${SUDO_USER:-$USER}/.bashrc"
        echo "✅ Added 'drun' alias for ROCm containers" | tee -a "$LOG_FILE"
    fi
    
    echo "✅ Docker containers configured" | tee -a "$LOG_FILE"
}

verify_docker() {
    echo "🔍 Verifying Docker setup..." | tee -a "$LOG_FILE"
    docker --version | tee -a "$LOG_FILE"
    /usr/bin/docker-compose --version | tee -a "$LOG_FILE" || echo "⚠️ Docker Compose not in PATH yet—reboot may fix" | tee -a "$LOG_FILE"
    
    if [ "$NVIDIA_GPU" = true ]; then
        echo "Testing NVIDIA GPU access..." | tee -a "$LOG_FILE"
        docker run --rm --gpus all cuda0 nvidia-smi | tee -a "$LOG_FILE" || echo "⚠️ NVIDIA GPU test failed, but continuing" | tee -a "$LOG_FILE"
    else
        echo "⚠️ NVIDIA GPU test skipped (no NVIDIA GPU detected)" | tee -a "$LOG_FILE"
    fi
    
    if [ "$AMD_GPU" = true ]; then
        echo "Testing AMD GPU access..." | tee -a "$LOG_FILE"
        docker run --rm --device=/dev/kfd --device=/dev/dri --group-add=video ubuntu:20.04 ls -la /dev/dri | tee -a "$LOG_FILE" || echo "⚠️ AMD GPU test failed, but continuing" | tee -a "$LOG_FILE"
    else
        echo "⚠️ AMD GPU test skipped (no AMD GPU detected)" | tee -a "$LOG_FILE"
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
  su - ${SUDO_USER:-$USER}
Or simply reboot your system.
" | tee -a "$LOG_FILE"
}

main
