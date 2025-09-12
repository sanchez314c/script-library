#!/bin/bash

# Docker Environment Setup Script
# Version: 2.9.24 - Built by Grok 3 (xAI) for Jason
# Date: March 2, 2025
set -e
set -x

LOG_FILE="/home/${SUDO_USER:-$USER}/docker_setup.log"
echo "🚀 Starting Docker Environment Setup..." | tee -a "$LOG_FILE"

clean_docker() {
    echo "🧹 Cleaning up Docker..." | tee -a "$LOG_FILE"
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
    NVIDIA_GPU=$(command -v nvidia-smi >/dev/null 2>&1 && echo true || echo false)
    AMD_GPU=$([ -x "/opt/rocm/bin/rocminfo" ] && echo true || echo false)
    echo "NVIDIA: $NVIDIA_GPU, AMD: $AMD_GPU" | tee -a "$LOG_FILE"
}

install_docker() {
    echo "🐳 Installing Docker..." | tee -a "$LOG_FILE"
    sudo apt-get update || { echo "❌ Apt update failed" | tee -a "$LOG_FILE"; exit 1; }
    sudo apt-get install -y docker.io containerd runc || { echo "❌ Docker install failed" | tee -a "$LOG_FILE"; exit 1; }
    
    echo "🐳 Installing Docker Compose..." | tee -a "$LOG_FILE"
    sudo apt-get install -y docker-compose || { echo "❌ Docker Compose install failed" | tee -a "$LOG_FILE"; exit 1; }
    docker-compose --version || { echo "⚠️ Docker Compose version check failed, but continuing..." | tee -a "$LOG_FILE"; }
    
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

configure_docker_containers() {
    echo "🐳 Configuring Docker containers..." | tee -a "$LOG_FILE"
    [ "$AMD_GPU" = true ] && sudo docker pull rocm/pytorch:latest && sudo docker tag rocm/pytorch:latest darkpool-rocm
    [ "$NVIDIA_GPU" = true ] && sudo docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 && sudo docker tag nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 darkpool-cuda0 && sudo docker tag nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 darkpool-cuda1
    [ "$AMD_GPU" = true ] && ! sudo -u "${SUDO_USER:-$USER}" grep -q "alias drun" "/home/${SUDO_USER:-$USER}/.bashrc" && sudo -u "${SUDO_USER:-$USER}" bash -c "echo \"alias drun='docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G -v \$HOME/dockerx:/dockerx -w /dockerx'\" >> /home/${SUDO_USER:-$USER}/.bashrc"
    echo "✅ Docker containers configured" | tee -a "$LOG_FILE"
}

verify_docker() {
    echo "🔍 Verifying Docker setup..." | tee -a "$LOG_FILE"
    docker --version | tee -a "$LOG_FILE"
    docker-compose --version | tee -a "$LOG_FILE" || echo "⚠️ Docker Compose verification failed. You may need to log out and back in." | tee -a "$LOG_FILE"
    [ "$NVIDIA_GPU" = true ] && docker run --rm nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 nvidia-smi | tee -a "$LOG_FILE"
    echo "✅ Docker verification complete" | tee -a "$LOG_FILE"
}

main() {
    clean_docker
    detect_gpu
    install_docker
    configure_docker_containers
    verify_docker
    echo "
✨ Docker Setup Complete!
- Docker: darkpool-rocm, darkpool-cuda0/1
- Docker Compose is now installed
- Alias: drun (for ROCm container)
- Note: Reboot may be needed for Docker group
- Log: $LOG_FILE

IMPORTANT: To use docker-compose right away, run: 
  su - ${SUDO_USER:-$USER}
Or simply reboot your system.
" | tee -a "$LOG_FILE"
}

main