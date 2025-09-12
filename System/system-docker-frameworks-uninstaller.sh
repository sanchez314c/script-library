#!/bin/bash

# Docker Environment Uninstaller Script
# Version: 2.9.24 - Built by Grok 3 (xAI) for Jason
# Date: March 2, 2025
set -e
set -x

LOG_FILE="/home/$USER/docker_uninstall.log"
echo "🗑️ Starting Docker Environment Uninstallation..." | tee -a "$LOG_FILE"

stop_docker() {
    echo "🛑 Stopping Docker services..." | tee -a "$LOG_FILE"
    sudo systemctl stop docker docker.socket containerd 2>/dev/null || true
    sudo pkill -f dockerd 2>/dev/null || true
    echo "✅ Docker services stopped" | tee -a "$LOG_FILE"
}

remove_docker_images() {
    echo "🧹 Removing Docker images..." | tee -a "$LOG_FILE"
    for img in darkpool-rocm darkpool-cuda0 darkpool-cuda1 rocm/pytorch:latest nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04; do
        sudo docker rmi -f "$img" 2>/dev/null || echo "⚠️ Image $img not found or already removed" | tee -a "$LOG_FILE"
    done
    echo "✅ Docker images removed" | tee -a "$LOG_FILE"
}

uninstall_docker() {
    echo "🗑️ Uninstalling Docker and Docker Compose..." | tee -a "$LOG_FILE"
    sudo apt-get purge -y docker.io docker-ce docker-ce-cli containerd docker-compose nvidia-container-toolkit 2>/dev/null || true
    sudo rm -rf /var/lib/docker /etc/docker /run/docker.sock /var/run/docker.sock ~/.docker
    sudo systemctl daemon-reload
    sudo gpasswd -d "$USER" docker 2>/dev/null || true
    echo "✅ Docker and Docker Compose uninstalled" | tee -a "$LOG_FILE"
}

clean_bashrc() {
    echo "🧹 Cleaning .bashrc..." | tee -a "$LOG_FILE"
    # Create a backup
    cp "$HOME/.bashrc" "$HOME/.bashrc.bak.$(date +%Y%m%d_%H%M%S)"
    # Remove Docker alias
    sed -i '/alias drun=/d' "$HOME/.bashrc"
    echo "✅ .bashrc cleaned (backup saved as .bashrc.bak.*)" | tee -a "$LOG_FILE"
}

verify_removal() {
    echo "🔍 Verifying removal..." | tee -a "$LOG_FILE"
    command -v docker >/dev/null 2>&1 && echo "⚠️ Docker command still available" | tee -a "$LOG_FILE" || echo "✅ Docker command removed" | tee -a "$LOG_FILE"
    command -v docker-compose >/dev/null 2>&1 && echo "⚠️ Docker Compose command still available" | tee -a "$LOG_FILE" || echo "✅ Docker Compose command removed" | tee -a "$LOG_FILE"
    [ -d "/var/lib/docker" ] && echo "⚠️ Docker data directory still exists" | tee -a "$LOG_FILE" || echo "✅ Docker data directory gone" | tee -a "$LOG_FILE"
    sudo docker images -q | grep -q . && echo "⚠️ Docker images still present" | tee -a "$LOG_FILE" || echo "✅ No Docker images found" | tee -a "$LOG_FILE"
    echo "✅ Verification complete" | tee -a "$LOG_FILE"
}

main() {
    stop_docker
    remove_docker_images
    uninstall_docker
    clean_bashrc
    verify_removal
    echo "
🗑️ Docker Uninstallation Complete!
- Images darkpool-rocm, darkpool-cuda0/1 removed
- Docker and Docker Compose uninstalled
- .bashrc cleaned (backup in ~/.bashrc.bak.*)
- Log: $LOG_FILE
- Note: Reboot may be needed to fully clear Docker group
" | tee -a "$LOG_FILE"
}

main
