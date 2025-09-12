#!/bin/bash

# ROCm Docker Environment Setup Script for gfx803
# Improved and Corrected Version

LOG_FILE="/home/${SUDO_USER:-$USER}/docker_setup.log"
echo "🚀 Starting ROCm Docker Environment Setup..." | tee -a "$LOG_FILE"

set -euxo pipefail

# Check for gfx803 GPU explicitly
detect_gpu() {
    if ! /opt/rocm/bin/rocminfo | grep -q 'gfx803'; then
        echo "❌ gfx803 GPU not detected. Exiting." | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "✅ gfx803 GPU detected." | tee -a "$LOG_FILE"
}

# Install Docker properly
install_docker() {
    sudo apt update
    sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io
    sudo usermod -aG docker "${SUDO_USER:-$USER}"
    sudo systemctl enable --now docker
}

# Clone gfx803_rocm repo
clone_gfx803_repo() {
    local repo_path="/home/${SUDO_USER:-$USER}/gfx803_rocm"
    if [ ! -d "$repo_path" ]; then
        git clone https://github.com/robertrosenbusch/gfx803_rocm.git "$repo_path"
        sudo chown -R "${SUDO_USER:-$USER}:${SUDO_USER:-$USER}" "$repo_path"
    fi
}

# Build Docker Images
build_docker_images() {
    local repo_path="/home/${SUDO_USER:-$USER}/gfx803_rocm"

    cd "$repo_path"
    sudo docker build -t rocm63_pt25:latest -f Dockerfile .
    sudo docker build -t rocm63_ollama:latest -f Dockerfile_rocm63_ollama .
}

# Setup ComfyUI container
setup_comfyui_container() {
    local comfyui_path="/home/${SUDO_USER:-$USER}/ComfyUI_Checkpoints"
    sudo mkdir -p "$comfyui_path"
    sudo chown -R "${SUDO_USER:-$USER}:${SUDO_USER:-$USER}" "$comfyui_path"

    sudo docker rm -f rocm63_pt25 || true
    sudo docker run -d \
        --device=/dev/kfd --device=/dev/dri --group-add=video \
        --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        -p 8188:8188 \
        -v "$comfyui_path":/comfy/ \
        --name rocm63_pt25 rocm63_pt25:latest sleep infinity

    sudo docker exec rocm63_pt25 bash -c '
        git clone https://github.com/comfyanonymous/ComfyUI.git /comfy/ComfyUI
        pip install -r /comfy/ComfyUI/requirements.txt
    '
}

# Setup Ollama container
setup_ollama_container() {
    sudo docker rm -f rocm63_ollama || true
    sudo docker run -d \
        --device=/dev/kfd --device=/dev/dri --group-add=video \
        --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        -p 11434:11434 \
        --name rocm63_ollama rocm63_ollama:latest sleep infinity

    sudo docker exec rocm63_ollama bash -c '
        curl -fsSL https://ollama.com/install.sh | sh
        ollama serve & sleep 5
        ollama pull llama3.2:1b
    '
}

# Add helpful Docker aliases
add_aliases() {
    echo "alias dps='docker ps -a'" >> ~/.bashrc
    echo "alias dexec='docker exec -it'" >> ~/.bashrc
    echo "alias start-comfy='docker start rocm63_pt25 && docker exec -it rocm63_pt25 python /comfy/ComfyUI/main.py --listen 0.0.0.0 --lowvram'" >> ~/.bashrc
    echo "alias start-ollama='docker start rocm63_ollama && docker exec -it rocm63_ollama ollama serve'" >> ~/.bashrc
}

# Main script execution
main() {
    detect_gpu
    install_docker
    clone_gfx803_repo
    build_docker_images
    setup_comfyui_container
    setup_ollama_container
    add_aliases

    echo "✅ ROCm Docker setup complete. Please log out and log in again to finalize Docker permissions." | tee -a "$LOG_FILE"
}

main