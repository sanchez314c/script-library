#!/bin/bash

# Unified AI/ML Environment Setup Script
# Version: 2.9.3 - Built by Cortana (via Grok 3, xAI) for Jason
# Date: February 25, 2025 - Fixed Conda PATH and aliases, bulletproof
set -e  # Exit on error
set -x  # Verbose mode: print commands as they run

echo "🚀 Starting Unified AI/ML Environment Setup on Ubuntu 24.04.1..."

clean_docker() {
    echo "🧹 Cleaning up existing Docker installations..."
    if systemctl is-active docker >/dev/null 2>&1; then
        sudo systemctl stop docker || true
    fi
    sudo apt-get purge -y docker.io docker-ce docker-ce-cli containerd docker-compose nvidia-container-toolkit || echo "⚠️ No Docker packages to remove"
    sudo rm -rf /var/lib/docker /etc/docker /etc/systemd/system/docker.service.d /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg /etc/apt/sources.list.d/nvidia-container-toolkit.list /run/docker.sock /var/run/docker.sock ~/.docker
    sudo systemctl daemon-reload
    if groups "$USER" | grep -q docker; then
        sudo gpasswd -d "$USER" docker
    fi
    echo "✅ Docker cleanup complete"
}

detect_gpu() {
    echo "🔍 Detecting GPU hardware..."
    if [ -x "/usr/local/cuda-11.4/bin/nvidia-smi" ]; then
        echo "✅ NVIDIA GPU detected"
        NVIDIA_GPU=true
    else
        echo "⚠️ NVIDIA GPU not detected (nvidia-smi missing)"
        NVIDIA_GPU=false
    fi
    
    if [ -x "/opt/rocm-6.3.3/bin/rocminfo" ]; then
        echo "✅ AMD GPU detected"
        AMD_GPU=true
    else
        echo "⚠️ AMD GPU not detected (rocminfo missing)"
        AMD_GPU=false
    fi
}

install_conda() {
    echo "📦 Installing Miniconda..."
    [ -d "$HOME/miniconda3" ] && { echo "🧹 Removing existing Conda..."; sudo rm -rf "$HOME/miniconda3" || true; }
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh || { echo "❌ Failed to download Miniconda"; exit 1; }
    bash miniconda.sh -b -p "$HOME/miniconda3" || { echo "❌ Miniconda install failed"; exit 1; }
    rm miniconda.sh
    sudo chown -R "$USER:$USER" "$HOME/miniconda3"
    # Force PATH in script and ensure .bashrc has it
    export PATH="$HOME/miniconda3/bin:$PATH"
    if ! grep -q "$HOME/miniconda3/bin" "$HOME/.bashrc"; then
        echo "# >>> conda initialize >>>" >> "$HOME/.bashrc"
        echo "export PATH=\"$HOME/miniconda3/bin:\$PATH\"" >> "$HOME/.bashrc"
        echo "# <<< conda initialize <<<" >> "$HOME/.bashrc"
        "$HOME/miniconda3/bin/conda" init bash || { echo "❌ Conda init failed"; exit 1; }
    fi
    # Ensure .bash_profile sources .bashrc for login shells
    if [ -f "$HOME/.bash_profile" ] && ! grep -q ".bashrc" "$HOME/.bash_profile"; then
        echo "[ -f ~/.bashrc ] && . ~/.bashrc" >> "$HOME/.bash_profile"
    elif [ ! -f "$HOME/.bash_profile" ]; then
        echo "[ -f ~/.bashrc ] && . ~/.bashrc" >> "$HOME/.bash_profile"
    fi
    # Source to apply immediately
    source "$HOME/.bashrc"
    "$HOME/miniconda3/bin/conda" config --set auto_activate_base false || { echo "❌ Conda config failed"; exit 1; }
    "$HOME/miniconda3/bin/conda" update -n base -c defaults conda -y || { echo "❌ Conda update failed"; exit 1; }
    command -v conda >/dev/null || { echo "❌ Conda not in PATH after init"; exit 1; }
    echo "✅ Miniconda installed"
}

setup_conda_env() {
    local env_name=$1
    echo "🌱 Creating environment: $env_name..."
    "$HOME/miniconda3/bin/conda" create -y -n "$env_name" python=3.10 || { echo "❌ Failed to create env $env_name"; exit 1; }
}

configure_base_env() {
    echo "🌱 Configuring darklake base environment..."
    setup_conda_env "darklake"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate darklake || { echo "❌ Failed to activate darklake"; exit 1; }
    "$HOME/miniconda3/bin/pip" install numpy pandas scipy scikit-learn matplotlib seaborn jupyter ipython tqdm requests || { echo "❌ Base env pip install failed"; exit 1; }
    if [ "$AMD_GPU" = true ]; then
        "$HOME/miniconda3/bin/pip" install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2 -v > torch_install.log 2>&1 || { echo "❌ Torch install failed for AMD, check torch_install.log"; exit 1; }
    fi
    if [ "$NVIDIA_GPU" = true ]; then
        "$HOME/miniconda3/bin/pip" install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu114 -v > torch_install.log 2>&1 || { echo "❌ Torch install failed for NVIDIA, check torch_install.log"; exit 1; }
    fi
    "$HOME/miniconda3/bin/pip" install transformers datasets accelerate lightning bitsandbytes deepspeed diffusers || { echo "❌ Additional pip install failed"; exit 1; }
    conda deactivate
    echo "✅ darklake configured"
}

configure_rocm_env() {
    if [ "$AMD_GPU" = true ]; then
        echo "🌱 Configuring darkpool-rocm environment..."
        setup_conda_env "darkpool-rocm"
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate darkpool-rocm || { echo "❌ Failed to activate darkpool-rocm"; exit 1; }
        for attempt in {1..3}; do
            if "$HOME/miniconda3/envs/darkpool-rocm/bin/pip" install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2 -v > torch_install_rocm.log 2>&1; then
                break
            fi
            echo "⚠️ Torch install attempt $attempt failed, retrying..."
            sleep 2
        done
        if ! "$HOME/miniconda3/envs/darkpool-rocm/bin/python3" -c "import torch" 2>/dev/null; then
            echo "❌ Torch install failed after retries, check torch_install_rocm.log"
            exit 1
        fi
        "$HOME/miniconda3/envs/darkpool-rocm/bin/pip" install transformers datasets || { echo "❌ Transformers/datasets install failed"; exit 1; }
        mkdir -p "$HOME/miniconda3/envs/darkpool-rocm/etc/conda/activate.d"
        cat > "$HOME/miniconda3/envs/darkpool-rocm/etc/conda/activate.d/rocm_vars.sh" << EOF
#!/bin/bash
export PATH="/opt/rocm-6.3.3/bin:\$PATH"
export LD_LIBRARY_PATH="/opt/rocm-6.3.3/lib:/opt/rocm-6.3.3/lib64:\$LD_LIBRARY_PATH"
export ROCM_PATH="/opt/rocm-6.3.3"
export HIP_PLATFORM=amd
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export ROC_ENABLE_PRE_VEGA=1
export PYTORCH_ROCM_ARCH=gfx803
export HIP_VISIBLE_DEVICES=0
EOF
        chmod +x "$HOME/miniconda3/envs/darkpool-rocm/etc/conda/activate.d/rocm_vars.sh"
        conda deactivate
        echo "✅ darkpool-rocm configured"
    fi
}

configure_cuda_envs() {
    if [ "$NVIDIA_GPU" = true ]; then
        echo "🌱 Configuring darkpool-cuda0 environment..."
        setup_conda_env "darkpool-cuda0"
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate darkpool-cuda0 || { echo "❌ Failed to activate darkpool-cuda0"; exit 1; }
        "$HOME/miniconda3/envs/darkpool-cuda0/bin/pip" install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu114 -v > torch_install_cuda0.log 2>&1 || { echo "❌ Torch install failed for cuda0, check torch_install_cuda0.log"; exit 1; }
        "$HOME/miniconda3/envs/darkpool-cuda0/bin/pip" install transformers datasets || { echo "❌ Transformers/datasets install failed"; exit 1; }
        mkdir -p "$HOME/miniconda3/envs/darkpool-cuda0/etc/conda/activate.d"
        cat > "$HOME/miniconda3/envs/darkpool-cuda0/etc/conda/activate.d/cuda_vars.sh" << EOF
#!/bin/bash
export PATH="/usr/local/cuda-11.4/bin:\$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:\$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda-11.4"
export CUDA_PATH="/usr/local/cuda-11.4"
export CUDA_VERSION=11.4
export CUDA_VISIBLE_DEVICES=0
EOF
        chmod +x "$HOME/miniconda3/envs/darkpool-cuda0/etc/conda/activate.d/cuda_vars.sh"
        conda deactivate

        echo "🌱 Configuring darkpool-cuda1 environment..."
        setup_conda_env "darkpool-cuda1"
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate darkpool-cuda1 || { echo "❌ Failed to activate darkpool-cuda1"; exit 1; }
        "$HOME/miniconda3/envs/darkpool-cuda1/bin/pip" install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu114 -v > torch_install_cuda1.log 2>&1 || { echo "❌ Torch install failed for cuda1, check torch_install_cuda1.log"; exit 1; }
        "$HOME/miniconda3/envs/darkpool-cuda1/bin/pip" install transformers datasets || { echo "❌ Transformers/datasets install failed"; exit 1; }
        mkdir -p "$HOME/miniconda3/envs/darkpool-cuda1/etc/conda/activate.d"
        cat > "$HOME/miniconda3/envs/darkpool-cuda1/etc/conda/activate.d/cuda_vars.sh" << EOF
#!/bin/bash
export PATH="/usr/local/cuda-11.4/bin:\$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:\$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda-11.4"
export CUDA_PATH="/usr/local/cuda-11.4"
export CUDA_VERSION=11.4
export CUDA_VISIBLE_DEVICES=1
EOF
        chmod +x "$HOME/miniconda3/envs/darkpool-cuda1/etc/conda/activate.d/cuda_vars.sh"
        conda deactivate
        echo "✅ darkpool-cuda0/1 configured"
    fi
}

install_docker() {
    echo "🐳 Installing Docker essentials..."
    sudo apt-get update || { echo "❌ Apt update failed"; exit 1; }
    sudo apt-get install -y docker.io containerd docker-compose || { echo "❌ Docker essentials install failed"; exit 1; }
    sudo systemctl daemon-reload
    echo "🔧 Starting containerd..."
    sudo systemctl start containerd || { echo "❌ Containerd start failed"; exit 1; }
    sleep 2
    sudo usermod -aG docker "$USER"
    echo "🔧 Starting Docker service..."
    for i in {1..3}; do
        if sudo systemctl start docker; then
            break
        fi
        echo "⚠️ Docker start attempt $i failed, retrying..."
        sudo systemctl stop docker || true
        sudo rm -f /run/docker.sock /var/run/docker.sock
        sudo systemctl reset-failed docker.service
        sleep 2
    done
    if ! systemctl is-active docker >/dev/null 2>&1; then
        echo "❌ Docker failed to start via systemd, trying manual start..."
        sudo dockerd --debug > /tmp/dockerd-debug.log 2>&1 &
        sleep 5
        if ! pgrep dockerd >/dev/null; then
            echo "❌ Manual Docker start failed, check /tmp/dockerd-debug.log"
            exit 1
        fi
    fi
    sudo systemctl enable docker || { echo "❌ Docker enable failed"; exit 1; }
    echo "✅ Docker essentials installed and running"
}

configure_docker_containers() {
    echo "🐳 Configuring Docker containers..."
    if [ "$AMD_GPU" = true ]; then
        echo "🔧 Pulling rocm/pytorch for darkpool-rocm..."
        sudo docker pull rocm/pytorch:latest || { echo "❌ Failed to pull rocm/pytorch"; exit 1; }
        sudo docker tag rocm/pytorch:latest darkpool-rocm || { echo "❌ Failed to tag darkpool-rocm"; exit 1; }
    fi
    if [ "$NVIDIA_GPU" = true ]; then
        echo "🔧 Pulling nvidia/cuda for darkpool-cuda0/1..."
        sudo docker pull nvidia/cuda:11.4.0-runtime-ubuntu24.04 || { echo "❌ Failed to pull nvidia/cuda"; exit 1; }
        sudo docker tag nvidia/cuda:11.4.0-runtime-ubuntu24.04 darkpool-cuda0 || { echo "❌ Failed to tag darkpool-cuda0"; exit 1; }
        sudo docker tag nvidia/cuda:11.4.0-runtime-ubuntu24.04 darkpool-cuda1 || { echo "❌ Failed to tag darkpool-cuda1"; exit 1; }
    fi
    if [ "$AMD_GPU" = true ] && ! grep -q "alias drun" "$HOME/.bashrc"; then
        echo "alias drun='docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G -v \$HOME/dockerx:/dockerx -w /dockerx'" >> "$HOME/.bashrc"
    fi
    echo "✅ Docker containers configured"
}

verify_envs() {
    echo "🔍 Verifying environments..."
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate darklake || { echo "❌ Failed to activate darklake"; exit 1; }
    echo "darklake:"; "$HOME/miniconda3/envs/darklake/bin/python3" -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, ROCm: {hasattr(torch, \"hip\") and torch.hip.is_available()}')" || { echo "❌ darklake failed"; exit 1; }
    conda deactivate
    
    if [ "$AMD_GPU" = true ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate darkpool-rocm || { echo "❌ Failed to activate darkpool-rocm"; exit 1; }
        echo "darkpool-rocm:"; /opt/rocm-6.3.3/bin/rocminfo | grep -A 5 "Name:.*gfx803" || echo "❌ rocminfo failed"; /opt/rocm-6.3.3/bin/rocm-smi || echo "❌ rocm-smi failed"
        "$HOME/miniconda3/envs/darkpool-rocm/bin/python3" -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'ROCM Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')" || { echo "❌ PyTorch verification failed"; exit 1; }
        conda deactivate
    fi
    
    if [ "$NVIDIA_GPU" = true ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate darkpool-cuda0 || { echo "❌ Failed to activate darkpool-cuda0"; exit 1; }
        echo "darkpool-cuda0:"; /usr/local/cuda-11.4/bin/nvidia-smi -i 0 || echo "❌ nvidia-smi failed"; /usr/local/cuda-11.4/bin/nvcc --version || echo "❌ nvcc failed"
        conda deactivate
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate darkpool-cuda1 || { echo "❌ Failed to activate darkpool-cuda1"; exit 1; }
        echo "darkpool-cuda1:"; /usr/local/cuda-11.4/bin/nvidia-smi -i 1 || echo "❌ nvidia-smi failed"; /usr/local/cuda-11.4/bin/nvcc --version || echo "❌ nvcc failed"
        conda deactivate
    fi
    echo "✅ Verification complete"
}

add_conda_aliases() {
    echo "⌨️ Adding Conda aliases..."
    if ! grep -q "# Conda Aliases" "$HOME/.bashrc"; then
        cat >> "$HOME/.bashrc" << 'EOF'
# Conda Aliases
alias da="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate darklake"
alias dr="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate darkpool-rocm"
alias dc0="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate darkpool-cuda0"
alias dc1="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate darkpool-cuda1"
alias dl="conda deactivate"
alias cl="conda list"
alias ce="conda env list"
EOF
    fi
    echo "✅ Aliases added"
}

main() {
    clean_docker
    detect_gpu
    install_conda
    configure_base_env
    configure_rocm_env
    configure_cuda_envs
    install_docker
    configure_docker_containers
    verify_envs
    add_conda_aliases
    echo "
✨ AI/ML Environment Setup Complete! ✨
- Conda envs: darklake, darkpool-rocm (RX580 with ROCm 6.2 PyTorch), darkpool-cuda0/1 (K80s)
- Docker containers: darkpool-rocm (rocm/pytorch:latest), darkpool-cuda0/1 (nvidia/cuda:11.4.0-runtime-ubuntu24.04)
Commands:
- da/dr/dc0/dc1 : Activate Conda envs
- drun darkpool-rocm : Run ROCm Docker
- docker run -it --gpus '\"device=0\"' darkpool-cuda0 : Run CUDA0 Docker
- docker run -it --gpus '\"device=1\"' darkpool-cuda1 : Run CUDA1 Docker
- dl : Deactivate Conda
- Source ~/.bashrc for aliases
Notes:
- Reboot recommended for Docker group changes
- Open-WebUI ready to install next
- Using ROCm 6.2 PyTorch wheels with ROCm 6.3.3 system install
"
}

main
