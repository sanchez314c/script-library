#!/bin/bash

# Unified AI/ML Environment Setup Script
# Version: 2.9.24 - Built by Grok 3 (xAI) for Jason
# Date: March 1, 2025 - Fixed Docker CUDA tag to 11.4.3-cudnn8-runtime-ubuntu20.04
set -e
set -x

LOG_FILE="/home/$USER/ai_ml_setup.log"
echo "🚀 Starting Unified AI/ML Environment Setup..." | tee -a "$LOG_FILE"

clean_docker() {
    echo "🧹 Cleaning up Docker..." | tee -a "$LOG_FILE"
    sudo systemctl stop docker docker.socket containerd 2>/dev/null || true
    sudo pkill -f dockerd 2>/dev/null || true
    sudo apt-get purge -y docker.io docker-ce docker-ce-cli containerd docker-compose nvidia-container-toolkit || true
    sudo rm -rf /var/lib/docker /etc/docker /run/docker.sock /var/run/docker.sock ~/.docker
    sudo systemctl daemon-reload
    sudo gpasswd -d "$USER" docker 2>/dev/null || true
    echo "✅ Docker cleaned" | tee -a "$LOG_FILE"
}

detect_gpu() {
    echo "🔍 Detecting GPUs..." | tee -a "$LOG_FILE"
    NVIDIA_GPU=$(command -v nvidia-smi >/dev/null 2>&1 && echo true || echo false)
    AMD_GPU=$([ -x "/opt/rocm/bin/rocminfo" ] && echo true || echo false)
    echo "NVIDIA: $NVIDIA_GPU, AMD: $AMD_GPU" | tee -a "$LOG_FILE"
}

install_conda() {
    echo "📦 Installing Miniconda..." | tee -a "$LOG_FILE"
    [ -d "$HOME/miniconda3" ] && sudo rm -rf "$HOME/miniconda3"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh || { echo "❌ Miniconda download failed" | tee -a "$LOG_FILE"; exit 1; }
    bash miniconda.sh -b -p "$HOME/miniconda3" || { echo "❌ Miniconda install failed" | tee -a "$LOG_FILE"; exit 1; }
    rm miniconda.sh
    sudo chown -R "$USER:$USER" "$HOME/miniconda3"
    export PATH="$HOME/miniconda3/bin:$PATH"
    echo "export PATH=\"$HOME/miniconda3/bin:\$PATH\"" > "$HOME/.bashrc"
    "$HOME/miniconda3/bin/conda" init bash || { echo "❌ Conda init failed" | tee -a "$LOG_FILE"; exit 1; }
    [ -f "$HOME/.bash_profile" ] || echo "[ -f ~/.bashrc ] && . ~/.bashrc" >> "$HOME/.bash_profile"
    source "$HOME/.bashrc"
    conda config --set auto_activate_base false
    conda update -n base -c defaults conda -y || { echo "❌ Conda update failed" | tee -a "$LOG_FILE"; exit 1; }
    echo "✅ Miniconda installed" | tee -a "$LOG_FILE"
}

setup_conda_env() {
    local env_name=$1
    echo "🌱 Creating $env_name..." | tee -a "$LOG_FILE"
    source "$HOME/miniconda3/etc/profile.d/conda.sh" || { echo "❌ Conda sourcing failed for $env_name" | tee -a "$LOG_FILE"; exit 1; }
    conda env remove -n "$env_name" 2>/dev/null || true
    conda create -y -n "$env_name" python=3.10 || { echo "❌ Failed to create $env_name" | tee -a "$LOG_FILE"; exit 1; }
    conda env list | grep -q "^$env_name " || { echo "❌ $env_name not found after creation" | tee -a "$LOG_FILE"; exit 1; }
    echo "✅ $env_name created" | tee -a "$LOG_FILE"
}

configure_base_env() {
    echo "🌱 Configuring darklake (CPU-only)..." | tee -a "$LOG_FILE"
    setup_conda_env "darklake"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate darklake || { echo "❌ Failed to activate darklake" | tee -a "$LOG_FILE"; exit 1; }
    pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter ipython tqdm requests || { echo "❌ Base pip failed" | tee -a "$LOG_FILE"; exit 1; }
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || { echo "❌ Torch CPU failed" | tee -a "$LOG_FILE"; exit 1; }
    pip install transformers datasets accelerate lightning bitsandbytes deepspeed diffusers || { echo "❌ Extra pip failed" | tee -a "$LOG_FILE"; exit 1; }
    conda deactivate
    echo "✅ darklake configured" | tee -a "$LOG_FILE"
}

configure_darkpool_rocm_env() {
    [ "$AMD_GPU" != true ] && return
    echo "🌱 Configuring darkpool-rocm..." | tee -a "$LOG_FILE"
    setup_conda_env "darkpool-rocm"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate darkpool-rocm || { echo "❌ Failed to activate darkpool-rocm" | tee -a "$LOG_FILE"; exit 1; }
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2 -v || { echo "❌ Torch ROCm failed" | tee -a "$LOG_FILE"; exit 1; }
    pip install cmake ninja || { echo "❌ CMake/Ninja failed" | tee -a "$LOG_FILE"; exit 1; }
    mkdir -p "$HOME/miniconda3/envs/darkpool-rocm/etc/conda/activate.d"
    cat > "$HOME/miniconda3/envs/darkpool-rocm/etc/conda/activate.d/rocm_vars.sh" << EOF
#!/bin/bash
export ROCM_PATH=/opt/rocm
export PATH=/opt/rocm/bin:\$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/hip/lib:\$LD_LIBRARY_PATH
export ROCM_CGO_CFLAGS="-I/opt/rocm/include -I/opt/rocm/hip/include"
export ROCM_CGO_LDFLAGS="-L/opt/rocm/lib -L/opt/rocm/lib64 -L/opt/rocm/hip/lib -lamdhip64 -lhiprtc -lrocm_smi64 -lroctx64 -lrocfft"
export ROCM_GOFLAGS="-tags=hip"
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export ROC_ENABLE_PRE_VEGA=1
export PYTORCH_ROCM_ARCH=gfx803
export HIP_VISIBLE_DEVICES=0
export HSA_ENABLE_SDMA=0
export HCC_AMDGPU_TARGET=gfx803
export TORCH_BLAS_PREFER_HIPBLASLT=0
export ROCM_ARCH=gfx803
export DAMDGPU_TARGETS=gfx803
EOF
    chmod +x "$HOME/miniconda3/envs/darkpool-rocm/etc/conda/activate.d/rocm_vars.sh"
    conda deactivate
    echo "✅ darkpool-rocm configured" | tee -a "$LOG_FILE"
}

configure_darkpool_cuda_envs() {
    [ "$NVIDIA_GPU" != true ] && return
    for env in cuda0 cuda1; do
        echo "🌱 Configuring darkpool-$env..." | tee -a "$LOG_FILE"
        setup_conda_env "darkpool-$env"
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate "darkpool-$env" || { echo "❌ Failed to activate darkpool-$env" | tee -a "$LOG_FILE"; exit 1; }
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 -v || { echo "❌ Torch CUDA failed" | tee -a "$LOG_FILE"; exit 1; }
        pip install cmake ninja || { echo "❌ CMake/Ninja failed" | tee -a "$LOG_FILE"; exit 1; }
        mkdir -p "$HOME/miniconda3/envs/darkpool-$env/etc/conda/activate.d"
        cat > "$HOME/miniconda3/envs/darkpool-$env/etc/conda/activate.d/cuda_vars.sh" << EOF
#!/bin/bash
export PATH=/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export CUDA_CGO_CFLAGS="-I/usr/local/cuda/include"
export CUDA_CGO_LDFLAGS="-L/usr/local/cuda/lib64 -lcudart -lcublas -lcudnn"
export CUDA_GOFLAGS="-tags=cuda"
export CUDA_VISIBLE_DEVICES=$([ "$env" = "cuda0" ] && echo 0 || echo 1)
export CUDA_ARCH="35 37 50 52"
export CUDA_COMPUTE_MAJOR=3
export CUDA_COMPUTE_MINOR=5
EOF
        chmod +x "$HOME/miniconda3/envs/darkpool-$env/etc/conda/activate.d/cuda_vars.sh"
        conda deactivate
    done
    echo "✅ darkpool-cuda0/1 configured" | tee -a "$LOG_FILE"
}

install_docker() {
    echo "🐳 Installing Docker..." | tee -a "$LOG_FILE"
    sudo apt-get update || { echo "❌ Apt update failed" | tee -a "$LOG_FILE"; exit 1; }
    sudo apt-get install -y docker.io containerd runc || { echo "❌ Docker install failed" | tee -a "$LOG_FILE"; exit 1; }
    
    # Install Docker Compose
    echo "🐳 Installing Docker Compose..." | tee -a "$LOG_FILE"
    sudo apt-get install -y docker-compose || { echo "❌ Docker Compose install failed" | tee -a "$LOG_FILE"; exit 1; }
    docker-compose --version || { echo "⚠️ Docker Compose version check failed, but continuing..." | tee -a "$LOG_FILE"; }
    
    sudo systemctl daemon-reload
    sudo systemctl start containerd || { echo "❌ Containerd start failed" | tee -a "$LOG_FILE"; exit 1; }
    sudo systemctl enable containerd
    sleep 2
    sudo systemctl start docker || { echo "❌ Docker start failed, trying manual..." | tee -a "$LOG_FILE"; sudo dockerd --debug > /tmp/dockerd-debug.log 2>&1 & sleep 5; sudo pkill -f dockerd; sudo systemctl start docker || { echo "❌ Docker still failed" | tee -a "$LOG_FILE"; cat /tmp/dockerd-debug.log | tee -a "$LOG_FILE"; exit 1; }; }
    sudo systemctl enable docker
    sudo usermod -aG docker "$USER"
    sudo systemctl reset-failed docker.service docker.socket containerd.service 2>/dev/null || true
    echo "✅ Docker and Docker Compose installed" | tee -a "$LOG_FILE"
}

configure_docker_containers() {
    echo "🐳 Configuring Docker containers..." | tee -a "$LOG_FILE"
    [ "$AMD_GPU" = true ] && sudo docker pull rocm/pytorch:latest && sudo docker tag rocm/pytorch:latest darkpool-rocm
    [ "$NVIDIA_GPU" = true ] && sudo docker pull nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04 && sudo docker tag nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04 darkpool-cuda0 && sudo docker tag nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04 darkpool-cuda1
    [ "$AMD_GPU" = true ] && ! grep -q "alias drun" "$HOME/.bashrc" && echo "alias drun='docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G -v \$HOME/dockerx:/dockerx -w /dockerx'" >> "$HOME/.bashrc"
    echo "✅ Docker containers configured" | tee -a "$LOG_FILE"
}

verify_envs() {
    echo "🔍 Verifying environments..." | tee -a "$LOG_FILE"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate darklake || { echo "❌ Failed to activate darklake" | tee -a "$LOG_FILE"; exit 1; }
    python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'ROCm: {hasattr(torch, \"hip\")}')" | tee -a "$LOG_FILE"
    conda deactivate
    
    [ "$AMD_GPU" = true ] && conda activate darkpool-rocm && /opt/rocm/bin/rocminfo | grep -A 5 "Name:.*gfx803" && python3 -c "import torch; print(f'ROCm: {torch.hip.is_available()}')" | tee -a "$LOG_FILE" && conda deactivate
    [ "$NVIDIA_GPU" = true ] && conda activate darkpool-cuda0 && nvidia-smi -i 0 && python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')" | tee -a "$LOG_FILE" && conda deactivate
    [ "$NVIDIA_GPU" = true ] && conda activate darkpool-cuda1 && nvidia-smi -i 1 && python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')" | tee -a "$LOG_FILE" && conda deactivate
    
    # Verify Docker Compose installation
    echo "🔍 Verifying Docker Compose..." | tee -a "$LOG_FILE"
    docker-compose --version | tee -a "$LOG_FILE" || echo "⚠️ Docker Compose verification failed. You may need to log out and back in." | tee -a "$LOG_FILE"
    
    echo "✅ Verification complete" | tee -a "$LOG_FILE"
}

add_conda_aliases() {
    echo "⌨️ Adding aliases..." | tee -a "$LOG_FILE"
    grep -q "# Conda Aliases" "$HOME/.bashrc" || cat >> "$HOME/.bashrc" << 'EOF'
# Conda Aliases
alias da="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate darklake"
alias dr="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate darkpool-rocm"
alias dc0="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate darkpool-cuda0"
alias dc1="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate darkpool-cuda1"
alias dl="conda deactivate"
EOF
    echo "✅ Aliases added" | tee -a "$LOG_FILE"
}

main() {
    clean_docker
    detect_gpu
    install_conda
    configure_base_env
    configure_darkpool_rocm_env
    configure_darkpool_cuda_envs
    install_docker
    configure_docker_containers
    verify_envs
    add_conda_aliases
    echo "
✨ Setup Complete!
- Conda envs: darklake (CPU), darkpool-rocm (RX580), darkpool-cuda0/1 (K80s)
- Docker: darkpool-rocm, darkpool-cuda0/1
- Docker Compose is now installed
- Commands: da/dr/dc0/dc1 (activate), dl (deactivate)
- Next: Run Ollama installer
- Note: Reboot may be needed for Docker group
- Log: $LOG_FILE

IMPORTANT: To use docker-compose right away, run: 
  su - $USER
Or simply reboot your system.
" | tee -a "$LOG_FILE"
}

main
