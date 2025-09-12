#!/bin/bash

# Unified AI/ML Environment Setup Script
# Version: 2.8.6 - Built by Cortana (via Grok 3, xAI) for Jason
# Date: February 23, 2025
set -e

echo "🚀 Starting Unified AI/ML Environment Setup on Ubuntu 24.04.1..."

ORIGINAL_PATH=$PATH

detect_gpu() {
    echo "🔍 Detecting GPU hardware..."
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected"
        NVIDIA_GPU=true
    else
        echo "⚠️ NVIDIA GPU not detected (nvidia-smi missing)"
        NVIDIA_GPU=false
    fi
    
    if command -v rocminfo &> /dev/null; then
        echo "✅ AMD GPU detected"
        AMD_GPU=true
    else
        echo "⚠️ AMD GPU not detected (rocminfo missing)"
        AMD_GPU=false
    fi
}

remove_existing_docker() {
    echo "🧹 Removing existing Docker..."
    sudo systemctl stop docker || true
    sudo apt-get remove -y docker docker-engine docker.io containerd runc || true
    sudo apt-get purge -y docker-ce docker-ce-cli containerd.io docker-compose-plugin || true
    sudo rm -rf /var/lib/docker /var/lib/containerd || true
}

remove_existing_conda() {
    echo "🧹 Removing existing Conda..."
    conda deactivate 2>/dev/null || true
    conda init --reverse --all 2>/dev/null || true
    rm -rf ~/anaconda3 ~/miniconda3 ~/.conda 2>/dev/null || true
    sudo rm -rf /opt/conda /usr/local/conda /root/miniconda3 2>/dev/null || true
    export PATH=$ORIGINAL_PATH
    sed -i '/^# >>> conda initialize >>>/,/^# <<< conda initialize <<</d' ~/.bashrc
}

install_prerequisites() {
    echo "🛠️ Installing prerequisites..."
    sudo apt update && sudo apt install -y wget curl git python3-pip apt-transport-https ca-certificates gnupg lsb-release
}

install_conda() {
    echo "📦 Installing Miniconda..."
    remove_existing_conda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda3"
    rm miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    conda config --set auto_activate_base false
    conda update -n base -c defaults conda -y
    source ~/.bashrc
}

setup_conda_env() {
    local env_name=$1
    echo "🌱 Creating environment: $env_name..."
    conda create -y -n "$env_name" python=3.10
    conda activate "$env_name"
    conda install -y -c conda-forge numpy pandas scipy scikit-learn matplotlib seaborn jupyter ipython tqdm requests pip
    conda deactivate
}

create_base_environment() {
    setup_conda_env "darklake"
    conda activate darklake
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
    pip install tensorflow-rocm tensorflow
    pip install transformers datasets accelerate lightning bitsandbytes deepspeed diffusers
    conda deactivate
}

create_rocm_environment() {
    setup_conda_env "darkpool-rocm"
    conda activate darkpool-rocm
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
    pip install tensorflow-rocm transformers datasets
    mkdir -p "$HOME/.conda/envs/darkpool-rocm/etc/conda/activate.d"
    echo '#!/bin/bash
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export ROC_ENABLE_PRE_VEGA=1
export PYTORCH_ROCM_ARCH=gfx803
export HIP_VISIBLE_DEVICES=0' > "$HOME/.conda/envs/darkpool-rocm/etc/conda/activate.d/env_vars.sh"
    chmod +x "$HOME/.conda/envs/darkpool-rocm/etc/conda/activate.d/env_vars.sh"
    conda deactivate
}

create_cuda_environments() {
    setup_conda_env "darkpool-cuda0"
    conda activate darkpool-cuda0
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
    pip install tensorflow transformers datasets
    mkdir -p "$HOME/.conda/envs/darkpool-cuda0/etc/conda/activate.d"
    echo '#!/bin/bash
export CUDA_VISIBLE_DEVICES=0' > "$HOME/.conda/envs/darkpool-cuda0/etc/conda/activate.d/env_vars.sh"
    chmod +x "$HOME/.conda/envs/darkpool-cuda0/etc/conda/activate.d/env_vars.sh"
    conda deactivate

    setup_conda_env "darkpool-cuda1"
    conda activate darkpool-cuda1
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
    pip install tensorflow transformers datasets
    mkdir -p "$HOME/.conda/envs/darkpool-cuda1/etc/conda/activate.d"
    echo '#!/bin/bash
export CUDA_VISIBLE_DEVICES=1' > "$HOME/.conda/envs/darkpool-cuda1/etc/conda/activate.d/env_vars.sh"
    chmod +x "$HOME/.conda/envs/darkpool-cuda1/etc/conda/activate.d/env_vars.sh"
    conda deactivate
}

install_docker() {
    echo "🐳 Installing Docker for Ubuntu 24.04.1..."
    remove_existing_docker
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu noble stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io
    sudo usermod -aG docker "$USER"
    sudo systemctl start docker
    sudo systemctl enable docker
}

configure_docker_gpu() {
    echo "🔧 Configuring Docker for GPU support..."
    if [ "$NVIDIA_GPU" = true ]; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt update
        sudo apt install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
    fi

    if [ "$AMD_GPU" = true ]; then
        echo '{
            "runtimes": {
                "rocm": {
                    "path": "/usr/bin/rocm-container-runtime",
                    "runtimeArgs": []
                }
            }
        }' | sudo tee /etc/docker/daemon.json
        sudo systemctl restart docker
    fi
}

pull_ml_containers() {
    echo "📥 Pulling ML Docker containers..."
    if [ "$AMD_GPU" = true ]; then
        sudo docker pull ulyssesrr/rocm-xtra-pytorch:latest  # Switched to docker-rocm-xtra for gfx803
    fi
    if [ "$NVIDIA_GPU" = true ]; then
        sudo docker pull nvidia/cuda:12.1.0-devel-ubuntu22.04
        sudo docker tag nvidia/cuda:12.1.0-devel-ubuntu22.04 cuda-k80-gpu0
        sudo docker tag nvidia/cuda:12.1.0-devel-ubuntu22.04 cuda-k80-gpu1
    fi
}

create_env_check_script() {
    echo "🩺 Creating environment check script..."
    cat << 'EOF' | sudo tee /usr/local/bin/check-conda-envs
#!/bin/bash
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

echo "Checking darklake environment..."
conda activate darklake || echo "darklake not found"
python3 -c "import numpy; import pandas; import torch; print(f'Base packages OK, CUDA: {torch.cuda.is_available()}, ROCm: {hasattr(torch, \"hip\") and torch.hip.is_available()}')" || echo "Python check failed"

echo "Checking darkpool-rocm environment..."
conda activate darkpool-rocm || echo "darkpool-rocm not found"
python3 -c "import numpy; print('ROCm env OK')" || echo "Python check failed"

echo "Checking darkpool-cuda0 environment..."
conda activate darkpool-cuda0 || echo "darkpool-cuda0 not found"
python3 -c "import numpy; print('CUDA0 env OK')" || echo "Python check failed"

echo "Checking darkpool-cuda1 environment..."
conda activate darkpool-cuda1 || echo "darkpool-cuda1 not found"
python3 -c "import numpy; print('CUDA1 env OK')" || echo "Python check failed"
EOF
    sudo chmod +x /usr/local/bin/check-conda-envs
}

add_conda_aliases() {
    echo "⌨️ Adding Conda aliases and auto-activation..."
    if ! grep -q "# Conda Aliases" ~/.bashrc; then
        echo '
# Conda Aliases
alias conda-check="/usr/local/bin/check-conda-envs"
alias da="conda activate darklake"
alias dr="conda activate darkpool-rocm"
alias dc0="conda activate darkpool-cuda0"
alias dc1="conda activate darkpool-cuda1"
alias dl="conda deactivate"
alias cl="conda list"
alias ce="conda env list"

# Auto-activate darklake
conda activate darklake' >> ~/.bashrc
    fi
}

main() {
    detect_gpu
    install_prerequisites
    install_conda
    create_base_environment
    create_rocm_environment
    create_cuda_environments
    install_docker
    configure_docker_gpu
    pull_ml_containers
    create_env_check_script
    add_conda_aliases

    echo "✨ Setup Complete! ✨"
    echo "Open a new terminal (or source ~/.bashrc) to use 'darklake' automatically."
    echo "Run 'conda-check' to verify environments."
    echo "Docker run commands:"
    if [ "$AMD_GPU" = true ]; then
        echo "  ROCm (RX 580): sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 16G ulyssesrr/rocm-xtra-pytorch:latest"
    fi
    if [ "$NVIDIA_GPU" = true ]; then
        echo "  CUDA (K80 GPU0): sudo docker run -it --gpus '\"device=0\"' cuda-k80-gpu0"
        echo "  CUDA (K80 GPU1): sudo docker run -it --gpus '\"device=1\"' cuda-k80-gpu1"
    fi
    if [ "$AMD_GPU" = true ]; then
        echo "✅ Note: Using ulyssesrr/rocm-xtra-pytorch for RX 580 (gfx803) support."
        echo "  Verify GPU with: docker run ... ulyssesrr/rocm-xtra-pytorch:latest rocm-smi"
    fi
}

main
