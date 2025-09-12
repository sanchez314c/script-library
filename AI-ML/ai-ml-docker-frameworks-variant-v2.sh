#!/bin/bash

# Unified AI/ML Environment Setup Script
# Version: 2.8.6 - Built by Cortana (via Grok 3, xAI) for Jason
# Date: February 23, 2025
set -e

echo "🚀 Starting Unified AI/ML Environment Setup on Ubuntu 24.04.1..."

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

install_conda() {
    echo "📦 Installing Miniconda..."
    [ -d "$HOME/miniconda3" ] && { echo "🧹 Removing existing Conda..."; rm -rf "$HOME/miniconda3"; }
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda3"
    rm miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    "$HOME/miniconda3/bin/conda" init bash
    "$HOME/miniconda3/bin/conda" config --set auto_activate_base false
    "$HOME/miniconda3/bin/conda" update -n base -c defaults conda -y
    source ~/.bashrc
    echo "✅ Miniconda installed"
}

setup_conda_env() {
    local env_name=$1
    echo "🌱 Creating environment: $env_name..."
    "$HOME/miniconda3/bin/conda" create -y -n "$env_name" python=3.10
}

configure_base_env() {
    echo "🌱 Configuring darklake base environment..."
    setup_conda_env "darklake"
    source "$HOME/miniconda3/bin/activate" darklake
    pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter ipython tqdm requests
    if [ "$AMD_GPU" = true ]; then
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
    fi
    if [ "$NVIDIA_GPU" = true ]; then
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu114
    fi
    pip install transformers datasets accelerate lightning bitsandbytes deepspeed diffusers
    conda deactivate
    echo "✅ darklake configured"
}

configure_rocm_env() {
    if [ "$AMD_GPU" = true ]; then
        echo "🌱 Configuring darkpool-rocm environment..."
        setup_conda_env "darkpool-rocm"
        source "$HOME/miniconda3/bin/activate" darkpool-rocm
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
        pip install transformers datasets
        mkdir -p "$HOME/miniconda3/envs/darkpool-rocm/etc/conda/activate.d"
        cat > "$HOME/miniconda3/envs/darkpool-rocm/etc/conda/activate.d/rocm_vars.sh" << EOF
#!/bin/bash
export PATH="/opt/rocm-6.2/bin:\$PATH"
export LD_LIBRARY_PATH="/opt/rocm-6.2/lib:/opt/rocm-6.2/lib64:\$LD_LIBRARY_PATH"
export ROCM_PATH="/opt/rocm-6.2"
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
        source "$HOME/miniconda3/bin/activate" darkpool-cuda0
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu114
        pip install transformers datasets
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
        source "$HOME/miniconda3/bin/activate" darkpool-cuda1
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu114
        pip install transformers datasets
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
    echo "🐳 Installing Docker..."
    sudo apt-get install -y docker.io containerd
    sudo usermod -aG docker "$USER"
    sudo systemctl start docker
    sudo systemctl enable docker
    echo "✅ Docker installed"
}

configure_docker_gpu() {
    if [ "$NVIDIA_GPU" = true ]; then
        echo "🔧 Configuring Docker for NVIDIA GPU..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt update
        sudo apt install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        echo "✅ NVIDIA Docker configured"
    fi
    if [ "$AMD_GPU" = true ]; then
        echo "🔧 Configuring Docker for AMD GPU..."
        echo '{"runtimes": {"rocm": {"path": "/opt/rocm-6.2/bin/rocm-container-runtime", "runtimeArgs": []}}}' | sudo tee /etc/docker/daemon.json
        sudo systemctl restart docker
        echo "✅ AMD Docker configured"
    fi
}

verify_envs() {
    echo "🔍 Verifying environments..."
    source "$HOME/miniconda3/bin/activate" darklake
    echo "darklake:"; python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, ROCm: {hasattr(torch, \"hip\") and torch.hip.is_available()}')" || echo "❌ darklake failed"
    conda deactivate
    
    if [ "$AMD_GPU" = true ]; then
        source "$HOME/miniconda3/bin/activate" darkpool-rocm
        echo "darkpool-rocm:"; rocminfo | grep -A 5 "Name:.*gfx803" || echo "❌ rocminfo failed"; rocm-smi || echo "❌ rocm-smi failed"
        conda deactivate
    fi
    
    if [ "$NVIDIA_GPU" = true ]; then
        source "$HOME/miniconda3/bin/activate" darkpool-cuda0
        echo "darkpool-cuda0:"; nvidia-smi -i 0 || echo "❌ nvidia-smi failed"; nvcc --version || echo "❌ nvcc failed"
        conda deactivate
        source "$HOME/miniconda3/bin/activate" darkpool-cuda1
        echo "darkpool-cuda1:"; nvidia-smi -i 1 || echo "❌ nvidia-smi failed"; nvcc --version || echo "❌ nvcc failed"
        conda deactivate
    fi
    echo "✅ Verification complete"
}

add_conda_aliases() {
    echo "⌨️ Adding Conda aliases..."
    grep -q "# Conda Aliases" ~/.bashrc || echo '
# Conda Aliases
alias da="conda activate darklake"
alias dr="conda activate darkpool-rocm"
alias dc0="conda activate darkpool-cuda0"
alias dc1="conda activate darkpool-cuda1"
alias dl="conda deactivate"
alias cl="conda list"
alias ce="conda env list"' >> ~/.bashrc
    echo "✅ Aliases added"
}

main() {
    detect_gpu
    install_conda
    configure_base_env
    configure_rocm_env
    configure_cuda_envs
    install_docker
    configure_docker_gpu
    verify_envs
    add_conda_aliases
    echo "
✨ AI/ML Environment Setup Complete! ✨
- Conda envs: darklake, darkpool-rocm (RX580), darkpool-cuda0/1 (K80s)
- Docker installed with GPU support
Commands:
- da/dr/dc0/dc1 : Activate envs
- dl : Deactivate
- Source ~/.bashrc for aliases
Notes:
- Reboot recommended for Docker group changes
- Open-WebUI ready to install next
"
}

main
