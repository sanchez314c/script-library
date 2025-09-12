#!/bin/bash
# Conda AI/ML Environment Setup Script
# Version: 3.9
# Date: March 23, 2025
# Author: Aether (xAI) for heathen-admin
# Description: Sets up BASE, ROCM, CUDA0, and CUDA1 Conda environments; BASE auto-activates for general use

set -e
set -x

# Define paths and user
CONDA_BASE="/home/${SUDO_USER:-$USER}/miniconda3"
USER="heathen-admin"
LOG_FILE="/home/$USER/conda_ai_ml_setup.log"

echo "🚀 Starting Conda AI/ML Environment Setup..." | tee -a "$LOG_FILE"

remove_existing_conda() {
    echo "🧹 Checking for and removing existing Conda installations..." | tee -a "$LOG_FILE"
    local conda_paths=("/home/$USER/miniconda3" "/home/$USER/anaconda3" "/opt/miniconda3" "/opt/anaconda3")
    for path in "${conda_paths[@]}"; do
        if [ -d "$path" ]; then
            echo "🗑️ Removing Conda at $path..." | tee -a "$LOG_FILE"
            sudo rm -rf "$path" || { echo "❌ Failed to remove $path" | tee -a "$LOG_FILE"; exit 1; }
        fi
    done

    local shell_files=("/home/$USER/.bashrc" "/home/$USER/.bash_profile")
    for file in "${shell_files[@]}"; do
        if [ -f "$file" ]; then
            echo "🧹 Cleaning Conda init from $file..." | tee -a "$LOG_FILE"
            sudo sed -i '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' "$file" || true
            sudo sed -i '/conda.sh/d' "$file" || true
            sudo sed -i '/conda activate/d' "$file" || true
        fi
    done
    echo "✅ Existing Conda installations removed" | tee -a "$LOG_FILE"
}

install_conda() {
    echo "📦 Installing Miniconda..." | tee -a "$LOG_FILE"
    [ -d "$CONDA_BASE" ] && sudo rm -rf "$CONDA_BASE"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh || { echo "❌ Miniconda download failed" | tee -a "$LOG_FILE"; exit 1; }
    sudo -u "$USER" bash miniconda.sh -b -p "$CONDA_BASE" || { echo "❌ Miniconda install failed" | tee -a "$LOG_FILE"; exit 1; }
    rm miniconda.sh
    sudo chown -R "$USER:$USER" "$CONDA_BASE"
    sudo -u "$USER" "$CONDA_BASE/bin/conda" init bash || { echo "❌ Conda init failed" | tee -a "$LOG_FILE"; exit 1; }
    [ -f "/home/$USER/.bash_profile" ] || sudo -u "$USER" bash -c "echo '[ -f ~/.bashrc ] && . ~/.bashrc' >> /home/$USER/.bash_profile"
    sudo -u "$USER" "$CONDA_BASE/bin/conda" config --set auto_activate_base false
    sudo -u "$USER" "$CONDA_BASE/bin/conda" update -n base -c defaults conda -y || { echo "❌ Conda update failed" | tee -a "$LOG_FILE"; exit 1; }
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    echo "✅ Miniconda installed and sourced" | tee -a "$LOG_FILE"
}

setup_conda_env() {
    local env_name=$1
    local python_ver=$2
    echo "🌱 Creating $env_name..." | tee -a "$LOG_FILE"
    sudo -u "$USER" "$CONDA_BASE/bin/conda" env remove -n "$env_name" 2>/dev/null || true
    sudo -u "$USER" "$CONDA_BASE/bin/conda" create -y -n "$env_name" python="$python_ver" || { echo "❌ Failed to create $env_name" | tee -a "$LOG_FILE"; exit 1; }
    sudo -u "$USER" "$CONDA_BASE/bin/conda" env list | grep -q "^$env_name " || { echo "❌ $env_name not found after creation" | tee -a "$LOG_FILE"; exit 1; }
    echo "✅ $env_name created" | tee -a "$LOG_FILE"
}

configure_global_env() {
    echo "🌍 Configuring /etc/environment..." | tee -a "$LOG_FILE"
    sudo cp /etc/environment /etc/environment.bak || { echo "❌ Failed to backup /etc/environment" | tee -a "$LOG_FILE"; exit 1; }
    sudo bash -c "cat > /etc/environment" << 'EOF'
DEBIAN_FRONTEND=noninteractive
PYTHONUNBUFFERED=1
PYTHONENCODING=UTF-8
PIP_ROOT_USER_ACTION=ignore
ROCM_PATH=/opt/rocm-6.3.4
HIP_PATH=/opt/rocm-6.3.4/hip
PATH=/opt/rocm-6.3.4/bin:/usr/local/cuda/bin:$PATH
LD_LIBRARY_PATH=/opt/rocm-6.3.4/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
CUDA_HOME=/usr/local/cuda
CUDA_PATH=/usr/local/cuda
EOF
    echo "✅ /etc/environment configured" | tee -a "$LOG_FILE"
}

configure_base_env() {
    echo "🌱 Configuring BASE environment..." | tee -a "$LOG_FILE"
    setup_conda_env "BASE" "3.10"
    sudo -u "$USER" bash -c "eval \"\$($CONDA_BASE/bin/conda shell.bash hook)\" && conda activate BASE && pip install numpy jupyter" || { echo "❌ Failed to install packages in BASE" | tee -a "$LOG_FILE"; exit 1; }
    echo "✅ BASE configured" | tee -a "$LOG_FILE"
}

configure_rocm_env() {
    echo "🌱 Configuring ROCM environment..." | tee -a "$LOG_FILE"
    setup_conda_env "ROCM" "3.10"
    sudo -u "$USER" bash -c "eval \"\$($CONDA_BASE/bin/conda shell.bash hook)\" && conda activate ROCM && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm6.2" || { echo "❌ Failed to install packages in ROCM" | tee -a "$LOG_FILE"; exit 1; }
    sudo -u "$USER" bash -c "eval \"\$($CONDA_BASE/bin/conda shell.bash hook)\" && conda activate ROCM && pip install ollama" || { echo "❌ Failed to install Ollama in ROCM" | tee -a "$LOG_FILE"; exit 1; }
    sudo mkdir -p "$CONDA_BASE/envs/ROCM/etc/conda/activate.d" && sudo chown -R "$USER:$USER" "$CONDA_BASE/envs/ROCM/etc/conda"
    echo "#!/bin/bash" | sudo tee "$CONDA_BASE/envs/ROCM/etc/conda/activate.d/env_vars.sh" > /dev/null
    cat << 'EOF' | sudo tee -a "$CONDA_BASE/envs/ROCM/etc/conda/activate.d/env_vars.sh" > /dev/null
export HSA_ENABLE_SDMA=0
export ROC_ENABLE_PRE_VEGA=1
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export HIP_VISIBLE_DEVICES=0
export HIP_PLATFORM=amd
export ROCR_VISIBLE_DEVICES=0
export HSA_NO_SCRATCH=1
export HIP_FORCE_DEV=0
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export USE_CUDA=0
export USE_ROCM=1
export USE_NINJA=1

# HIP paths and compiler setup for CMake
export hip_DIR=/opt/rocm-6.3.4/lib/cmake/hip
export CMAKE_PREFIX_PATH=/opt/rocm-6.3.4:$CMAKE_PREFIX_PATH
export CMAKE_HIP_COMPILER=/opt/rocm-6.3.4/bin/hipcc
export MAKE_HIP_FLAGS=-I/opt/rocm-6.3.4/hip/include
export CMAKE_HIP_FLAGS="-I/opt/rocm-6.3.4/hip/include"
export CC=/opt/rocm-6.3.4/bin/hipcc
export CXX=/opt/rocm-6.3.4/bin/hipcc

export PYTORCH_ROCM_ARCH=gfx803
export ROCM_ARCH=gfx803
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_ENABLED=1
export num_gpu=1
export OLLAMA_GPU_OVERHEAD=0.2
export AMD_LOG_LEVEL=3
export LLAMA_HIPBLAS=1
export OLLAMA_LLM_LIBRARY=rocm_v6
export OLLAMA_DEBUG=true
export OLLAMA_FLASH_ATTENTION=true
export GPU_MAX_HEAP_SIZE=100
export GPU_USE_SYNC_OBJECTS=1
export GPU_MAX_ALLOC_PERCENT=100
export GPU_SINGLE_ALLOC_PERCENT=100
export OLLAMA_NUM_THREADS=20
EOF
    sudo chmod +x "$CONDA_BASE/envs/ROCM/etc/conda/activate.d/env_vars.sh"
    sudo mkdir -p "$CONDA_BASE/envs/ROCM/etc/conda/deactivate.d" && sudo chown -R "$USER:$USER" "$CONDA_BASE/envs/ROCM/etc/conda"
    echo "#!/bin/bash" | sudo tee "$CONDA_BASE/envs/ROCM/etc/conda/deactivate.d/env_vars.sh" > /dev/null
    cat << 'EOF' | sudo tee -a "$CONDA_BASE/envs/ROCM/etc/conda/deactivate.d/env_vars.sh" > /dev/null
unset HSA_ENABLE_SDMA
unset ROC_ENABLE_PRE_VEGA
unset HSA_OVERRIDE_GFX_VERSION
unset HIP_VISIBLE_DEVICES
unset HIP_PLATFORM
unset ROCR_VISIBLE_DEVICES
unset HSA_NO_SCRATCH
unset HIP_FORCE_DEV
unset TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL
unset USE_CUDA
unset USE_ROCM
unset USE_NINJA
unset CMAKE_PREFIX_PATH
unset MAKE_HIP_FLAGS
unset CMAKE_HIP_FLAGS
unset CMAKE_C_COMPILER
unset CMAKE_CXX_COMPILER
unset CMAKE_HIP_COMPILER
unset hip_DIR
unset CC
unset CXX
unset PYTORCH_ROCM_ARCH
unset ROCM_ARCH
unset TORCH_BLAS_PREFER_HIPBLASLT
unset PYTORCH_TUNABLEOP_ENABLED
unset num_gpu
unset OLLAMA_GPU_OVERHEAD
unset AMD_LOG_LEVEL
unset LLAMA_HIPBLAS
unset OLLAMA_LLM_LIBRARY
unset OLLAMA_DEBUG
unset OLLAMA_FLASH_ATTENTION
unset GPU_MAX_HEAP_SIZE
unset GPU_USE_SYNC_OBJECTS
unset GPU_MAX_ALLOC_PERCENT
unset GPU_SINGLE_ALLOC_PERCENT
unset OLLAMA_NUM_THREADS
EOF
    sudo chmod +x "$CONDA_BASE/envs/ROCM/etc/conda/deactivate.d/env_vars.sh"
    echo "✅ ROCM configured" | tee -a "$LOG_FILE"
}

configure_cuda_envs() {
    for env in CUDA0 CUDA1; do
        echo "🌱 Configuring $env..." | tee -a "$LOG_FILE"
        setup_conda_env "$env" "3.10"
        sudo -u "$USER" bash -c "eval \"\$($CONDA_BASE/bin/conda shell.bash hook)\" && conda activate $env && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118" || { echo "❌ Failed to install packages in $env" | tee -a "$LOG_FILE"; exit 1; }
        sudo mkdir -p "$CONDA_BASE/envs/$env/etc/conda/activate.d" && sudo chown -R "$USER:$USER" "$CONDA_BASE/envs/$env/etc/conda"
        echo "#!/bin/bash" | sudo tee "$CONDA_BASE/envs/$env/etc/conda/activate.d/env_vars.sh" > /dev/null
        cat << EOF | sudo tee -a "$CONDA_BASE/envs/$env/etc/conda/activate.d/env_vars.sh" > /dev/null
export CUDA_VISIBLE_DEVICES=$([ "$env" = "CUDA0" ] && echo 0 || echo 1)
export TORCH_CUDA_ARCH_LIST="3.5;3.7"
export CUDA_COMPUTE_MAJOR=3
export CUDA_COMPUTE_MINOR=7
export FORCE_CUDA=1
export CUDA_CACHE_PATH=~/.nv/ComputeCache
export OLLAMA_NUM_THREADS=20
EOF
        sudo chmod +x "$CONDA_BASE/envs/$env/etc/conda/activate.d/env_vars.sh"
        sudo mkdir -p "$CONDA_BASE/envs/$env/etc/conda/deactivate.d" && sudo chown -R "$USER:$USER" "$CONDA_BASE/envs/$env/etc/conda"
        echo "#!/bin/bash" | sudo tee "$CONDA_BASE/envs/$env/etc/conda/deactivate.d/env_vars.sh" > /dev/null
        cat << 'EOF' | sudo tee -a "$CONDA_BASE/envs/$env/etc/conda/deactivate.d/env_vars.sh" > /dev/null
unset CUDA_VISIBLE_DEVICES
unset TORCH_CUDA_ARCH_LIST
unset CUDA_COMPUTE_MAJOR
unset CUDA_COMPUTE_MINOR
unset FORCE_CUDA
unset CUDA_CACHE_PATH
unset OLLAMA_NUM_THREADS
EOF
        sudo chmod +x "$CONDA_BASE/envs/$env/etc/conda/deactivate.d/env_vars.sh"
    done
    echo "✅ CUDA0 and CUDA1 configured" | tee -a "$LOG_FILE"
}

create_cmake_template() {
    echo "📝 Creating CMake template for Ollama compilation..." | tee -a "$LOG_FILE"
    mkdir -p "/home/$USER/bin" && sudo chown -R "$USER:$USER" "/home/$USER/bin"
    cat << 'EOF' | sudo tee "/home/$USER/bin/build-ollama.sh" > /dev/null
#!/bin/bash
# Ollama build script for AMD RX580 (gfx803)
# Usage: ./build-ollama.sh [path-to-ollama-source]

set -e

OLLAMA_SRC=${1:-"$HOME/ollama-for-amd"}

if [ ! -d "$OLLAMA_SRC" ]; then
    echo "❌ Ollama source directory not found at $OLLAMA_SRC"
    echo "Please clone the repository first or specify the correct path"
    exit 1
fi

cd "$OLLAMA_SRC"

echo "🔧 Building Ollama with HIP/ROCm for gfx803..."
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMDGPU_TARGETS=gfx803 \
  -DGGML_HIPBLAS=ON \
  -DGGML_NO_CUBLAS=ON \
  -Dhip_DIR=/opt/rocm-6.3.4/lib/cmake/hip

echo "🔨 Compiling..."
cmake --build build -- -j$(nproc)

echo "✅ Build complete! The binary should be in the 'build' directory."
EOF
    sudo chmod +x "/home/$USER/bin/build-ollama.sh"
    echo "✅ CMake template created at /home/$USER/bin/build-ollama.sh" | tee -a "$LOG_FILE"
}

verify_envs() {
    echo "🔍 Verifying environments..." | tee -a "$LOG_FILE"
    sudo -u "$USER" bash -c "eval \"\$($CONDA_BASE/bin/conda shell.bash hook)\" && conda activate BASE && python -c 'import numpy; print(\"BASE: NumPy OK\")'" | tee -a "$LOG_FILE"
    # Note: We skip verification that depends on GPU drivers being present
    echo "✅ BASE environment verification complete" | tee -a "$LOG_FILE"
}

add_conda_aliases() {
    echo "⌨️ Adding aliases and auto-activating BASE..." | tee -a "$LOG_FILE"
    sudo -u "$USER" bash -c "grep -q '# Conda Aliases and Auto-Activation' /home/$USER/.bashrc || cat >> /home/$USER/.bashrc" << 'EOF'
# Conda Aliases and Auto-Activation
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate BASE
fi
alias rocm="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate ROCM"
alias cuda0="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate CUDA0"
alias cuda1="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate CUDA1"
alias nanorocm="nano $HOME/miniconda3/envs/ROCM/etc/conda/activate.d/env_vars.sh"
alias nanocuda0="nano $HOME/miniconda3/envs/CUDA0/etc/conda/activate.d/env_vars.sh"
alias nanocuda1="nano $HOME/miniconda3/envs/CUDA1/etc/conda/activate.d/env_vars.sh"
alias nanobash="sudo nano ~/.bashrc"
alias nanoenv="sudo nano /etc/environment"
alias watchrocm="watch -n 1 rocm-smi"
alias watchcuda="watch -n 1 nvidia-smi"
# Ollama Service Edit Aliases
alias ollamarocmss="sudo nano /etc/systemd/system/ollama-rocm.service"
alias ollamacuda0ss="sudo nano /etc/systemd/system/ollama-cuda0.service"
alias ollamacuda1ss="sudo nano /etc/systemd/system/ollama-cuda1.service"

# Ollama Config Edit Aliases
alias ollamarocmcfg="nano ~/.ollama-rocm/config.json"
alias ollamacuda0cfg="nano ~/.ollama-cuda0/config.json"
alias ollamacuda1cfg="nano ~/.ollama-cuda1/config.json"

# Ollama Service Control Aliases
alias ollamarocmstart="sudo systemctl start ollama-rocm"
alias ollamacuda0start="sudo systemctl start ollama-cuda0"
alias ollamacuda1start="sudo systemctl start ollama-cuda1"

alias ollamarocmstop="sudo systemctl stop ollama-rocm"
alias ollamacuda0stop="sudo systemctl stop ollama-cuda0"
alias ollamacuda1stop="sudo systemctl stop ollama-cuda1"

alias ollamarocmrestart="sudo systemctl restart ollama-rocm"
alias ollamacuda0restart="sudo systemctl restart ollama-cuda0"
alias ollamacuda1restart="sudo systemctl restart ollama-cuda1"

alias ollamastatus="sudo systemctl status ollama-rocm ollama-cuda0 ollama-cuda1"
alias ollamaports="sudo netstat -tulpn | grep ollama"

# Add $HOME/bin to PATH
export PATH="$HOME/bin:$PATH"
EOF
    echo "✅ Aliases and auto-activation added" | tee -a "$LOG_FILE"
}

main() {
    [[ $EUID -ne 0 ]] && { echo "❌ This script must be run as root (sudo)." | tee -a "$LOG_FILE"; exit 1; }
    remove_existing_conda
    install_conda
    configure_global_env
    configure_base_env
    configure_rocm_env
    configure_cuda_envs
    create_cmake_template
    verify_envs
    add_conda_aliases
    echo "
✨ Conda AI/ML Setup Complete!
# Envs: BASE (general), ROCM (AMD RX580), CUDA0 (NVIDIA K80 GPU 0), CUDA1 (NVIDIA K80 GPU 1)
# BASE auto-activates on shell spawn
# Commands: rocm, cuda0, cuda1 (activate envs); nanorocm, nanocuda0, nanocuda1 (edit env vars)
# Added build-ollama.sh script in ~/bin for easy compilation with proper HIP settings
# Example usage: ~/bin/build-ollama.sh ~/ollama-for-amd
# Log: $LOG_FILE
" | tee -a "$LOG_FILE"
    source "/home/$USER/.bashrc"
}

main
