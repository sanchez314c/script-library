#!/bin/bash

# Conda AI/ML Environment Setup Script
# Version: 2.9.24 - Built by Grok 3 (xAI) for Jason
# Date: March 2, 2025
set -e
set -x

# Determine the correct username: prefer SUDO_USER, then LOGNAME, then whoami
TARGET_USER="${SUDO_USER:-${LOGNAME:-$(whoami)}}"
if [ "$TARGET_USER" = "root" ]; then
    echo "⚠️ Warning: Running as root, trying to guess the real user..."
    # Try to find a non-root user from /home directory
    FIRST_USER=$(ls -1 /home | head -n 1)
    if [ -n "$FIRST_USER" ]; then
        TARGET_USER="$FIRST_USER"
        echo "ℹ️ Using first user found in /home: $TARGET_USER"
    fi
fi

LOG_FILE="/home/$TARGET_USER/conda_setup.log"
echo "🚀 Starting Conda AI/ML Environment Setup..." | tee -a "$LOG_FILE"

detect_gpu() {
    echo "🔍 Detecting GPUs..." | tee -a "$LOG_FILE"
    NVIDIA_GPU=$(command -v nvidia-smi >/dev/null 2>&1 && echo true || echo false)
    AMD_GPU=$([ -x "/opt/rocm/bin/rocminfo" ] && echo true || echo false)
    echo "NVIDIA: $NVIDIA_GPU, AMD: $AMD_GPU" | tee -a "$LOG_FILE"
}

install_conda() {
    echo "📦 Installing Miniconda..." | tee -a "$LOG_FILE"
    [ -d "/home/$TARGET_USER/miniconda3" ] && sudo rm -rf "/home/$TARGET_USER/miniconda3"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh || { echo "❌ Miniconda download failed" | tee -a "$LOG_FILE"; exit 1; }
    sudo -u "$TARGET_USER" bash miniconda.sh -b -p "/home/$TARGET_USER/miniconda3" || { echo "❌ Miniconda install failed" | tee -a "$LOG_FILE"; exit 1; }
    rm miniconda.sh
    sudo chown -R "$TARGET_USER:$TARGET_USER" "/home/$TARGET_USER/miniconda3"
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/conda" init bash || { echo "❌ Conda init failed" | tee -a "$LOG_FILE"; exit 1; }
    [ -f "/home/$TARGET_USER/.bash_profile" ] || sudo -u "$TARGET_USER" bash -c "echo '[ -f ~/.bashrc ] && . ~/.bashrc' >> /home/$TARGET_USER/.bash_profile"
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/conda" config --set auto_activate_base false
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/conda" update -n base -c defaults conda -y || { echo "❌ Conda update failed" | tee -a "$LOG_FILE"; exit 1; }
    echo "✅ Miniconda installed" | tee -a "$LOG_FILE"
}

update_bashrc() {
    echo "⚙️ Updating .bashrc with environment variables..." | tee -a "$LOG_FILE"
    sudo -u "$TARGET_USER" bash -c "grep -q '# CUDA and ROCm Environment Variables' /home/$TARGET_USER/.bashrc || cat >> /home/$TARGET_USER/.bashrc" << 'EOF'
# CUDA and ROCm Environment Variables

# CUDA Environment Variables (available globally)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_ARCH="35 37 50 52"
export CUDA_COMPUTE_MAJOR=3
export CUDA_COMPUTE_MINOR=5

# ROCm Environment Variables (available globally)
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export ROC_ENABLE_PRE_VEGA=1
export PYTORCH_ROCM_ARCH=gfx803
export HIP_VISIBLE_DEVICES=0
export HSA_ENABLE_SDMA=0
export HCC_AMDGPU_TARGET=gfx803
export TORCH_BLAS_PREFER_HIPBLASLT=0
export ROCM_ARCH=gfx803
export DAMDGPU_TARGETS=gfx803
export AMD_SERIALIZE_KERNEL=3
export OLLAMA_LLM_LIBRARY=rocm_v60002

# Activate darklake by default
[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ] && source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate darklake
EOF
    echo "✅ .bashrc updated" | tee -a "$LOG_FILE"
}

setup_conda_env() {
    local env_name=$1
    echo "🌱 Creating $env_name..." | tee -a "$LOG_FILE"
    sudo -u "$TARGET_USER" bash -c "source /home/$TARGET_USER/miniconda3/etc/profile.d/conda.sh" || { echo "❌ Conda sourcing failed for $env_name" | tee -a "$LOG_FILE"; exit 1; }
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/conda" env remove -n "$env_name" 2>/dev/null || true
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/conda" create -y -n "$env_name" python=3.10 || { echo "❌ Failed to create $env_name" | tee -a "$LOG_FILE"; exit 1; }
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/conda" env list | grep -q "^$env_name " || { echo "❌ $env_name not found after creation" | tee -a "$LOG_FILE"; exit 1; }
    echo "✅ $env_name created" | tee -a "$LOG_FILE"
}

configure_base_env() {
    echo "🌱 Configuring darklake (CPU-only)..." | tee -a "$LOG_FILE"
    setup_conda_env "darklake"
    # Use the env's pip directly instead of trying to activate first with --no-interactive flag
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/envs/darklake/bin/pip" --no-input install numpy pandas scipy scikit-learn matplotlib seaborn jupyter ipython tqdm requests || { echo "⚠️ Base pip installation incomplete, continuing anyway" | tee -a "$LOG_FILE"; }
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/envs/darklake/bin/pip" --no-input install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || { echo "⚠️ Torch CPU installation incomplete, continuing anyway" | tee -a "$LOG_FILE"; }
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/envs/darklake/bin/pip" --no-input install transformers datasets accelerate lightning bitsandbytes deepspeed diffusers || { echo "⚠️ Extra pip installation incomplete, continuing anyway" | tee -a "$LOG_FILE"; }
    # Skip deactivation since we're not activating
    echo "✅ darklake configured" | tee -a "$LOG_FILE"
}

configure_darkpool_rocm_env() {
    [ "$AMD_GPU" != true ] && return
    echo "🌱 Configuring darkpool-rocm..." | tee -a "$LOG_FILE"
    setup_conda_env "darkpool-rocm"
    # Use the env's pip directly instead of trying to activate first
    if ! sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/envs/darkpool-rocm/bin/pip" --no-input install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2; then
        echo "⚠️ Trying fallback to ROCm 5.7..." | tee -a "$LOG_FILE"
        if ! sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/envs/darkpool-rocm/bin/pip" --no-input install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7; then
            echo "⚠️ Trying fallback to CPU version for ROCm environment..." | tee -a "$LOG_FILE"
            if ! sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/envs/darkpool-rocm/bin/pip" --no-input install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
                echo "❌ All PyTorch installs failed, continuing with environment setup anyway" | tee -a "$LOG_FILE"
            fi
        fi
    fi
    # Continue regardless of PyTorch install status
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/envs/darkpool-rocm/bin/pip" --no-input install cmake ninja || echo "⚠️ CMake/Ninja install failed, continuing anyway" | tee -a "$LOG_FILE"
    sudo -u "$TARGET_USER" mkdir -p "/home/$TARGET_USER/miniconda3/envs/darkpool-rocm/etc/conda/activate.d"
    sudo -u "$TARGET_USER" bash -c "cat > /home/$TARGET_USER/miniconda3/envs/darkpool-rocm/etc/conda/activate.d/rocm_vars.sh" << EOF
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
    sudo chmod +x "/home/$TARGET_USER/miniconda3/envs/darkpool-rocm/etc/conda/activate.d/rocm_vars.sh"
    # Skip deactivation since we're not activating
    echo "✅ darkpool-rocm configured" | tee -a "$LOG_FILE"
}

configure_darkpool_cuda_envs() {
    [ "$NVIDIA_GPU" != true ] && return
    for env in cuda0 cuda1; do
        echo "🌱 Configuring darkpool-$env..." | tee -a "$LOG_FILE"
        setup_conda_env "darkpool-$env"
        # Use the env's pip directly instead of trying to activate first
        if ! sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/envs/darkpool-$env/bin/pip" --no-input install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118; then
            echo "⚠️ Trying fallback to cu121..." | tee -a "$LOG_FILE"
            if ! sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/envs/darkpool-$env/bin/pip" --no-input install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121; then
                echo "⚠️ Trying fallback to CPU version for CUDA environment..." | tee -a "$LOG_FILE"
                if ! sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/envs/darkpool-$env/bin/pip" --no-input install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
                    echo "❌ All PyTorch installs failed, continuing with environment setup anyway" | tee -a "$LOG_FILE"
                fi
            fi
        fi
        # Continue regardless of PyTorch install status
        sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/envs/darkpool-$env/bin/pip" --no-input install cmake ninja || echo "⚠️ CMake/Ninja install failed, continuing anyway" | tee -a "$LOG_FILE"
        sudo -u "$TARGET_USER" mkdir -p "/home/$TARGET_USER/miniconda3/envs/darkpool-$env/etc/conda/activate.d"
        sudo -u "$TARGET_USER" bash -c "cat > /home/$TARGET_USER/miniconda3/envs/darkpool-$env/etc/conda/activate.d/cuda_vars.sh" << EOF
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
        sudo chmod +x "/home/$TARGET_USER/miniconda3/envs/darkpool-$env/etc/conda/activate.d/cuda_vars.sh"
        # Skip deactivation since we're not activating
    done
    echo "✅ darkpool-cuda0/1 configured" | tee -a "$LOG_FILE"
}

verify_envs() {
    echo "🔍 Verifying environments..." | tee -a "$LOG_FILE"
    
    # Create a function to safely verify an environment
    verify_env() {
        local env_name=$1
        local python_path="/home/$TARGET_USER/miniconda3/envs/$env_name/bin/python3"
        local cmd=$2
        
        if [ -x "$python_path" ]; then
            echo "🔍 Verifying $env_name environment..." | tee -a "$LOG_FILE"
            if sudo -u "$TARGET_USER" "$python_path" -c "$cmd" 2>/dev/null; then
                echo "✅ $env_name verification successful" | tee -a "$LOG_FILE"
                return 0
            else
                echo "⚠️ $env_name verification failed but continuing" | tee -a "$LOG_FILE"
                return 1
            fi
        else
            echo "⚠️ $env_name environment not found, skipping verification" | tee -a "$LOG_FILE"
            return 1
        fi
    }
    
    # Verify base CPU environment
    verify_env "darklake" "import torch; print(f'CPU Environment - CUDA: {torch.cuda.is_available()}')"
    
    # Verify ROCm environment if AMD GPU detected
    if [ "$AMD_GPU" = true ]; then
        # Check if rocminfo is available
        if command -v /opt/rocm/bin/rocminfo >/dev/null 2>&1; then
            echo "🔍 Checking AMD GPU..." | tee -a "$LOG_FILE"
            /opt/rocm/bin/rocminfo | grep -A 5 "Name:.*gfx" 2>/dev/null || echo "⚠️ AMD GPU not fully detected by rocminfo" | tee -a "$LOG_FILE"
        else
            echo "⚠️ rocminfo not found" | tee -a "$LOG_FILE"
        fi
        
        # Verify ROCm PyTorch
        verify_env "darkpool-rocm" "import sys; print('Python:', sys.version); import torch; print(f'ROCm available: {hasattr(torch, \"hip\")}')"
    else
        echo "ℹ️ Skipping ROCm verification (no AMD GPU detected)" | tee -a "$LOG_FILE"
    fi
    
    # Verify CUDA environments if NVIDIA GPU detected
    if [ "$NVIDIA_GPU" = true ]; then
        # Check if nvidia-smi is available
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "🔍 Checking NVIDIA GPUs..." | tee -a "$LOG_FILE"
            nvidia-smi -L | tee -a "$LOG_FILE" || echo "⚠️ NVIDIA GPU info not available" | tee -a "$LOG_FILE"
        else
            echo "⚠️ nvidia-smi not found" | tee -a "$LOG_FILE"
        fi
        
        # Verify each CUDA environment
        for env in cuda0 cuda1; do
            verify_env "darkpool-$env" "import sys; print('Python:', sys.version); import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
        done
    else
        echo "ℹ️ Skipping CUDA verification (no NVIDIA GPU detected)" | tee -a "$LOG_FILE"
    fi
    
    echo "✅ Verification complete" | tee -a "$LOG_FILE"
}

add_conda_aliases() {
    echo "⌨️ Adding aliases..." | tee -a "$LOG_FILE"
    sudo -u "$TARGET_USER" bash -c "grep -q '# Conda Aliases' /home/$TARGET_USER/.bashrc || cat >> /home/$TARGET_USER/.bashrc" << 'EOF'
# Conda Aliases
alias darkl="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate darklake"
alias dr="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate darkpool-rocm"
alias darkp0="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate darkpool-cuda0"
alias darkp1="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate darkpool-cuda1"
alias dl="conda deactivate"
EOF
    echo "✅ Aliases added" | tee -a "$LOG_FILE"
}

main() {
    detect_gpu
    install_conda
    update_bashrc
    configure_base_env
    configure_darkpool_rocm_env
    configure_darkpool_cuda_envs
    verify_envs
    add_conda_aliases
    echo "
✨ Conda Setup Complete!
- Conda envs: darklake (CPU), darkpool-rocm (RX580), darkpool-cuda0/1 (K80s)
- Commands: darkl/dr/darkp0/darkp1 (activate), dl (deactivate)
- Next: Run Docker installer if needed
- Note: Darklake will activate by default on terminal open
- Log: $LOG_FILE
" | tee -a "$LOG_FILE"
    # Skip sourcing bashrc as it can cause issues in non-interactive sessions
    echo "✨ Installation complete. Please open a new terminal to load the environment." | tee -a "$LOG_FILE"
}

main