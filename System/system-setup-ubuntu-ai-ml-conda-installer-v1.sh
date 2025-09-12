#!/bin/bash
# Conda AI/ML Environment Setup Script
# Version: 4.0
# Date: March 23, 2025
# Description: Sets up BASE, ROCM, and CUDA Conda environments; BASE auto-activates for general use

set -e

# Check if script is run with sudo
if [ "$(id -u)" -ne 0 ]; then
    echo "❌ This script must be run with sudo" >&2
    exit 1
fi

# Determine real user (not root) even when run with sudo
REAL_USER="${SUDO_USER:-$USER}"
if [ "$REAL_USER" = "root" ] && [ -n "$SUDO_USER" ]; then
    REAL_USER="$SUDO_USER"
fi

# Define paths based on real user
CONDA_BASE="/home/${REAL_USER}/miniconda3"
LOG_FILE="/home/${REAL_USER}/conda_ai_ml_setup.log"

echo "🚀 Starting Conda AI/ML Environment Setup for user ${REAL_USER}..." | tee -a "$LOG_FILE"

# Check for GPU drivers before proceeding
check_gpu_drivers() {
    echo "🔍 Checking for GPU drivers..." | tee -a "$LOG_FILE"
    
    # Check for AMD ROCm
    ROCM_AVAILABLE=false
    if [ -d "/opt/rocm" ] || [ -d "/opt/rocm-6.3.4" ]; then
        if command -v rocminfo &> /dev/null; then
            if rocminfo 2>/dev/null | grep -q "gfx803"; then
                ROCM_AVAILABLE=true
                ROCM_VERSION=$(ls -d /opt/rocm* | grep -oE 'rocm-[0-9]+\.[0-9]+\.[0-9]+' | sort -V | tail -n 1 | cut -d- -f2)
                echo "✅ ROCm ${ROCM_VERSION} detected with gfx803 GPU" | tee -a "$LOG_FILE"
            else
                echo "⚠️ ROCm installed but gfx803 GPU not detected" | tee -a "$LOG_FILE"
            fi
        else
            echo "⚠️ ROCm directory found but rocminfo command missing" | tee -a "$LOG_FILE"
        fi
    else
        echo "❌ ROCm not installed" | tee -a "$LOG_FILE"
    fi
    
    # Check for NVIDIA CUDA
    CUDA_AVAILABLE=false
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &>/dev/null; then
            CUDA_AVAILABLE=true
            CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || echo "Unknown")
            NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
            echo "✅ CUDA ${CUDA_VERSION} detected with ${NUM_GPUS} GPU(s)" | tee -a "$LOG_FILE"
        else
            echo "⚠️ nvidia-smi command failed" | tee -a "$LOG_FILE"
        fi
    else
        echo "❌ NVIDIA drivers not installed" | tee -a "$LOG_FILE"
    fi
    
    # Verify at least one GPU driver is available
    if [ "$ROCM_AVAILABLE" = false ] && [ "$CUDA_AVAILABLE" = false ]; then
        echo "❌ No GPU drivers detected. Please install ROCm or CUDA drivers before continuing." | tee -a "$LOG_FILE"
        exit 1
    fi
}

remove_existing_conda() {
    echo "🧹 Checking for and removing existing Conda installations..." | tee -a "$LOG_FILE"
    local conda_paths=("/home/${REAL_USER}/miniconda3" "/home/${REAL_USER}/anaconda3" "/opt/miniconda3" "/opt/anaconda3")
    for path in "${conda_paths[@]}"; do
        if [ -d "$path" ]; then
            echo "🗑️ Removing Conda at $path..." | tee -a "$LOG_FILE"
            rm -rf "$path" || { echo "❌ Failed to remove $path" | tee -a "$LOG_FILE"; exit 1; }
        fi
    done

    local shell_files=("/home/${REAL_USER}/.bashrc" "/home/${REAL_USER}/.bash_profile")
    for file in "${shell_files[@]}"; do
        if [ -f "$file" ]; then
            echo "🧹 Cleaning Conda init from $file..." | tee -a "$LOG_FILE"
            # Use direct command paths and simple removal approaches
            cp "$file" "${file}.bak"
            grep -v "# >>> conda initialize >>>" "$file" | \
            grep -v "# <<< conda initialize <<<" | \
            grep -v "conda.sh" | \
            grep -v "conda activate" > "${file}.new"
            mv "${file}.new" "$file"
            chown ${REAL_USER}:${REAL_USER} "$file"
        fi
    done
    echo "✅ Existing Conda installations removed" | tee -a "$LOG_FILE"
}

install_conda() {
    echo "📦 Installing Miniconda..." | tee -a "$LOG_FILE"
    [ -d "$CONDA_BASE" ] && rm -rf "$CONDA_BASE"
    
    cd /tmp
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh || { 
        echo "❌ Miniconda download failed" | tee -a "$LOG_FILE"
        exit 1
    }
    
    # Install directly (not using su - user)
    bash miniconda.sh -b -p "$CONDA_BASE" || { 
        echo "❌ Miniconda install failed" | tee -a "$LOG_FILE"
        exit 1
    }
    
    # Fix ownership
    chown -R ${REAL_USER}:${REAL_USER} "$CONDA_BASE"
    
    rm -f miniconda.sh
    
    # Initialize conda (directly, not using su)
    "$CONDA_BASE/bin/conda" init bash || { 
        echo "❌ Conda init failed" | tee -a "$LOG_FILE"
        exit 1
    }
    
    # Create bash_profile if it doesn't exist
    if [ ! -f "/home/${REAL_USER}/.bash_profile" ]; then
        echo '[ -f ~/.bashrc ] && . ~/.bashrc' > "/home/${REAL_USER}/.bash_profile"
        chown ${REAL_USER}:${REAL_USER} "/home/${REAL_USER}/.bash_profile"
    fi
    
    "$CONDA_BASE/bin/conda" config --set auto_activate_base false
    "$CONDA_BASE/bin/conda" update -n base -c defaults conda -y || { 
        echo "❌ Conda update failed" | tee -a "$LOG_FILE"
        exit 1
    }
    
    echo "✅ Miniconda installed" | tee -a "$LOG_FILE"
}

# Function to check compatible Python versions
get_compatible_python_version() {
    local env_name=$1
    local default_version="3.11"
    
    # Check for PyTorch compatibility
    if [[ "$env_name" == "ROCM" ]]; then
        # ROCm PyTorch generally supports up to Python 3.10
        echo "3.10"
    elif [[ "$env_name" == "CUDA" ]]; then
        # CUDA PyTorch supports up to Python 3.11
        echo "3.11"
    else
        # Use latest for BASE
        echo "$default_version"
    fi
}

setup_conda_env() {
    local env_name=$1
    local python_ver=$(get_compatible_python_version "$env_name")
    
    echo "🌱 Creating $env_name with Python $python_ver..." | tee -a "$LOG_FILE"
    
    # Remove the environment if it exists
    "$CONDA_BASE/bin/conda" env remove -n "$env_name" 2>/dev/null || true
    
    # Create the environment with appropriate Python version
    "$CONDA_BASE/bin/conda" create -y -n "$env_name" python="$python_ver" || { 
        echo "❌ Failed to create $env_name" | tee -a "$LOG_FILE"
        exit 1
    }
    
    # Verify the environment was created successfully
    if ! "$CONDA_BASE/bin/conda" env list | grep -q "^$env_name "; then
        echo "❌ $env_name not found after creation" | tee -a "$LOG_FILE"
        exit 1
    fi
    
    # Fix ownership
    chown -R ${REAL_USER}:${REAL_USER} "$CONDA_BASE/envs/$env_name"
    
    echo "✅ $env_name created with Python $python_ver" | tee -a "$LOG_FILE"
}

configure_global_env() {
    echo "🌍 Configuring /etc/environment..." | tee -a "$LOG_FILE"
    # Backup the existing environment file
    cp /etc/environment /etc/environment.bak || { echo "❌ Failed to backup /etc/environment" | tee -a "$LOG_FILE"; exit 1; }
    
    # Create a new environment file
    cat > /etc/environment << 'EOF'
DEBIAN_FRONTEND=noninteractive
PYTHONUNBUFFERED=1
PYTHONENCODING=UTF-8
PIP_ROOT_USER_ACTION=ignore
EOF

    # Add ROCm entries if available
    if [ "$ROCM_AVAILABLE" = true ]; then
        ROCM_PATH=$(ls -d /opt/rocm* | sort -V | tail -n 1)
        cat >> /etc/environment << EOF
ROCM_PATH=${ROCM_PATH}
HIP_PATH=${ROCM_PATH}/hip
PATH=${ROCM_PATH}/bin:\$PATH
LD_LIBRARY_PATH=${ROCM_PATH}/lib:\$LD_LIBRARY_PATH
EOF
    fi

    # Add CUDA entries if available
    if [ "$CUDA_AVAILABLE" = true ]; then
        cat >> /etc/environment << 'EOF'
CUDA_HOME=/usr/local/cuda
CUDA_PATH=/usr/local/cuda
PATH=/usr/local/cuda/bin:$PATH
LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
    fi

    echo "✅ /etc/environment configured" | tee -a "$LOG_FILE"
}

configure_base_env() {
    echo "🌱 Configuring BASE environment..." | tee -a "$LOG_FILE"
    setup_conda_env "BASE" 
    
    # Install packages in the BASE environment
    "$CONDA_BASE/bin/pip" install numpy jupyter ipython matplotlib pandas scipy scikit-learn || { 
        echo "❌ Failed to install packages in BASE" | tee -a "$LOG_FILE"
        exit 1
    }
    
    echo "✅ BASE configured" | tee -a "$LOG_FILE"
}

configure_rocm_env() {
    if [ "$ROCM_AVAILABLE" = false ]; then
        echo "⚠️ Skipping ROCM environment setup (ROCm not detected)" | tee -a "$LOG_FILE"
        return
    fi
    
    echo "🌱 Configuring ROCM environment..." | tee -a "$LOG_FILE"
    setup_conda_env "ROCM"
    
    # Determine appropriate PyTorch ROCm wheel URL based on installed ROCm version
    ROCM_MAJOR_MINOR=$(echo "$ROCM_VERSION" | cut -d. -f1,2)
    TORCH_ROCM_URL="https://download.pytorch.org/whl/rocm6.2"
    
    # Install PyTorch with ROCm support directly (not using su)
    "$CONDA_BASE/bin/conda" run -n ROCM pip install torch torchvision torchaudio --extra-index-url $TORCH_ROCM_URL || { 
        echo "❌ Failed to install PyTorch in ROCM environment" | tee -a "$LOG_FILE"
        exit 1
    }
    
    # Install Ollama
    "$CONDA_BASE/bin/conda" run -n ROCM pip install ollama || { 
        echo "❌ Failed to install Ollama in ROCM environment" | tee -a "$LOG_FILE"
        exit 1
    }
    
    # Create activate script directory
    mkdir -p "$CONDA_BASE/envs/ROCM/etc/conda/activate.d"
    chown -R "${REAL_USER}:${REAL_USER}" "$CONDA_BASE/envs/ROCM/etc/conda"
    
    # Create activation script
    cat > "$CONDA_BASE/envs/ROCM/etc/conda/activate.d/env_vars.sh" << 'EOF'
#!/bin/bash
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
ROCM_PATH=$(ls -d /opt/rocm* | sort -V | tail -n 1)
export hip_DIR=${ROCM_PATH}/lib/cmake/hip
export CMAKE_PREFIX_PATH=${ROCM_PATH}:$CMAKE_PREFIX_PATH
export CMAKE_HIP_COMPILER=${ROCM_PATH}/bin/hipcc
export MAKE_HIP_FLAGS=-I${ROCM_PATH}/hip/include
export CMAKE_HIP_FLAGS="-I${ROCM_PATH}/hip/include"
export CC=${ROCM_PATH}/bin/hipcc
export CXX=${ROCM_PATH}/bin/hipcc

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
    chmod +x "$CONDA_BASE/envs/ROCM/etc/conda/activate.d/env_vars.sh"
    chown "${REAL_USER}:${REAL_USER}" "$CONDA_BASE/envs/ROCM/etc/conda/activate.d/env_vars.sh"
    
    # Create deactivate script directory
    mkdir -p "$CONDA_BASE/envs/ROCM/etc/conda/deactivate.d"
    
    # Create deactivation script
    cat > "$CONDA_BASE/envs/ROCM/etc/conda/deactivate.d/env_vars.sh" << 'EOF'
#!/bin/bash
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
    chmod +x "$CONDA_BASE/envs/ROCM/etc/conda/deactivate.d/env_vars.sh"
    chown "${REAL_USER}:${REAL_USER}" "$CONDA_BASE/envs/ROCM/etc/conda/deactivate.d/env_vars.sh"
    
    echo "✅ ROCM configured" | tee -a "$LOG_FILE"
}

configure_cuda_env() {
    if [ "$CUDA_AVAILABLE" = false ]; then
        echo "⚠️ Skipping CUDA environment setup (CUDA not detected)" | tee -a "$LOG_FILE"
        return
    fi
    
    echo "🌱 Configuring CUDA environment..." | tee -a "$LOG_FILE"
    setup_conda_env "CUDA"
    
    # Install PyTorch with CUDA support
    "$CONDA_BASE/bin/conda" run -n CUDA pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 || { 
        echo "❌ Failed to install PyTorch in CUDA environment" | tee -a "$LOG_FILE"
        exit 1
    }
    
    # Create activate script directory
    mkdir -p "$CONDA_BASE/envs/CUDA/etc/conda/activate.d"
    chown -R "${REAL_USER}:${REAL_USER}" "$CONDA_BASE/envs/CUDA/etc/conda"
    
    # Create activation script
    cat > "$CONDA_BASE/envs/CUDA/etc/conda/activate.d/env_vars.sh" << 'EOF'
#!/bin/bash
# Enable all available CUDA devices
export CUDA_VISIBLE_DEVICES=all
export TORCH_CUDA_ARCH_LIST="3.5;3.7;5.0;6.0;7.0;7.5;8.0;8.6"
export FORCE_CUDA=1
export CUDA_CACHE_PATH=~/.nv/ComputeCache
export OLLAMA_NUM_THREADS=20
EOF
    chmod +x "$CONDA_BASE/envs/CUDA/etc/conda/activate.d/env_vars.sh"
    chown "${REAL_USER}:${REAL_USER}" "$CONDA_BASE/envs/CUDA/etc/conda/activate.d/env_vars.sh"
    
    # Create deactivate script directory
    mkdir -p "$CONDA_BASE/envs/CUDA/etc/conda/deactivate.d"
    
    # Create deactivation script
    cat > "$CONDA_BASE/envs/CUDA/etc/conda/deactivate.d/env_vars.sh" << 'EOF'
#!/bin/bash
unset CUDA_VISIBLE_DEVICES
unset TORCH_CUDA_ARCH_LIST
unset CUDA_COMPUTE_MAJOR
unset CUDA_COMPUTE_MINOR
unset FORCE_CUDA
unset CUDA_CACHE_PATH
unset OLLAMA_NUM_THREADS
EOF
    chmod +x "$CONDA_BASE/envs/CUDA/etc/conda/deactivate.d/env_vars.sh"
    chown "${REAL_USER}:${REAL_USER}" "$CONDA_BASE/envs/CUDA/etc/conda/deactivate.d/env_vars.sh"
    
    echo "✅ CUDA configured" | tee -a "$LOG_FILE"
}

create_cmake_template() {
    echo "📝 Creating CMake template for Ollama compilation..." | tee -a "$LOG_FILE"
    mkdir -p "/home/${REAL_USER}/bin"
    chown -R "${REAL_USER}:${REAL_USER}" "/home/${REAL_USER}/bin"
    
    cat > "/home/${REAL_USER}/bin/build-ollama.sh" << 'EOF'
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

# Find the ROCm installation
ROCM_PATH=$(ls -d /opt/rocm* | sort -V | tail -n 1)

echo "🔧 Building Ollama with HIP/ROCm for gfx803 using $ROCM_PATH..."
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMDGPU_TARGETS=gfx803 \
  -DGGML_HIPBLAS=ON \
  -DGGML_NO_CUBLAS=ON \
  -Dhip_DIR=${ROCM_PATH}/lib/cmake/hip

echo "🔨 Compiling..."
cmake --build build -- -j$(nproc)

echo "✅ Build complete! The binary should be in the 'build' directory."
EOF
    chmod +x "/home/${REAL_USER}/bin/build-ollama.sh"
    chown "${REAL_USER}:${REAL_USER}" "/home/${REAL_USER}/bin/build-ollama.sh"
    echo "✅ CMake template created at /home/${REAL_USER}/bin/build-ollama.sh" | tee -a "$LOG_FILE"
}

verify_envs() {
    echo "🔍 Verifying environments..." | tee -a "$LOG_FILE"
    
    # Verify BASE environment
    echo "Verifying BASE environment..." | tee -a "$LOG_FILE"
    if ! "$CONDA_BASE/bin/conda" run -n BASE python -c 'import numpy; print("BASE: NumPy OK")'; then
        echo "❌ BASE environment verification failed" | tee -a "$LOG_FILE"
        # Continue despite failure
    fi
    
    # Verify ROCM environment
    if [ "$ROCM_AVAILABLE" = true ]; then
        echo "Verifying ROCM environment..." | tee -a "$LOG_FILE"
        if ! "$CONDA_BASE/bin/conda" run -n ROCM python -c 'import torch; print("ROCM: PyTorch", torch.__version__); print("ROCM: ROCm available:", hasattr(torch, "version") and hasattr(torch.version, "hip"))'; then
            echo "❌ ROCM environment verification failed" | tee -a "$LOG_FILE"
            # Continue despite failure
        fi
    fi
    
    # Verify CUDA environment
    if [ "$CUDA_AVAILABLE" = true ]; then
        echo "Verifying CUDA environment..." | tee -a "$LOG_FILE"
        if ! "$CONDA_BASE/bin/conda" run -n CUDA python -c 'import torch; print("CUDA: PyTorch", torch.__version__); print("CUDA: CUDA available:", torch.cuda.is_available())'; then
            echo "❌ CUDA environment verification failed" | tee -a "$LOG_FILE"
            # Continue despite failure
        fi
    fi
    
    echo "✅ Environment verification complete" | tee -a "$LOG_FILE"
}

add_conda_aliases() {
    echo "⌨️ Adding aliases and auto-activating BASE..." | tee -a "$LOG_FILE"
    
    BASHRC_FILE="/home/${REAL_USER}/.bashrc"
    
    # Check if aliases already exist
    if ! grep -q '# Conda Aliases and Auto-Activation' "$BASHRC_FILE"; then
        # Add aliases directly to bashrc
        cat >> "$BASHRC_FILE" << 'EOF'
# Conda Aliases and Auto-Activation
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate BASE
fi
EOF

        # Add ROCM alias if available
        if [ "$ROCM_AVAILABLE" = true ]; then
            cat >> "$BASHRC_FILE" << 'EOF'
alias rocm="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate ROCM"
alias nanorocm="nano $HOME/miniconda3/envs/ROCM/etc/conda/activate.d/env_vars.sh"
alias watchrocm="watch -n 1 rocm-smi"
alias ollamarocmss="sudo nano /etc/systemd/system/ollama-rocm.service"
alias ollamarocmcfg="nano ~/.ollama-rocm/config.json"
alias ollamarocmstart="sudo systemctl start ollama-rocm"
alias ollamarocmstop="sudo systemctl stop ollama-rocm"
alias ollamarocmrestart="sudo systemctl restart ollama-rocm"
EOF
        fi
        
        # Add CUDA alias if available
        if [ "$CUDA_AVAILABLE" = true ]; then
            cat >> "$BASHRC_FILE" << 'EOF'
alias cuda="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate CUDA"
alias nanocuda="nano $HOME/miniconda3/envs/CUDA/etc/conda/activate.d/env_vars.sh"
alias watchcuda="watch -n 1 nvidia-smi"
alias ollamacudass="sudo nano /etc/systemd/system/ollama-cuda.service"
alias ollamacudacfg="nano ~/.ollama-cuda/config.json"
alias ollamacudastart="sudo systemctl start ollama-cuda"
alias ollamacudastop="sudo systemctl stop ollama-cuda"
alias ollamacudarestart="sudo systemctl restart ollama-cuda"
EOF
        fi
        
        # Add common aliases
        cat >> "$BASHRC_FILE" << 'EOF'
alias nanobash="sudo nano ~/.bashrc"
alias nanoenv="sudo nano /etc/environment"
alias ollamastatus="sudo systemctl status ollama-*"
alias ollamaports="sudo netstat -tulpn | grep ollama"

# Add $HOME/bin to PATH
export PATH="$HOME/bin:$PATH"
EOF
        
        # Fix ownership
        chown "${REAL_USER}:${REAL_USER}" "$BASHRC_FILE"
        
        echo "✅ Aliases and auto-activation added" | tee -a "$LOG_FILE"
    else
        echo "⚠️ Aliases already exist in .bashrc - not modifying" | tee -a "$LOG_FILE"
    fi
}

main() {
    check_gpu_drivers
    remove_existing_conda
    install_conda
    configure_global_env
    configure_base_env
    configure_rocm_env
    configure_cuda_env
    create_cmake_template
    verify_envs
    add_conda_aliases
    
    # Print completion message
    echo "
✨ Conda AI/ML Setup Complete!
# Envs: BASE (general), ROCM (AMD RX580), CUDA (NVIDIA GPUs)
# BASE auto-activates on shell spawn
# Commands: rocm, cuda (activate envs); nanorocm, nanocuda (edit env vars)
# Added build-ollama.sh script in ~/bin for easy compilation with proper HIP settings
# Example usage: ~/bin/build-ollama.sh ~/ollama-for-amd
# Log: $LOG_FILE
" | tee -a "$LOG_FILE"
}

main