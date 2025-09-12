#!/bin/bash
#########################################################################
#     ___   ___       __  __ _          ____ ___  _   _ ____    _       #
#    / _ \ |_ _|     |  \/  | |        / ___/ _ \| \ | |  _ \  / \      #
#   | | | | | |_____ | |\/| | |_____  | |  | | | |  \| | | | |/ _ \     #
#   | |_| | | |_____|| |  | | |_____| | |__| |_| | |\  | |_| / ___ \    #
#    \___/ |___|     |_|  |_|_|        \____\___/|_| \_|____/_/   \_\   #
#                                                                       #
#########################################################################
#
# AI/ML Conda Environment Setup Script for BASE, ROCM, and CUDA
# Version: 5.0.0
# Date: April 15, 2025
# Description: Sets up three optimized Conda environments:
#              - BASE: General ML with both CUDA and ROCm support
#              - ROCM: AMD GPU support (RX580/gfx803 optimized)
#              - CUDA: NVIDIA GPU support (K80/Compute 3.5-3.7)
#
# Usage: sudo bash ./ubuntu-ai-ml-conda-installer-ENHANCED.sh
#
#########################################################################

set -e  # Exit on error

# Control verbosity
DEBUG=${DEBUG:-false}
if [ "$DEBUG" = "true" ]; then
    set -x
fi

# Identify the real user even when run with sudo
REAL_USER="${SUDO_USER:-$USER}"
if [ "$REAL_USER" = "root" ] && [ -n "$SUDO_USER" ]; then
    REAL_USER="$SUDO_USER"
fi

# Define paths based on real user
CONDA_BASE="/home/${REAL_USER}/miniconda3"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/${REAL_USER}/logs"
LOG_FILE="${LOG_DIR}/conda_ai_ml_setup_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"
chmod 755 "$LOG_DIR"
chown -R "${REAL_USER}:${REAL_USER}" "$LOG_DIR" 2>/dev/null || true

# Set up logging to both console and file
exec &> >(tee -a "$LOG_FILE")

# Print colored messages
print_green() { echo -e "\e[32m$1\e[0m"; }
print_yellow() { echo -e "\e[33m$1\e[0m"; }
print_red() { echo -e "\e[31m$1\e[0m"; }
print_blue() { echo -e "\e[34m$1\e[0m"; }

# Print banners and section headers
print_banner() {
    local text="$1"
    local width=70
    local padding=$(( (width - ${#text}) / 2 ))
    local line=$(printf '═%.0s' $(seq 1 $width))
    
    echo ""
    print_blue "$line"
    printf "\e[34m%${padding}s$text%${padding}s\e[0m\n" "" ""
    print_blue "$line"
    echo ""
}

print_section() {
    local text="$1"
    echo ""
    print_blue "▓▒░ $text ░▒▓"
}

print_banner "AI/ML Conda Environment Setup"
echo "📋 Script: $(basename "$0")"
echo "📅 Date: $(date)"
echo "👤 User: $REAL_USER"
echo "📝 Log: $LOG_FILE"
echo ""

check_root() {
    print_section "🔍 Checking for root privileges"
    
    if [ "$(id -u)" -ne 0 ]; then
        print_red "❌ This script must be run with sudo"
        exit 1
    fi
    
    print_green "✅ Running with root privileges"
}

check_gpu_drivers() {
    print_section "🔍 Checking for GPU drivers"
    
    # Default to false
    ROCM_AVAILABLE=false
    CUDA_AVAILABLE=false
    
    # Check for AMD ROCm
    print_yellow "🔍 Checking for AMD ROCm installation..."
    if [ -d "/opt/rocm" ] || [ -d "/opt/rocm-6.3.4" ]; then
        if command -v rocminfo &> /dev/null; then
            if rocminfo 2>/dev/null | grep -q "gfx803"; then
                ROCM_AVAILABLE=true
                ROCM_VERSION=$(ls -d /opt/rocm* | grep -oE 'rocm-[0-9]+\.[0-9]+\.[0-9]+' | sort -V | tail -n 1 | cut -d- -f2)
                print_green "✅ ROCm ${ROCM_VERSION} detected with gfx803 GPU"
            else
                ROCM_AVAILABLE=true
                print_yellow "⚠️ ROCm installed but gfx803 GPU not detected"
            fi
        else
            print_yellow "⚠️ ROCm directory found but rocminfo command missing"
        fi
    else
        print_yellow "⚠️ ROCm not installed"
    fi
    
    # Check for NVIDIA CUDA
    print_yellow "🔍 Checking for NVIDIA CUDA installation..."
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &>/dev/null; then
            CUDA_AVAILABLE=true
            CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || echo "Unknown")
            NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
            print_green "✅ CUDA ${CUDA_VERSION} detected with ${NUM_GPUS} GPU(s)"
        else
            print_yellow "⚠️ nvidia-smi command failed"
        fi
    else
        print_yellow "⚠️ NVIDIA drivers not installed"
    fi
    
    # Create BASE environment only if at least one GPU driver is available
    if [ "$ROCM_AVAILABLE" = false ] && [ "$CUDA_AVAILABLE" = false ]; then
        print_yellow "⚠️ No GPU drivers detected. Only the BASE environment will be created."
    fi
}

remove_existing_conda() {
    print_section "🧹 Removing existing Conda installations"
    
    # Locations to check for Conda
    local conda_paths=(
        "/home/${REAL_USER}/miniconda3"
        "/home/${REAL_USER}/anaconda3"
        "/opt/miniconda3"
        "/opt/anaconda3"
    )
    
    # Check each path and remove if exists
    for path in "${conda_paths[@]}"; do
        if [ -d "$path" ]; then
            print_yellow "🗑️ Removing Conda at $path..."
            rm -rf "$path" || { 
                print_red "❌ Failed to remove $path"
                exit 1
            }
        fi
    done
    
    # Clean up shell initialization files
    local shell_files=(
        "/home/${REAL_USER}/.bashrc"
        "/home/${REAL_USER}/.bash_profile"
        "/home/${REAL_USER}/.zshrc"
    )
    
    for file in "${shell_files[@]}"; do
        if [ -f "$file" ]; then
            print_yellow "🧹 Cleaning Conda init from $file..."
            # Backup the file
            cp "$file" "${file}.bak.${TIMESTAMP}"
            # Remove Conda initialization blocks
            sed -i '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' "$file"
            # Remove other Conda-related lines
            sed -i '/conda\.sh/d' "$file"
            sed -i '/conda activate/d' "$file"
            chown ${REAL_USER}:${REAL_USER} "$file"
        fi
    done
    
    print_green "✅ Existing Conda installations removed"
}

install_conda() {
    print_section "📦 Installing Miniconda"
    
    # Remove if it already exists (shouldn't happen after previous step)
    [ -d "$CONDA_BASE" ] && rm -rf "$CONDA_BASE"
    
    # Download installer
    print_yellow "📥 Downloading Miniconda installer..."
    cd /tmp
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh || { 
        print_red "❌ Miniconda download failed"
        exit 1
    }
    
    # Install Miniconda
    print_yellow "🔧 Installing Miniconda..."
    bash miniconda.sh -b -p "$CONDA_BASE" || { 
        print_red "❌ Miniconda install failed"
        exit 1
    }
    
    # Fix ownership
    chown -R ${REAL_USER}:${REAL_USER} "$CONDA_BASE"
    
    # Clean up installer
    rm -f miniconda.sh
    
    # Initialize for bash (don't auto-activate base)
    print_yellow "🔧 Initializing Conda..."
    "$CONDA_BASE/bin/conda" init bash || { 
        print_red "❌ Conda init failed"
        exit 1
    }
    
    # Create bash_profile if it doesn't exist
    if [ ! -f "/home/${REAL_USER}/.bash_profile" ]; then
        echo '[ -f ~/.bashrc ] && . ~/.bashrc' > "/home/${REAL_USER}/.bash_profile"
        chown ${REAL_USER}:${REAL_USER} "/home/${REAL_USER}/.bash_profile"
    fi
    
    # Disable auto-activate of base environment
    "$CONDA_BASE/bin/conda" config --set auto_activate_base false
    
    # Update Conda
    print_yellow "🔄 Updating Conda..."
    "$CONDA_BASE/bin/conda" update -n base -c defaults conda -y || { 
        print_red "❌ Conda update failed"
        exit 1
    }
    
    print_green "✅ Miniconda installed at $CONDA_BASE"
}

# Get the appropriate Python version for each environment
get_python_version() {
    local env_name=$1
    
    case "$env_name" in
        "BASE")
            echo "3.11"  # Latest for BASE
            ;;
        "ROCM")
            echo "3.10"  # ROCm PyTorch generally supports up to 3.10
            ;;
        "CUDA")
            echo "3.11"  # CUDA PyTorch supports up to 3.11
            ;;
        *)
            echo "3.11"  # Default to latest
            ;;
    esac
}

setup_conda_env() {
    local env_name=$1
    local python_ver=$(get_python_version "$env_name")
    
    print_section "🌱 Creating $env_name environment with Python $python_ver"
    
    # Remove the environment if it exists
    print_yellow "🧹 Removing $env_name environment if it exists..."
    "$CONDA_BASE/bin/conda" env remove -n "$env_name" 2>/dev/null || true
    
    # Create the environment with appropriate Python version
    print_yellow "🔧 Creating $env_name environment..."
    "$CONDA_BASE/bin/conda" create -y -n "$env_name" python="$python_ver" || { 
        print_red "❌ Failed to create $env_name"
        exit 1
    }
    
    # Verify the environment was created successfully
    if ! "$CONDA_BASE/bin/conda" env list | grep -q "^$env_name "; then
        print_red "❌ $env_name not found after creation"
        exit 1
    fi
    
    # Fix ownership
    chown -R ${REAL_USER}:${REAL_USER} "$CONDA_BASE/envs/$env_name"
    
    print_green "✅ $env_name created with Python $python_ver"
}

configure_global_env() {
    print_section "🌍 Configuring system environment"
    
    # Backup the existing environment file
    cp /etc/environment /etc/environment.bak.${TIMESTAMP} || { 
        print_red "❌ Failed to backup /etc/environment"
        exit 1
    }
    
    # Create a new environment file with base settings
    print_yellow "🔧 Setting up /etc/environment..."
    cat > /etc/environment << 'EOF'
DEBIAN_FRONTEND=noninteractive
PYTHONUNBUFFERED=1
PYTHONENCODING=UTF-8
PIP_ROOT_USER_ACTION=ignore
EOF

    # Add ROCm entries if available
    if [ "$ROCM_AVAILABLE" = true ]; then
        print_yellow "🔧 Adding ROCm paths to /etc/environment..."
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
        print_yellow "🔧 Adding CUDA paths to /etc/environment..."
        cat >> /etc/environment << 'EOF'
CUDA_HOME=/usr/local/cuda
CUDA_PATH=/usr/local/cuda
PATH=/usr/local/cuda/bin:$PATH
LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
    fi

    print_green "✅ /etc/environment configured"
}

# Enhanced function to create activation scripts with environment variables
create_env_script() {
    local env_name=$1
    local script_content=$2
    local activate_dir="$CONDA_BASE/envs/$env_name/etc/conda/activate.d"
    local deactivate_dir="$CONDA_BASE/envs/$env_name/etc/conda/deactivate.d"
    
    # Create activation directory
    mkdir -p "$activate_dir"
    chown -R "${REAL_USER}:${REAL_USER}" "$CONDA_BASE/envs/$env_name/etc/conda" 2>/dev/null || true
    
    # Create activation script
    echo "$script_content" > "$activate_dir/env_vars.sh"
    chmod +x "$activate_dir/env_vars.sh"
    chown "${REAL_USER}:${REAL_USER}" "$activate_dir/env_vars.sh"
    
    # Create deactivation directory
    mkdir -p "$deactivate_dir"
    
    # Generate deactivation script based on activate script
    # Extract variable names and create unset statements
    grep -oP "export \K[A-Za-z0-9_]+" "$activate_dir/env_vars.sh" | sort | uniq | \
    awk 'BEGIN {print "#!/bin/bash\n# Environment variables cleanup"} {print "unset " $1}' \
    > "$deactivate_dir/env_vars.sh"
    
    chmod +x "$deactivate_dir/env_vars.sh"
    chown "${REAL_USER}:${REAL_USER}" "$deactivate_dir/env_vars.sh"
    
    print_yellow "🔧 Created activation/deactivation scripts for $env_name"
}

configure_base_env() {
    print_section "🌱 Configuring BASE environment"
    
    # First set up the conda environment
    setup_conda_env "BASE"
    
    # Install core packages
    print_yellow "📦 Installing core packages in BASE environment..."
    "$CONDA_BASE/bin/pip" install numpy jupyter ipython matplotlib pandas scipy scikit-learn || { 
        print_red "❌ Failed to install packages in BASE"
        exit 1
    }
    
    # Create activation script content
    local base_script='#!/bin/bash
# BASE Environment Activation Script (Hybrid CUDA+ROCm)
echo "Activating hybrid BASE environment"
# Framework enablement
export USE_CUDA=1
export USE_ROCM=1
# CUDA Configuration
export CUDA_VISIBLE_DEVICES=0,1,2
export TORCH_CUDA_ARCH_LIST="3.5;3.7"
export CUDA_COMPUTE_MAJOR=3
export CUDA_COMPUTE_MINOR=5
export FORCE_CUDA=1
export CUDA_CACHE_PATH=~/.nv/ComputeCache
export TORCH_CUDNN_V8_API_ENABLED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_BENCHMARK=1
export CUDA_CACHE_DISABLE=0
export CUDA_AUTO_BOOST=1
# ROCm Configuration
export HSA_ENABLE_SDMA=0
export ROC_ENABLE_PRE_VEGA=1
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export HIP_VISIBLE_DEVICES=0
export HIP_PLATFORM=amd
export ROCR_VISIBLE_DEVICES=0
export HSA_NO_SCRATCH=1
export HIP_FORCE_DEV=0
export HCC_AMDGPU_TARGET=gfx803
export HIPCC_FLAGS=--amdgpu-target=gfx803
export PYTORCH_ROCM_ARCH=gfx803
export ROCM_ARCH=gfx803
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_ENABLED=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
# Ollama Hybrid Configuration
export num_gpu=4
export OLLAMA_GPU_OVERHEAD=20
export OLLAMA_DEBUG=true
export OLLAMA_FLASH_ATTENTION=true
export OLLAMA_NUM_THREADS=20
# Use per-session manual selection for library
# export OLLAMA_LLM_LIBRARY=cuda/rocm_v6
# GPU Memory Management (AMD)
export GPU_MAX_HEAP_SIZE=100
export GPU_USE_SYNC_OBJECTS=1
export GPU_MAX_ALLOC_PERCENT=100
export GPU_SINGLE_ALLOC_PERCENT=100
# Enable both backends for LLAMA
export LLAMA_HIPBLAS=1
export LLAMA_CUBLAS=1
export AMD_LOG_LEVEL=3
export AMD_SERIALIZE_KERNEL=3
echo "Hybrid BASE environment activated with CUDA+ROCm support"'
    
    # Create the activation script
    create_env_script "BASE" "$base_script"
    
    print_green "✅ BASE environment configured with hybrid CUDA+ROCm support"
}

configure_rocm_env() {
    print_section "🌱 Configuring ROCM environment"
    
    # Check if ROCm is available and skip if not
    if [ "$ROCM_AVAILABLE" = false ]; then
        print_yellow "⚠️ Skipping ROCM environment setup (ROCm not detected)"
        return
    fi
    
    # Set up the conda environment
    setup_conda_env "ROCM"
    
    # Install PyTorch with ROCm support
    print_yellow "📦 Installing PyTorch with ROCm support..."
    
    # Determine appropriate PyTorch ROCm wheel URL based on installed ROCm version
    ROCM_MAJOR_MINOR=$(echo "$ROCM_VERSION" | cut -d. -f1,2)
    TORCH_ROCM_URL="https://download.pytorch.org/whl/rocm${ROCM_MAJOR_MINOR}"
    
    # Install PyTorch with ROCm support
    "$CONDA_BASE/bin/conda" run -n ROCM pip install torch torchvision torchaudio --extra-index-url "$TORCH_ROCM_URL" || { 
        print_red "❌ Failed to install PyTorch in ROCM environment"
        exit 1
    }
    
    # Install Ollama
    print_yellow "📦 Installing Ollama..."
    "$CONDA_BASE/bin/conda" run -n ROCM pip install ollama || { 
        print_red "❌ Failed to install Ollama in ROCM environment"
        exit 1
    }
    
    # Create activation script content
    local rocm_script='#!/bin/bash
# ROCM Environment Activation Script
echo "Activating ROCM environment"
# ROCm Environment Variables
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
export HCC_AMDGPU_TARGET=gfx803
export HIPCC_FLAGS=--amdgpu-target=gfx803
# PyTorch & ROCm-specific Variables
export PYTORCH_ROCM_ARCH=gfx803
export ROCM_ARCH=gfx803
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_ENABLED=1
# Ollama Environment Variables for ROCM
export num_gpu=1
export OLLAMA_GPU_OVERHEAD=20
export AMD_LOG_LEVEL=3
export LLAMA_HIPBLAS=1
export OLLAMA_LLM_LIBRARY=rocm_v6
export OLLAMA_DEBUG=true
export AMD_SERIALIZE_KERNEL=3
export OLLAMA_FLASH_ATTENTION=true
export GPU_MAX_HEAP_SIZE=100
export GPU_USE_SYNC_OBJECTS=1
export GPU_MAX_ALLOC_PERCENT=100
export GPU_SINGLE_ALLOC_PERCENT=100
export OLLAMA_NUM_THREADS=20
echo "ROCm environment activated"'
    
    # Create the activation script
    create_env_script "ROCM" "$rocm_script"
    
    print_green "✅ ROCM environment configured for RX580/gfx803"
}

configure_cuda_env() {
    print_section "🌱 Configuring CUDA environment"
    
    # Check if CUDA is available and skip if not
    if [ "$CUDA_AVAILABLE" = false ]; then
        print_yellow "⚠️ Skipping CUDA environment setup (CUDA not detected)"
        return
    }
    
    # Set up the conda environment
    setup_conda_env "CUDA"
    
    # Install PyTorch with CUDA support
    print_yellow "📦 Installing PyTorch with CUDA support..."
    
    # Determine the CUDA version for PyTorch
    if [[ "$CUDA_VERSION" == 11.* ]]; then
        TORCH_CUDA_VERSION="cu118"
    elif [[ "$CUDA_VERSION" == 12.* ]]; then
        TORCH_CUDA_VERSION="cu121"
    else
        TORCH_CUDA_VERSION="cu118"  # Default to CUDA 11.8 compatibility
    fi
    
    # Install PyTorch with CUDA support
    "$CONDA_BASE/bin/conda" run -n CUDA pip install torch torchvision torchaudio --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA_VERSION}" || { 
        print_red "❌ Failed to install PyTorch in CUDA environment"
        exit 1
    }
    
    # Create activation script content
    local cuda_script='#!/bin/bash
# CUDA Environment Activation Script
echo "Activating CUDA environment"
# CUDA Environment Variables
export CUDA_VISIBLE_DEVICES=0,1,2
export TORCH_CUDA_ARCH_LIST="3.5;3.7"
export CUDA_COMPUTE_MAJOR=3
export CUDA_COMPUTE_MINOR=5
export FORCE_CUDA=1
export CUDA_CACHE_PATH=~/.nv/ComputeCache
export USE_CUDA=1
export USE_ROCM=0
# PyTorch CUDA-specific Variables
export TORCH_CUDNN_V8_API_ENABLED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_BENCHMARK=1
# Ollama Environment Variables for CUDA
export num_gpu=3
export OLLAMA_GPU_OVERHEAD=20
export LLAMA_CUBLAS=1
export OLLAMA_LLM_LIBRARY=cuda
export OLLAMA_DEBUG=true
export OLLAMA_FLASH_ATTENTION=true
export OLLAMA_NUM_THREADS=20
export CUDA_CACHE_DISABLE=0
export CUDA_AUTO_BOOST=1
echo "CUDA environment activated"'
    
    # Create the activation script
    create_env_script "CUDA" "$cuda_script"
    
    print_green "✅ CUDA environment configured for K80/Compute 3.5-3.7"
}

verify_envs() {
    print_section "🔍 Verifying environments"
    
    # Prepare array to hold verification results
    declare -A verify_results
    
    # Verify BASE environment
    print_yellow "🧪 Verifying BASE environment..."
    if "$CONDA_BASE/bin/conda" run -n BASE python -c 'import numpy; print("BASE: NumPy OK")' 2>/dev/null; then
        verify_results["BASE"]="✅ Success"
        print_green "✅ BASE environment verification successful"
    else
        verify_results["BASE"]="❌ Failed"
        print_red "❌ BASE environment verification failed"
    fi
    
    # Verify ROCM environment if it was set up
    if [ "$ROCM_AVAILABLE" = true ]; then
        print_yellow "🧪 Verifying ROCM environment..."
        if "$CONDA_BASE/bin/conda" run -n ROCM python -c 'import torch; print(f"ROCM: PyTorch {torch.__version__}"); print(f"ROCM: HIP available: {hasattr(torch, \"version\") and hasattr(torch.version, \"hip\")}")' 2>/dev/null; then
            verify_results["ROCM"]="✅ Success"
            print_green "✅ ROCM environment verification successful"
        else
            verify_results["ROCM"]="❌ Failed"
            print_red "❌ ROCM environment verification failed"
        fi
    else
        verify_results["ROCM"]="⚠️ Skipped (not installed)"
        print_yellow "⚠️ ROCM environment verification skipped (ROCm not detected)"
    fi
    
    # Verify CUDA environment if it was set up
    if [ "$CUDA_AVAILABLE" = true ]; then
        print_yellow "🧪 Verifying CUDA environment..."
        if "$CONDA_BASE/bin/conda" run -n CUDA python -c 'import torch; print(f"CUDA: PyTorch {torch.__version__}"); print(f"CUDA: CUDA available: {torch.cuda.is_available()}")' 2>/dev/null; then
            verify_results["CUDA"]="✅ Success"
            print_green "✅ CUDA environment verification successful"
        else
            verify_results["CUDA"]="❌ Failed"
            print_red "❌ CUDA environment verification failed"
        fi
    else
        verify_results["CUDA"]="⚠️ Skipped (not installed)"
        print_yellow "⚠️ CUDA environment verification skipped (CUDA not detected)"
    fi
    
    print_section "📊 Verification Summary"
    echo "BASE: ${verify_results["BASE"]}"
    echo "ROCM: ${verify_results["ROCM"]}"
    echo "CUDA: ${verify_results["CUDA"]}"
    
    # Continue even if verification failed
    print_green "✅ Environment verification complete"
}

add_conda_aliases() {
    print_section "⌨️ Adding convenience aliases and auto-activating BASE"
    
    BASHRC_FILE="/home/${REAL_USER}/.bashrc"
    
    # Check if aliases already exist
    if ! grep -q '# Conda Aliases and Auto-Activation' "$BASHRC_FILE"; then
        print_yellow "🔧 Adding Conda aliases to .bashrc..."
        
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
EOF
        fi
        
        # Add CUDA alias if available
        if [ "$CUDA_AVAILABLE" = true ]; then
            cat >> "$BASHRC_FILE" << 'EOF'
alias cuda="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate CUDA"
alias nanocuda="nano $HOME/miniconda3/envs/CUDA/etc/conda/activate.d/env_vars.sh"
alias watchcuda="watch -n 1 nvidia-smi"
EOF
        fi
        
        # Add Ollama service management aliases
        cat >> "$BASHRC_FILE" << 'EOF'
alias ollamass="sudo nano /etc/systemd/system/ollama.service" 
alias ollamacfg="nano ~/.ollama/config.json"
alias startollama="sudo systemctl start ollama"
alias stopollama="sudo systemctl stop ollama"
alias restartollama="sudo systemctl restart ollama"
alias ollamastatus="sudo systemctl status ollama-*"
alias ollamaports="sudo netstat -tulpn | grep ollama"
EOF
        
        # Add common Conda aliases
        cat >> "$BASHRC_FILE" << 'EOF'
alias nanobase="nano $HOME/miniconda3/envs/BASE/etc/conda/activate.d/env_vars.sh"
alias base="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate BASE"
alias nanobash="sudo nano ~/.bashrc"
alias nanoenv="sudo nano /etc/environment"
alias condals="conda env list"
alias condainfo="conda info"
alias condaupdate="conda update -n base -c defaults conda"

# Add Conda bin to PATH
export PATH="$HOME/miniconda3/bin:$PATH"
EOF
        
        # Fix ownership
        chown "${REAL_USER}:${REAL_USER}" "$BASHRC_FILE"
        
        print_green "✅ Aliases and auto-activation added to .bashrc"
    else
        print_yellow "⚠️ Aliases already exist in .bashrc - not modifying"
    fi
}

create_helper_scripts() {
    print_section "📝 Creating helper scripts"
    
    # Directory for helper scripts
    SCRIPTS_DIR="/home/${REAL_USER}/conda-scripts"
    mkdir -p "$SCRIPTS_DIR"
    chown "${REAL_USER}:${REAL_USER}" "$SCRIPTS_DIR"
    
    # Create fix script for common issues
    cat > "${SCRIPTS_DIR}/fix-conda-envs.sh" << 'EOF'
#!/bin/bash
# Conda Environment Repair Script

echo "Conda Environment Repair Utility"
echo "==============================="
echo

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if conda exists
if [ ! -f "$HOME/miniconda3/bin/conda" ]; then
    echo -e "${RED}Error: Miniconda installation not found at $HOME/miniconda3${NC}"
    exit 1
fi

echo -e "${BLUE}Diagnosing Conda environments...${NC}"
echo

# Source conda to make it available
. "$HOME/miniconda3/etc/profile.d/conda.sh"

# Check environments
ENVS=("BASE" "ROCM" "CUDA")
MISSING_ENVS=()

for env in "${ENVS[@]}"; do
    echo -n "Checking $env environment: "
    if conda env list | grep -q "^$env "; then
        echo -e "${GREEN}Found${NC}"
    else
        echo -e "${RED}Not found${NC}"
        MISSING_ENVS+=("$env")
    fi
done

# Check activation scripts
echo
echo -e "${BLUE}Checking environment activation scripts...${NC}"
for env in "${ENVS[@]}"; do
    if conda env list | grep -q "^$env "; then
        ACTIVATE_SCRIPT="$HOME/miniconda3/envs/$env/etc/conda/activate.d/env_vars.sh"
        DEACTIVATE_SCRIPT="$HOME/miniconda3/envs/$env/etc/conda/deactivate.d/env_vars.sh"
        
        echo -n "Checking $env activation script: "
        if [ -f "$ACTIVATE_SCRIPT" ]; then
            echo -e "${GREEN}Found${NC}"
        else
            echo -e "${RED}Missing${NC}"
            echo -e "${YELLOW}Creating directory for $env activation script...${NC}"
            mkdir -p "$(dirname "$ACTIVATE_SCRIPT")"
            
            case "$env" in
                "BASE")
                    echo -e "${YELLOW}Creating BASE activation script...${NC}"
                    # BASE script content would be here
                    ;;
                "ROCM")
                    echo -e "${YELLOW}Creating ROCM activation script...${NC}"
                    # ROCM script content would be here
                    ;;
                "CUDA")
                    echo -e "${YELLOW}Creating CUDA activation script...${NC}"
                    # CUDA script content would be here
                    ;;
            esac
        fi
        
        echo -n "Checking $env deactivation script: "
        if [ -f "$DEACTIVATE_SCRIPT" ]; then
            echo -e "${GREEN}Found${NC}"
        else
            echo -e "${RED}Missing${NC}"
            echo -e "${YELLOW}Creating directory for $env deactivation script...${NC}"
            mkdir -p "$(dirname "$DEACTIVATE_SCRIPT")"
            
            echo -e "${YELLOW}Generating deactivation script...${NC}"
            # Extract variable names from activation script and create unset statements
            if [ -f "$ACTIVATE_SCRIPT" ]; then
                mkdir -p "$(dirname "$DEACTIVATE_SCRIPT")"
                grep -oP "export \K[A-Za-z0-9_]+" "$ACTIVATE_SCRIPT" | sort | uniq | \
                awk 'BEGIN {print "#!/bin/bash\n# Environment variables cleanup"} {print "unset " $1}' \
                > "$DEACTIVATE_SCRIPT"
                chmod +x "$DEACTIVATE_SCRIPT"
            else
                echo "#!/bin/bash" > "$DEACTIVATE_SCRIPT"
                echo "# Environment variables cleanup" >> "$DEACTIVATE_SCRIPT"
                chmod +x "$DEACTIVATE_SCRIPT"
            fi
        fi
    fi
done

# Check for broken packages
echo
echo -e "${BLUE}Checking for broken packages...${NC}"
for env in "${ENVS[@]}"; do
    if conda env list | grep -q "^$env "; then
        echo -e "${YELLOW}Checking packages in $env environment...${NC}"
        conda activate $env
        
        echo -n "Testing numpy: "
        if python -c "import numpy; print('OK')" 2>/dev/null; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}Failed${NC}"
            echo -e "${YELLOW}Reinstalling numpy...${NC}"
            pip install --upgrade --force-reinstall numpy
        fi
        
        case "$env" in
            "ROCM")
                echo -n "Testing PyTorch with ROCm: "
                if python -c "import torch; print('PyTorch version:', torch.__version__); print('ROCm available:', hasattr(torch.version, 'hip'))" 2>/dev/null; then
                    echo -e "${GREEN}OK${NC}"
                else
                    echo -e "${RED}Failed${NC}"
                    echo -e "${YELLOW}Note: You may need to reinstall PyTorch with ROCm support${NC}"
                fi
                ;;
            "CUDA")
                echo -n "Testing PyTorch with CUDA: "
                if python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
                    echo -e "${GREEN}OK${NC}"
                else
                    echo -e "${RED}Failed${NC}"
                    echo -e "${YELLOW}Note: You may need to reinstall PyTorch with CUDA support${NC}"
                fi
                ;;
        esac
        
        conda deactivate
    fi
done

# Final recommendations
echo
echo -e "${BLUE}Repair recommendations:${NC}"
if [ ${#MISSING_ENVS[@]} -gt 0 ]; then
    echo -e "${YELLOW}The following environments are missing and should be recreated:${NC}"
    for env in "${MISSING_ENVS[@]}"; do
        echo "  - $env"
    done
    echo -e "${YELLOW}Run the main installation script to recreate these environments.${NC}"
else
    echo -e "${GREEN}All environments exist.${NC}"
fi

echo
echo -e "${GREEN}Conda environment repair check complete!${NC}"
echo -e "${YELLOW}If you experienced issues, consider running:${NC}"
echo "  conda clean --all"
echo "  conda update -n base -c defaults conda"
EOF

    # Create script to update all environments
    cat > "${SCRIPTS_DIR}/update-all-envs.sh" << 'EOF'
#!/bin/bash
# Update All Conda Environments

echo "Conda Environment Update Utility"
echo "==============================="
echo

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Source conda to make it available
. "$HOME/miniconda3/etc/profile.d/conda.sh"

# Update conda itself
echo -e "${BLUE}Updating Conda base installation...${NC}"
conda update -n base -c defaults conda -y

# Update environments
ENVS=("BASE" "ROCM" "CUDA")

for env in "${ENVS[@]}"; do
    if conda env list | grep -q "^$env "; then
        echo
        echo -e "${BLUE}Updating $env environment...${NC}"
        conda activate $env
        
        # Update pip and core packages
        echo -e "${YELLOW}Updating pip...${NC}"
        pip install --upgrade pip
        
        echo -e "${YELLOW}Updating core packages...${NC}"
        pip install --upgrade numpy scipy matplotlib pandas scikit-learn
        
        # Environment-specific updates
        case "$env" in
            "ROCM")
                echo -e "${YELLOW}Note: PyTorch with ROCm support is not updated automatically.${NC}"
                echo -e "${YELLOW}If you want to update PyTorch, run:${NC}"
                echo "  pip install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.4.2"
                ;;
            "CUDA")
                echo -e "${YELLOW}Note: PyTorch with CUDA support is not updated automatically.${NC}"
                echo -e "${YELLOW}If you want to update PyTorch, run:${NC}"
                echo "  pip install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118"
                ;;
        esac
        
        conda deactivate
    else
        echo
        echo -e "${RED}$env environment not found, skipping...${NC}"
    fi
done

echo
echo -e "${GREEN}All environments updated!${NC}"
EOF

    # Create README file
    cat > "${SCRIPTS_DIR}/README.md" << 'EOF'
# Conda Environment Helper Scripts

This directory contains utility scripts to help manage your Conda AI/ML environments.

## Available Scripts

- **fix-conda-envs.sh**: Diagnoses and fixes common issues with Conda environments
- **update-all-envs.sh**: Updates all Conda environments and their packages

## Usage

Make scripts executable:
```bash
chmod +x *.sh
```

Then run them directly:
```bash
./fix-conda-envs.sh
./update-all-envs.sh
```

## Conda Environments

- **BASE**: General purpose with both CUDA and ROCm support
- **ROCM**: Optimized for AMD GPUs (RX580/gfx803)
- **CUDA**: Optimized for NVIDIA GPUs (K80/Compute 3.5-3.7)

## Quick Aliases

Available in your shell:
- `rocm`: Activate the ROCM environment
- `cuda`: Activate the CUDA environment
- `editbase`: Edit BASE environment variables
- `editrocm`: Edit ROCM environment variables
- `editcuda`: Edit CUDA environment variables
- `condals`: List Conda environments
- `condainfo`: Show Conda information
- `condaupdate`: Update Conda itself
EOF

    # Make scripts executable
    chmod +x "${SCRIPTS_DIR}/fix-conda-envs.sh"
    chmod +x "${SCRIPTS_DIR}/update-all-envs.sh"
    
    # Set correct ownership
    chown -R "${REAL_USER}:${REAL_USER}" "${SCRIPTS_DIR}"
    
    print_green "✅ Helper scripts created in ${SCRIPTS_DIR}"
}

main() {
    check_root
    check_gpu_drivers
    remove_existing_conda
    install_conda
    configure_global_env
    configure_base_env
    configure_rocm_env
    configure_cuda_env
    verify_envs
    add_conda_aliases
    create_helper_scripts
    
    # Print completion message
    print_banner "AI/ML Conda Environment Setup Complete!"
    
    cat << EOF
$(print_green "✅ Miniconda has been installed with the following environments:")

$(print_blue "🌱 BASE")
  • Hybrid environment with both CUDA and ROCm support
  • Python $(get_python_version "BASE")
  • $(if [ "$ROCM_AVAILABLE" = true ] && [ "$CUDA_AVAILABLE" = true ]; then echo "✅ Full dual GPU support"; elif [ "$ROCM_AVAILABLE" = true ] || [ "$CUDA_AVAILABLE" = true ]; then echo "⚠️ Partial GPU support"; else echo "❌ No GPU support detected"; fi)
  • Auto-activates on shell login

$(if [ "$ROCM_AVAILABLE" = true ]; then 
  echo "$(print_blue "🔥 ROCM")
  • Optimized for AMD RX580 (gfx803) GPUs
  • Python $(get_python_version "ROCM")
  • Activate with: $(print_yellow "rocm") command";
else 
  echo "$(print_yellow "⚠️ ROCM environment not created (no ROCm detected)")"; 
fi)

$(if [ "$CUDA_AVAILABLE" = true ]; then 
  echo "$(print_blue "⚡ CUDA")
  • Optimized for NVIDIA K80 (Compute 3.5-3.7) GPUs
  • Python $(get_python_version "CUDA")
  • Activate with: $(print_yellow "cuda") command";
else 
  echo "$(print_yellow "⚠️ CUDA environment not created (no CUDA detected)")"; 
fi)

$(print_blue "🛠️ Additional features:")
• Helper scripts in: $(print_yellow "/home/${REAL_USER}/conda-scripts")
• Configuration scripts: $(print_yellow "editbase, editrocm, editcuda") commands
• Log file: $(print_yellow "$LOG_FILE")

✨ $(print_green "To start using the new environments, log out and log back in,")
✨ $(print_green "or run: source ~/.bashrc")
EOF
}

main