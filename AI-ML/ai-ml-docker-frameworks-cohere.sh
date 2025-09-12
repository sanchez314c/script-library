#!/bin/bash

# Unified AI/ML Environment Setup Script
# Combines Docker, Conda, CUDA, and ROCm support
# Created by Cortana for Jason
set -e

echo "🚀 Starting Unified AI/ML Environment Setup..."

# Save original PATH
ORIGINAL_PATH=$PATH

# Function to detect GPU hardware
detect_gpu() {
    echo "🔍 Detecting GPU hardware..."
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected"
        NVIDIA_GPU=true
    else
        NVIDIA_GPU=false
    fi

    if command -v rocminfo &> /dev/null; then
        echo "AMD GPU detected"
        AMD_GPU=true
    fi
}

# Function to remove existing Docker
remove_existing_docker() {
    echo "🧹 Removing existing Docker installations..."
    sudo systemctl stop docker || true
    sudo systemctl disable docker || true
    sudo apt-get remove -y docker docker-engine docker.io containerd runc || true
    sudo apt-get purge -y docker-ce docker-ce-cli containerd.io docker-compose-plugin || true
    sudo rm -rf /var/lib/docker || true
    sudo rm -rf /var/lib/containerd || true
}

# Function to install Docker
install_docker() {
    echo "🐋 Installing Docker..."

    # Remove any old versions
    remove_existing_docker

    # Install prerequisites
    sudo apt-get update
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

    # Set up stable repository
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io

    # Add user to docker group
    sudo usermod -aG docker $USER

    # Start and enable Docker service
    sudo systemctl start docker
    sudo systemctl enable docker
}

# Function to configure Docker GPU support
configure_docker_gpu() {
    echo "🎮 Configuring Docker GPU support..."

    if [ "$NVIDIA_GPU" = true ]; then
        # Install NVIDIA Container Toolkit using the new method
        echo "Installing NVIDIA Container Toolkit..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list

        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
    fi

    if [ "$AMD_GPU" = true ]; then
        # Configure for ROCm
        sudo mkdir -p /etc/docker
        echo '{
            "runtimes": {
                "rocm": {
                    "path": "/usr/bin/rocm-container-runtime",
                    "runtimeArgs": []
                }
            },
            "default-runtime": "runc",
            "features": {
                "buildkit": true
            }
        }' | sudo tee /etc/docker/daemon.json

        sudo systemctl restart docker
    fi
}

# Function to remove existing Conda installations
remove_existing_conda() {
    echo "🧹 Removing existing Conda installations..."

    # Remove conda from path
    if command -v conda &>/dev/null; then
        conda deactivate 2>/dev/null || true
        conda init --reverse --all 2>/dev/null || true
    fi

    # Remove common Conda directories
    rm -rf ~/anaconda3 2>/dev/null || true
    rm -rf ~/miniconda3 2>/dev/null || true
    rm -rf ~/.conda 2>/dev/null || true
    sudo rm -rf /opt/conda 2>/dev/null || true
    sudo rm -rf /usr/local/conda 2>/dev/null || true

    # Restore original PATH
    export PATH=$ORIGINAL_PATH

    # Clean bashrc of conda references
    sed -i '/^# >>> conda initialize >>>/,/^# <<< conda initialize <<</d' ~/.bashrc
    sed -i '/^export PATH=.*miniconda3.*$/d' ~/.bashrc
    sed -i '/^export PATH=.*anaconda3.*$/d' ~/.bashrc

    echo "✅ Removed existing Conda installations"
}

# Install Miniconda
install_conda() {
    echo "📦 Installing Miniconda..."

    # Download latest Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

    # Install Miniconda
    bash miniconda.sh -b -p "$HOME/miniconda3"
    rm miniconda.sh

    # Add conda to PATH
    export PATH="$HOME/miniconda3/bin:$PATH"

    # Initialize conda properly
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash

    # Disable auto activation of base environment
    conda config --set auto_activate_base false

    echo "✅ Miniconda installed successfully"

    # Source bashrc to ensure conda is available
    source ~/.bashrc
}

# Function to setup conda environment
setup_conda_env() {
    local env_name=$1
    echo "🌱 Creating environment ($env_name)..."

    # Ensure conda is in PATH and initialized
    export PATH="$HOME/miniconda3/bin:$PATH"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

    # Create environment
    conda create -y -n "$env_name" python=3.10

    # Activate environment
    conda activate "$env_name"

    # Install base packages
    conda install -y -c conda-forge \
        numpy=1.24.3 \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
        jupyter \
        ipython \
        tqdm \
        requests \
        pip
}

# Create and configure base environment (darklake)
create_base_environment() {
    setup_conda_env "darklake"

    # Install PyTorch and related packages
    pip install torch torchvision torchaudio
    pip install tensorflow
    pip install transformers datasets accelerate
    pip install lightning bitsandbytes deepspeed diffusers

    # Install dependencies with compatible versions
    pip install -U numpy==1.24.3
    pip install "scipy>=1.8.1"

    # Add environment variables
    mkdir -p "$HOME/.conda/envs/darklake/etc/conda/activate.d"
    cat << 'EOF' > "$HOME/.conda/envs/darklake/etc/conda/activate.d/env_vars.sh"
#!/bin/bash
# ROCm Environment Variables
export PATH=$PATH:/opt/rocm-6.3.3/bin
export LD_LIBRARY_PATH=/opt/rocm-6.3.3/lib:/opt/rocm-6.3.3/lib64:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export ROC_ENABLE_PRE_VEGA=1
export PYTORCH_ROCM_ARCH=gfx803
export HIP_VISIBLE_DEVICES=0
export HSA_ENABLE_SDMA=0

# CUDA Environment Variables
export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-11.4
export CUDA_PATH=/usr/local/cuda-11.4
export CUDA_VERSION_OVERRIDE=11.4
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_ARCH="35 37 50 52"
export CUDA_COMPUTE_MAJOR=3
export CUDA_COMPUTE_MINOR=5
EOF
    chmod +x "$HOME/.conda/envs/darklake/etc/conda/activate.d/env_vars.sh"

    echo "✅ Base environment created successfully"
}

# Create and configure CUDA environment
create_cuda_environment() {
    setup_conda_env "darklake-cuda"

    # Install CUDA-specific packages
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
    pip install tensorflow[gpu]
    pip install transformers datasets accelerate
    pip install lightning bitsandbytes deepspeed diffusers

    # Install dependencies with compatible versions
    pip install -U numpy==1.24.3
    pip install "scipy>=1.8.1"

    # Add CUDA-specific environment variables
    mkdir -p "$HOME/.conda/envs/darklake-cuda/etc/conda/activate.d"
    cat << 'EOF' > "$HOME/.conda/envs/darklake-cuda/etc/conda/activate.d/env_vars.sh"
#!/bin/bash
export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-11.4
export CUDA_PATH=/usr/local/cuda-11.4
export CUDA_VERSION_OVERRIDE=11.4
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_ARCH="35 37 50 52"
export CUDA_COMPUTE_MAJOR=3
export CUDA_COMPUTE_MINOR=5
EOF
    chmod +x "$HOME/.conda/envs/darklake-cuda/etc/conda/activate.d/env_vars.sh"

    echo "✅ CUDA environment created successfully"
}

# Create and configure ROCm environment
create_rocm_environment() {
    setup_conda_env "darklake-rocm"

    # Install ROCm PyTorch dependencies
    conda install -y -c conda-forge \
        hip-runtime-amd \
        hipblas \
        rocsparse \
        rocrand

    # Install PyTorch with ROCm support
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.7
    pip install tensorflow-rocm
    pip install transformers datasets accelerate
    pip install lightning bitsandbytes deepspeed diffusers

    # Install dependencies with compatible versions
    pip install -U numpy==1.24.3
    pip install "scipy>=1.8.1"

    # Add ROCm-specific environment variables
    mkdir -p "$HOME/.conda/envs/darklake-rocm/etc/conda/activate.d"
    cat << 'EOF' > "$HOME/.conda/envs/darklake-rocm/etc/conda/activate.d/env_vars.sh"
#!/bin/bash
export PATH=$PATH:/opt/rocm-6.3.3/bin
export LD_LIBRARY_PATH=/opt/rocm-6.3.3/lib:/opt/rocm-6.3.3/lib64:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export ROC_ENABLE_PRE_VEGA=1
export PYTORCH_ROCM_ARCH=gfx803
export HIP_VISIBLE_DEVICES=0
export HSA_ENABLE_SDMA=0
EOF
    chmod +x "$HOME/.conda/envs/darklake-rocm/etc/conda/activate.d/env_vars.sh"

    echo "✅ ROCm environment created successfully"
}

# Function to pull ML containers
pull_ml_containers() {
    echo "📥 Pulling ML containers..."

    # Pull base containers for Jupyter
    docker pull jupyter/base-notebook:latest
    docker pull jupyter/scipy-notebook:latest

    if [ "$NVIDIA_GPU" = true ]; then
        # Latest CUDA images for Ubuntu
        docker pull nvidia/cuda:12.8.0-base-ubuntu22.04
        docker pull nvidia/cuda:12.8.0-runtime-ubuntu22.04
    fi

    if [ "$AMD_GPU" = true ]; then
        # Latest ROCm images
        docker pull rocm/pytorch:latest
        docker pull rocm/dev-ubuntu-22.04:latest
    fi
}

# Create environment verification script
create_env_check_script() {
    echo "📝 Creating environment verification script..."

    cat << 'EOF' | sudo tee /usr/local/bin/check-conda-envs
#!/bin/bash
echo "Conda Environment Check"
echo "======================"
echo

# Ensure conda is available
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Check base environment
echo "Checking darklake environment..."
conda activate darklake
python -c "import numpy; import pandas; import scipy; import torch; print('Base packages OK')"
echo "Environment variables:"
env | grep -E "CUDA|ROC|HSA|HIP|PYTORCH"
echo

# Check CUDA environment
echo "Checking darklake-cuda environment..."
conda activate darklake-cuda
python - << END
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
END
echo "Environment variables:"
env | grep -E "CUDA"
echo

# Check ROCm environment
echo "Checking darklake-rocm environment..."
conda activate darklake-rocm
python - << END
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available: {hasattr(torch, 'hip') and torch.hip.is_available()}")
END
echo "Environment variables:"
env | grep -E "ROC|HSA|HIP|PYTORCH"
echo

conda activate darklake
EOF

    sudo chmod +x /usr/local/bin/check-conda-envs
}

# Add Conda-related aliases and default environment
add_conda_aliases() {
    echo "📝 Adding Conda aliases and setting default environment..."

    # Only add aliases if they don't exist
    if ! grep -q "# Conda Aliases" ~/.bashrc; then
        cat << 'EOF' >> ~/.bashrc

# Conda Aliases
alias conda-check='/usr/local/bin/check-conda-envs'
alias da='conda activate darklake'
alias dc='conda activate darklake-cuda'
alias dr='conda activate darklake-rocm'
alias dl='conda deactivate'
alias cl='conda list'
alias ce='conda env list'

# Activate darklake environment by default
conda activate darklake
EOF
    fi
}

# Main installation process
main() {
    detect_gpu
    remove_existing_docker
    remove_existing_conda
    install_docker
    install_conda
    create_base_environment
    create_cuda_environment
    create_rocm_environment
    configure_docker_gpu
    pull_ml_containers
    create_env_check_script
    add_conda_aliases

    echo "
✨ Unified AI/ML Environment Setup Complete! ✨

Installation Summary:

Docker Engine installed and configured with GPU support
Miniconda installed with three environments:
- darklake (base ML environment with all variables)
- darklake-cuda (CUDA-enabled environment)
- darklake-rocm (ROCm-enabled environment)
Environment verification script created
Convenient aliases added
Default environment set to darklake

Available Commands:

conda-check : Run comprehensive environment verification
da : Activate darklake environment
dc : Activate darklake-cuda environment
dr : Activate darklake-rocm environment
dl : Deactivate current environment
cl : List packages in current environment
ce : List all conda environments

Next Steps:
1. Log out and back in for Docker group changes to take effect
2. Run 'conda-check' to verify all environments
3. Use 'da', 'dc', or 'dr' to switch between environments

Note: Some features may require system restart to fully activate.

For support, contact Jason's AI assistant Cortana.
"

    # Final cleanup
    source ~/.bashrc
}

# Execute main installation
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo privileges"
    exit 1
fi

main "$@"