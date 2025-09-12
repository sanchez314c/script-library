#!/bin/bash

# Unified ML Docker Environment Setup Script
# Created by Cortana for Jason
# Supports both NVIDIA CUDA and AMD ROCm, Conda, and PyTorch
set -e

echo "🚀 Starting Unified ML Environment Setup..."

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
    else
        AMD_GPU=false
    fi
}

# Function to install Conda
install_conda() {
    echo "🐍 Installing Miniconda..."
    
    # Download Miniconda installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    
    # Install Miniconda
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    
    # Initialize conda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init bash
    
    # Create base ML environment
    echo "🌟 Creating base ML environment..."
    conda create -y -n ml-base python=3.10
    
    # Activate and setup base environment
    conda activate ml-base
    
    # Install PyTorch based on GPU availability
    if [ "$NVIDIA_GPU" = true ]; then
        echo "Installing PyTorch with CUDA support..."
        conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    elif [ "$AMD_GPU" = true ]; then
        echo "Installing PyTorch with ROCm support..."
        conda install -y pytorch torchvision torchaudio rocm-pytorch -c pytorch -c amd
    else
        echo "Installing PyTorch CPU version..."
        conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
    fi
    
    # Install common ML packages
    conda install -y \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        jupyter \
        ipython
    
    # Create environment file
    conda env export > $HOME/ml-workspace/environment.yml
}

[Previous Docker-related functions remain unchanged...]

# Update workspace setup function to include Conda test
setup_workspace() {
    echo "📂 Setting up ML workspace and test scripts..."
    
    mkdir -p $HOME/ml-workspace
    
    # Add Conda environment test
    cat << 'EOF' > $HOME/ml-workspace/test_conda_env.py
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
import sklearn

print("Python Environment Test")
print("-" * 30)
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")

if torch.cuda.is_available():
    print("\nGPU Information:")
    print(f"CUDA available: Yes")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    x = torch.rand(5, 3).cuda()
    print("GPU tensor test successful!")
else:
    print("\nNo CUDA GPU available")
EOF

    [Previous test scripts remain unchanged...]
}

# Update aliases function to include Conda shortcuts
add_aliases() {
    echo "🔧 Adding convenience aliases..."
    
    cat << 'EOF' >> ~/.bashrc

# Conda Environment Aliases
alias activate-ml='conda activate ml-base'
alias test-conda='python3 $HOME/ml-workspace/test_conda_env.py'
alias list-envs='conda env list'
alias update-env='conda env update -f $HOME/ml-workspace/environment.yml'

[Previous aliases remain unchanged...]
EOF
}

# Update main execution
main() {
    detect_gpu
    install_conda
    install_docker
    configure_docker_gpu
    pull_ml_containers
    create_launch_scripts
    setup_workspace
    add_aliases
    
    echo "
✨ Unified ML Environment Setup Complete! ✨

Installation Summary:
- Miniconda with ml-base environment
- PyTorch installed with GPU support (if available)
- Common ML packages (NumPy, Pandas, etc.)
- Docker Engine and containers
- ML workspace directory at $HOME/ml-workspace

Additional Commands:
1. activate-ml    : Activate the ML base environment
2. test-conda    : Test Conda environment setup
3. list-envs     : List all Conda environments
4. update-env    : Update environment from yml file

[Previous instructions remain unchanged...]
"

    # Apply new group memberships
    echo "🔄 Applying new group memberships..."
    exec newgrp docker
}

# Start installation
main