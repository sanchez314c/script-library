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

# Function to install Docker
install_docker() {
    echo "🐋 Installing Docker..."
    
    # Remove any old versions
    sudo apt-get remove -y docker docker-engine docker.io containerd runc || true
    
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
        # Install NVIDIA Container Toolkit
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

        sudo apt-get update
        sudo apt-get install -y nvidia-docker2
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
            }
        }' | sudo tee /etc/docker/daemon.json
        sudo systemctl restart docker
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

# Function to pull ML containers
pull_ml_containers() {
    echo "📥 Pulling ML containers..."
    
    # Pull base containers
    docker pull jupyter/base-notebook:latest
    docker pull jupyter/scipy-notebook:latest
    
    if [ "$NVIDIA_GPU" = true ]; then
        docker pull nvidia/cuda:12.1.0-base-ubuntu22.04
    fi
    
    if [ "$AMD_GPU" = true ]; then
        docker pull rocm/pytorch:latest
    fi
}

# Function to create launch scripts
create_launch_scripts() {
    echo "📜 Creating launch scripts..."
    
    mkdir -p $HOME/ml-workspace/scripts
    
    # Create Jupyter launch script
    cat << 'EOF' > $HOME/ml-workspace/scripts/launch_jupyter.sh
#!/bin/bash
docker run -it --rm \
    -p 8888:8888 \
    -v "$HOME/ml-workspace:/home/jovyan/work" \
    jupyter/scipy-notebook:latest
EOF
    
    chmod +x $HOME/ml-workspace/scripts/launch_jupyter.sh
}

# Function to setup workspace
setup_workspace() {
    echo "📂 Setting up ML workspace and test scripts..."
    
    mkdir -p $HOME/ml-workspace
    
    # Create Conda environment test
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

    # Create Docker test script
    cat << 'EOF' > $HOME/ml-workspace/test_docker.py
import torch
import numpy as np
print("Docker Container Test")
print("-" * 30)
print("NumPy version:", np.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF
}

# Function to add aliases
add_aliases() {
    echo "🔧 Adding convenience aliases..."
    
    cat << 'EOF' >> ~/.bashrc

# ML Environment Aliases
alias activate-ml='conda activate ml-base'
alias test-conda='python3 $HOME/ml-workspace/test_conda_env.py'
alias list-envs='conda env list'
alias update-env='conda env update -f $HOME/ml-workspace/environment.yml'
alias jupyter-start='$HOME/ml-workspace/scripts/launch_jupyter.sh'
alias docker-test='docker run --rm python:3.10-slim python3 -c "import numpy; print(numpy.__version__)"'
EOF
}

# Main execution
main() {
    detect_gpu
    install_docker
    install_conda
    configure_docker_gpu
    pull_ml_containers
    create_launch_scripts
    setup_workspace
    add_aliases
    
    echo "
✨ Unified ML Environment Setup Complete! ✨

Installation Summary:
- Docker Engine installed and configured
- Miniconda with ml-base environment
- PyTorch installed with GPU support (if available)
- Common ML packages (NumPy, Pandas, etc.)
- ML workspace directory at $HOME/ml-workspace

Available Commands:
1. activate-ml     : Activate the ML base environment
2. test-conda     : Test Conda environment setup
3. list-envs      : List all Conda environments
4. update-env     : Update environment from yml file
5. jupyter-start  : Launch Jupyter notebook server
6. docker-test    : Test Docker Python environment

Next Steps:
1. Log out and back in for group changes to take effect
2. Run 'activate-ml' to enter the ML environment
3. Run 'test-conda' to verify the installation
4. Use 'jupyter-start' to begin working with notebooks

For GPU Support:
- NVIDIA GPU: CUDA and PyTorch GPU support configured
- AMD GPU: ROCm and PyTorch GPU support configured
"

    # Apply new group memberships
    echo "🔄 Applying new group memberships..."
    exec newgrp docker
}

# Start installation
main