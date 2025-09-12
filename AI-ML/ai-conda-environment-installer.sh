#!/bin/bash

# Conda Installation and Environment Setup Script
# Designed to remove existing Conda installations and set up fresh environments
# for AI/ML development with ROCm and CUDA support
set -e

echo "🚀 Starting Conda Installation and Environment Setup..."

# Check for existing Conda installations
check_existing_conda() {
    echo "🔍 Checking for existing Conda installations..."
    
    # Check common Conda installation locations
    CONDA_LOCATIONS=(
        "$HOME/anaconda3"
        "$HOME/miniconda3"
        "$HOME/.conda"
        "/opt/conda"
        "/usr/local/conda"
    )
    
    FOUND_CONDA=false
    for loc in "${CONDA_LOCATIONS[@]}"; do
        if [ -d "$loc" ]; then
            echo "Found Conda installation at: $loc"
            FOUND_CONDA=true
        fi
    done
    
    # Check for conda in PATH
    if command -v conda &>/dev/null; then
        echo "Found 'conda' command in PATH"
        FOUND_CONDA=true
    fi
    
    if [ "$FOUND_CONDA" = true ]; then
        read -p "⚠️ Existing Conda installation(s) found. Remove them? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            remove_existing_conda
        else
            echo "❌ Please remove existing Conda installations before continuing"
            exit 1
        fi
    fi
}

# Remove existing Conda installations
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
    
    # Remove Conda from shell configs
    for file in ~/.bashrc ~/.zshrc ~/.bash_profile ~/.profile; do
        if [ -f "$file" ]; then
            sed -i '/conda/d' "$file"
            sed -i '/miniconda3/d' "$file"
            sed -i '/anaconda3/d' "$file"
        fi
    done
    
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
    
    # Initialize Conda
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    
    # Disable auto activation of base environment
    conda config --set auto_activate_base false
    
    echo "✅ Miniconda installed successfully"
}

# Create and configure base environment
create_base_environment() {
    echo "🌱 Creating base environment (darklake)..."
    
    conda create -y -n darklake python=3.10
    
    conda activate darklake
    conda install -y \
        numpy \
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
        
    echo "✅ Base environment created successfully"
}

# Create and configure CUDA environment
create_cuda_environment() {
    echo "🎯 Creating CUDA environment (darklake-cuda)..."
    
    conda create -y -n darklake-cuda python=3.10
    
    conda activate darklake-cuda
    conda install -y \
        numpy \
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
        
    # Install CUDA-specific packages
    conda install -y -c pytorch \
        pytorch \
        torchvision \
        torchaudio \
        cudatoolkit
        
    conda install -y -c conda-forge \
        tensorflow-gpu \
        keras \
        jax \
        jaxlib \
        transformers \
        datasets \
        accelerate
        
    pip install --upgrade \
        lightning \
        bitsandbytes \
        deepspeed \
        diffusers
        
    echo "✅ CUDA environment created successfully"
}

# Create and configure ROCm environment
create_rocm_environment() {
    echo "🎯 Creating ROCm environment (darklake-rocm)..."
    
    conda create -y -n darklake-rocm python=3.10
    
    conda activate darklake-rocm
    conda install -y \
        numpy \
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
        
    # Install ROCm-specific packages
    pip install --upgrade \
        torch \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/rocm5.6
        
    pip install --upgrade \
        tensorflow-rocm \
        keras \
        jax \
        jaxlib \
        transformers \
        datasets \
        accelerate \
        lightning \
        bitsandbytes \
        deepspeed \
        diffusers
        
    echo "✅ ROCm environment created successfully"
}

# Create environment verification script
create_env_check_script() {
    echo "📝 Creating environment verification script..."
    
    cat << 'EOF' | sudo tee /usr/local/bin/check-conda-envs
#!/bin/bash
echo "Conda Environment Check"
echo "======================"
echo

# Check base environment
echo "Checking darklake environment..."
conda activate darklake
python -c "import numpy; import pandas; import scipy; print('Base packages OK')"
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
echo

# Check ROCm environment
echo "Checking darklake-rocm environment..."
conda activate darklake-rocm
python - << END
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available: {hasattr(torch, 'hip') and torch.hip.is_available()}")
END
echo

conda activate base
EOF

    sudo chmod +x /usr/local/bin/check-conda-envs
}

# Add Conda-related aliases
add_conda_aliases() {
    echo "📝 Adding Conda aliases..."
    
    # Remove any existing Conda aliases
    sed -i '/# Conda Aliases/,/^$/d' ~/.bashrc
    
    cat << 'EOF' >> ~/.bashrc

# Conda Aliases
alias conda-check='/usr/local/bin/check-conda-envs'
alias da='conda activate darklake'
alias dc='conda activate darklake-cuda'
alias dr='conda activate darklake-rocm'
alias dl='conda deactivate'
alias cl='conda list'
alias ce='conda env list'
EOF
}

# Main installation process
main() {
    check_existing_conda
    install_conda
    create_base_environment
    create_cuda_environment
    create_rocm_environment
    create_env_check_script
    add_conda_aliases
    
    echo "
✨ Conda Installation Complete! ✨

Installation Summary:
- Miniconda installed
- Three environments created:
  1. darklake (base ML environment)
  2. darklake-cuda (CUDA-enabled environment)
  3. darklake-rocm (ROCm-enabled environment)
- Environment verification script created
- Convenient aliases added

Available Commands:
1. conda-check : Run comprehensive environment verification
2. da         : Activate darklake environment
3. dc         : Activate darklake-cuda environment
4. dr         : Activate darklake-rocm environment
5. dl         : Deactivate current environment
6. cl         : List packages in current environment
7. ce         : List all conda environments

To verify installation:
1. Start a new terminal session or run: source ~/.bashrc
2. Run 'conda-check' for full environment verification
3. Try activating each environment:
   - da (darklake)
   - dc (darklake-cuda)
   - dr (darklake-rocm)

Note: Please open a new terminal session for all changes to take effect.
"
}

# Start installation
main