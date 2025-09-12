#!/bin/bash

# K80 CUDA & ROCm Setup Script - Updated for Current Repository Structure
# Created by Cortana for Jason
set -e

echo "🚀 Starting CUDA & ROCm Setup Process..."

# Install essential dependencies
sudo apt-get update && sudo apt-get install -y \
    git \
    golang \
    build-essential \
    cmake \
    wget \
    curl \
    software-properties-common

# Install ROCm components
echo "🔧 Installing ROCm components..."
wget https://repo.radeon.com/amdgpu-install/6.3.2/ubuntu/noble/amdgpu-install_6.3.60302-1_all.deb
sudo apt install ./amdgpu-install_6.3.60302-1_all.deb
sudo apt update

echo "📦 Installing ROCm use cases..."
amdgpu-install --usecase=graphics,rocm,rocmdev,rocmdevtools,amf,lrt,opencl,openclsdk,hip,hiplibsdk,mllib,mlsdk

# Clean up any existing CUDA repository entries
echo "🧹 Cleaning up existing CUDA repository entries..."
sudo rm -f /etc/apt/sources.list.d/cuda*
sudo rm -f /etc/apt/sources.list.d/archive_uri-https_developer_download_nvidia_com_compute_cuda*

# Add CUDA repository (without installing drivers)
echo "📦 Adding CUDA repository..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Updated key installation
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg

# Add repository with signed-by option
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda-ubuntu2004-x86_64.list

sudo apt-get update

# Pin current NVIDIA driver
echo "📌 Pinning current NVIDIA driver..."
sudo apt-mark hold nvidia-driver-470

# First install CUDNN base package
echo "🛠️ Installing CUDNN base package..."
sudo apt-get install -y --no-install-recommends libcudnn8=8.2.4.15-1+cuda11.4

# Then install CUDA development tools
echo "🛠️ Installing CUDA development tools..."
sudo apt-get install -y --no-install-recommends \
    cuda-compiler-11-4 \
    cuda-cudart-dev-11-4 \
    cuda-nvcc-11-4 \
    libcudnn8-dev=8.2.4.15-1+cuda11.4

# Set up persistent environment variables
echo "🔧 Setting up persistent environment variables..."

# Add CUDA and ROCm environment variables to /etc/environment
echo "📝 Adding CUDA and ROCm environment variables to /etc/environment..."
if ! grep -q "CUDA_HOME=/usr/local/cuda-11.4" /etc/environment; then
    echo "CUDA_HOME=/usr/local/cuda-11.4" | sudo tee -a /etc/environment
    echo "PATH=/usr/local/cuda-11.4/bin:/opt/rocm-6.3.2/bin:$PATH" | sudo tee -a /etc/environment
    echo "LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/opt/rocm-6.3.2/lib:$LD_LIBRARY_PATH" | sudo tee -a /etc/environment
    echo "HSA_OVERRIDE_GFX_VERSION=8.0.3" | sudo tee -a /etc/environment
    echo "ROC_ENABLE_PRE_VEGA=1" | sudo tee -a /etc/environment
    echo "PYTORCH_ROCM_ARCH=gfx803" | sudo tee -a /etc/environment
    echo "HIP_VISIBLE_DEVICES=0" | sudo tee -a /etc/environment
    echo "CUDA_VISIBLE_DEVICES=0" | sudo tee -a /etc/environment
    echo "CUDA_CACHE_DISABLE=0" | sudo tee -a /etc/environment
    echo "CUDA_AUTO_BOOST=1" | sudo tee -a /etc/environment
    echo "HSA_ENABLE_SDMA=0" | sudo tee -a /etc/environment
fi

# Create cuda.conf and rocm.conf in ld.so.conf.d
echo "📝 Creating persistent library configurations..."
sudo tee /etc/ld.so.conf.d/cuda.conf << 'EOL'
/usr/local/cuda-11.4/lib64
EOL

sudo tee /etc/ld.so.conf.d/rocm.conf << 'EOL'
/opt/rocm/lib
/opt/rocm/lib64
EOL

sudo ldconfig

# Add user to video and render groups for ROCm
echo "👥 Adding user to video and render groups..."
sudo usermod -a -G video $LOGNAME
sudo usermod -a -G render $LOGNAME

# Also add to profile.d for shell sessions
echo "📝 Setting up CUDA and ROCm profile script..."
sudo tee /etc/profile.d/cuda-rocm.sh << 'EOL'
export CUDA_HOME=/usr/local/cuda-11.4
export PATH=/usr/local/cuda-11.4/bin:/opt/rocm-6.3.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/opt/rocm-6.3.2/lib:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export ROC_ENABLE_PRE_VEGA=1
export PYTORCH_ROCM_ARCH=gfx803
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_DISABLE=0
export CUDA_AUTO_BOOST=1
export HSA_ENABLE_SDMA=0
EOL

# Make cuda-rocm.sh executable
sudo chmod +x /etc/profile.d/cuda-rocm.sh

# Source for current session
source /etc/profile.d/cuda-rocm.sh

# Create persistent symlinks
echo "🔗 Creating persistent CUDA symlinks..."
sudo ln -sf /usr/local/cuda-11.4/bin/nvcc /usr/local/bin/nvcc
sudo ln -sf /usr/local/cuda-11.4 /usr/local/cuda

# Verify CUDA Installation
echo "🔍 Verifying CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA compiler (nvcc) not found! Please ensure CUDA 11.4 is properly installed."
    exit 1
fi

echo "📊 CUDA Compiler Version:"
nvcc --version

echo "📊 NVIDIA Driver Information:"
if ! nvidia-smi; then
    echo "❌ nvidia-smi failed! Please check NVIDIA driver installation."
    exit 1
fi

echo "📊 CUDA Libraries:"
ldconfig -p | grep cuda
if [ $? -ne 0 ]; then
    echo "⚠️ No CUDA libraries found in ldconfig. This might cause issues."
fi

# Verify ROCm Installation
echo "🔍 Verifying ROCm installation..."
if [ -d "/opt/rocm-6.3.2" ]; then
    echo "✅ ROCm 6.3.2 installation found!"
    echo "📊 ROCm Version Information:"
    /opt/rocm-6.3.2/bin/rocminfo 2>/dev/null || echo "⚠️ rocminfo not available"
else
    echo "⚠️ ROCm installation not found in expected location"
fi

# Add persistent environment variables to .bashrc
echo "📝 Adding persistent environment variables to .bashrc..."
cat << 'EOL' >> ~/.bashrc

# CUDA and ROCm Environment Variables
export CUDA_HOME=/usr/local/cuda-11.4
export PATH=/usr/local/cuda-11.4/bin:/opt/rocm-6.3.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/opt/rocm-6.3.2/lib:$LD_LIBRARY_PATH

# GPU Compatibility Environment Variables
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export ROC_ENABLE_PRE_VEGA=1
export PYTORCH_ROCM_ARCH=gfx803
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_DISABLE=0
export CUDA_AUTO_BOOST=1
export HSA_ENABLE_SDMA=0

# CUDA and ROCm Aliases
alias nvidia-info='nvidia-smi'
alias rocm-info='rocminfo'
alias gpu-check='nvidia-smi && echo "---" && rocminfo'
EOL

# Source .bashrc for current session
source ~/.bashrc

echo "✨ CUDA and ROCm Setup Complete!"

# Final persistence check
echo "🔍 Verifying persistent configurations..."
if [ -f "/etc/profile.d/cuda-rocm.sh" ] && \
   [ -f "/etc/ld.so.conf.d/cuda.conf" ] && \
   [ -f "/etc/ld.so.conf.d/rocm.conf" ] && \
   [ -L "/usr/local/cuda" ]; then
    echo "✅ All persistent configurations are in place!"
else
    echo "⚠️ Some persistent configurations may be missing. Please check the setup."
fi

# Print final configuration status
echo "
🎉 Installation Complete! 

Environment Variables Set:
- CUDA_HOME=/usr/local/cuda-11.4
- PATH includes CUDA and ROCm bins
- LD_LIBRARY_PATH includes CUDA and ROCm libs
- HSA_OVERRIDE_GFX_VERSION=8.0.3
- ROC_ENABLE_PRE_VEGA=1
- PYTORCH_ROCM_ARCH=gfx803
- And other GPU compatibility variables

User Groups Added:
- video
- render

Persistence Configured In:
- /etc/environment
- /etc/profile.d/cuda-rocm.sh
- ~/.bashrc
- /etc/ld.so.conf.d/

Verify installations with:
1. CUDA: nvidia-smi
2. ROCm: rocminfo
3. Environment: echo \$PATH
4. Quick Check: gpu-check (alias added to .bashrc)
"

# Apply new group memberships without requiring logout
echo "🔄 Applying new group memberships..."
exec newgrp video << 'EONG'
newgrp render
EONG