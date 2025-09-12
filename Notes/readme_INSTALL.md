##################################################################
######################### ESSENTIALS #############################
##################################################################

sudo apt update && sudo apt upgrade -y

sudo apt install -y git git-lfs curl gnupg software-properties-common build-essential

sudo apt install -y libstdc++-12-dev libtcmalloc-minimal4 nvtop radeontop rovclock libopenmpi3 libdnnl-dev ninja-build libopenblas-dev libpng-dev libjpeg-dev

##################################################################
######################## ROCm 6.3.4 ##############################
##################################################################

sudo apt install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"

sudo apt install -y python3-setuptools python3-wheel

sudo apt update

wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb

sudo apt install ./amdgpu-install_6.3.60303-1_all.deb

sudo apt update

sudo apt upgrade

sudo amdgpu-install --usecase=graphics,rocm,rocmdev,rocmdevtools,lrt,opencl,openclsdk,hip,hiplibsdk,mllib,mlsdk -y

sudo tee -a /etc/ld.so.conf.d/rocm.conf <<EOF
/opt/rocm-6.3.3/lib
/opt/rocm-6.3.3/lib64
EOF
sudo ldconfig

sudo /opt/rocm/bin/rocminfo | grep gfx

sudo adduser `whoami` video
sudo adduser `whoami` render

newgrp video
newgrp render
rocminfo

sudo ln -sf /opt/rocm-6.3.3 /opt/rocm

sudo reboot

##################################################################
######################## ROCBLAS 803 #############################
##################################################################

wget https://github.com/ROCm/rocBLAS/archive/refs/tags/rocm-6.3.0.zip
unzip rocm-6.3.0.zip
cd rocBLAS-rocm-6.3.0
./install.sh --cmake_install -ida gfx803

##################################################################
######################## NVIDIA 470 & CUDA #######################
##################################################################

# Add NVIDIA PPA
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update

# Install kernel headers
sudo apt install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"

# Install NVIDIA 470 and CUDA
sudo apt install nvidia-driver-470
sudo apt install -y nvidia-utils-470
sudo apt install -y nvidia-cuda-toolkit

# Verify
nvidia-smi
nvcc --version

##################################################################
######################## CLONE OLLAMA ############################
##################################################################

git clone https://github.com/ollama/ollama.git
git clone https://github.com/likelovewant/ollama-for-amd.git
git clone https://github.com/austinksmith/ollama37.git (Legacy CUDA)

##################################################################
####################### COMPILE OLLAMA ###########################
##################################################################

wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.tar.gz
tar -xf cmake-3.26.4-linux-x86_64.tar.gz
sudo mv cmake-3.26.4-linux-x86_64 /opt/cmake
sudo ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake

##################################################################
######################### AMD OLLAMA #############################
##################################################################

cmake -B build -DAMDGPU_TARGETS=gfx803 && cmake --build build -- -j18
    
##################################################################
######################### CUDA OLLAMA ############################
##################################################################

cmake -B build -DCMAKE_CUDA_ARCHITECTURES="37;86" && cmake --build build -- -j18

##################################################################
########################## ROCM+CUDA #############################
##################################################################

cmake -B build -DCMAKE_CXX_COMPILER=/opt/rocm-6.3.4/llvm/bin/amdclang++ -DHIP_PATH=/opt/rocm-6.3.4 -DAMDGPU_TARGETS=gfx803 && cmake --build build -- -j18 -verbose

##################################################################
######################## BUILD OLLAMA ############################
##################################################################

sudo snap install go --classic

go generate -tags rocm ./... &&  go build

##################################################################
######################## SERVE OLLAMA ############################
##################################################################

DRI_PRIME=1 VERBOSE=1 OLLAMA_DEBUG=1 OLLAMA_SCHED_SPREAD=0 HSA_OVERRIDE_GFX_VERSION=8.0.3 ollama serve &

##################################################################
######################### RUN OLLAMA #@###########################
##################################################################

DRI_PRIME=1 VERBOSE=1 OLLAMA_DEBUG=1 OLLAMA_SCHED_SPREAD=0 HSA_OVERRIDE_GFX_VERSION=8.0.3 ollama run --verbose  
