heathen-admin@LLM:~$ history
    1  snap connect nordpass:password-manager-service
    2  sudo bash '/media/heathen-admin/BLUE USB/Core/ubuntu-application-installer.sh' 
    3  rocminfo
    4  sudo ln -sf /opt/rocm-6.3.3 /opt/rocm
    5  sudo reboot
    6  sudo apt update && sudo apt upgrade -y
    7  sudo apt install -y git git-lfs curl gnupg software-properties-common build-essential
    8  sudo apt install -y libstdc++-12-dev libtcmalloc-minimal4 nvtop radeontop rovclock libopenmpi3 libdnnl-dev ninja-build libopenblas-dev libpng-dev libjpeg-dev
    9  sudo apt install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
   10  sudo apt install -y python3-setuptools python3-wheel
   11  sudo apt update
   12  sudo apt upgrade
   13  wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb
   14  sudo apt install ./amdgpu-install_6.3.60303-1_all.deb
   15  sudo apt update
   16  sudo amdgpu-install --usecase=graphics,rocm,rocmdev,rocmdevtools,lrt,opencl,openclsdk,asan,hip,hiplibsdk,mllib,mlsdk -y
   17  echo 'export PATH=$PATH:/opt/rocm-6.3.3/bin' >> ~/.bashrc
   18  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm-6.3.3/lib' >> ~/.bashrc
   19  echo 'export ROCM_PATH=/opt/rocm-6.3.3' >> ~/.bashrc
   20  echo 'export HIP_PATH=/opt/rocm-6.3.3' >> ~/.bashrc
   21  echo 'export CXX=/opt/rocm-6.3.3/bin/amdclang++' >> ~/.bashrc
   22  echo 'export HSA_OVERRIDE_GFX_VERSION=8.0.3' >> ~/.bashrc
   23  source ~/.bashrc
   24  echo 'export PATH=$PATH:/opt/rocm-6.3.3/bin' >> ~/.bashrc
   25  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm-6.3.3/lib' >> ~/.bashrc
   26  echo 'export ROCM_PATH=/opt/rocm-6.3.3' >> ~/.bashrc
   27  echo 'export HIP_PATH=/opt/rocm-6.3.3' >> ~/.bashrc
   28  echo 'export CXX=/opt/rocm-6.3.3/bin/amdclang++' >> ~/.bashrc
   29  echo 'export HSA_OVERRIDE_GFX_VERSION=8.0.3' >> ~/.bashrc
   30  echo 'export OLLAMA_DEBUG=1' >> ~/.bashrc
   31  echo 'export HCC_AMDGPU_TARGET=gfx803' >> ~/.bashrc
   32  echo 'export ROCR_VISIBLE_DEVICES=0' >> ~/.bashrc
   33  echo 'export OLLAMA_DEBUG=1' >> ~/.bashrc
   34  echo 'export AMD_SERIALIZE_KERNEL=3' >> ~/.bashrc
   35  echo 'export OLLAMA_LLM_LIBRARY=rocm_v6' >> ~/.bashrc
   36  source ~/.bashrc
   37  sudo amdgpu-install --usecase=graphics,rocm,rocmdev,rocmdevtools,lrt,opencl,openclsdk,hip,hiplibsdk,mllib,mlsdk -y
   38  sudo tee --append /etc/ld.so.conf.d/rocm.conf <<EOF
   39  /opt/rocm/lib
   40  /opt/rocm/lib64
   41  EOF
   42  sudo ldconfig
   43  sudo /opt/rocm/bin/rocminfo | grep gfx
   44  sudo adduser `whoami` video
   45  sudo adduser `whoami` render
   46  newgrp video
   47  rocminfo
   48  newgrp render
   49  rovclock
   50  rovclock -i
   51  sudo rovclock -i
   52  git clone https://github.com/ollama/ollama.git
   53  cd ollama
   54  radeontop
   55  clinfo
   56  sudo apt install cmake g++ python3 python3-pip libpci-dev libboost-all-dev
   57  sudo pip3 install CppHeaderParser
   58  git clone https://github.com/ROCmSoftwarePlatform/rocBLAS.git
   59  cd rocBLAS
   60  git checkout rocm-6.3.3
   61  ./install.sh -ida gfx803
   62  cd ..
   63  cd Ollama
   64  cmake -B build -DAMDGPU_TARGETS=gfx803 && cmake --build build -- -j18
   65  ls -la /usr/lib/x86_64-linux-gnu/cmake/CLBlast/
   66  ls -la /usr/lib/x86_64-linux-gnu/libclblast*
   67  sudo find / -name "*CLBlast*.cmake" 2>/dev/null
   68  pkg-config --list-all | grep -i clblast
   69  cd ..
   70  git clone https://github.com/CNugteren/CLBlast.git
   71  cd CLBlast
   72  mkdir build && cd build
   73  cmake ..
   74  # Check for OpenCL headers
   75  find /opt/rocm* -name cl.h
   76  # Check for OpenCL libraries
   77  find /opt/rocm* -name "libOpenCL*"
   78  heathen-admin@LLM:~/CLBlast/build$ # Check for OpenCL headers
   79  find /opt/rocm* -name cl.h
   80  # Check for OpenCL libraries
   81  find /opt/rocm* -name "libOpenCL*"
   82  /opt/rocm-6.3.3/include/CL/cl.h
   83  heathen-admin@LLM:~/CLBlast/build$ ^C
   84  heathen-admin@LLM:~/CLBlast/build$ 
   85  # Try configuring CMake with explicit paths
   86  cd ~/CLBlast/build
   87  rm -rf *  # Clear previous failed build files
   88  cmake -DOPENCL_INCLUDE_DIRS=/opt/rocm-6.3.3/include -DOPENCL_LIBRARIES=/path/to/your/libOpenCL.so ..
   89  make -j$(nproc)
   90  # Try to find OpenCL library in ROCm directories
   91  find /opt/rocm-6.3.3 -name "libOpenCL*.so*"
   92  # Check common system library locations
   93  find /usr/lib -name "libOpenCL*.so*"
   94  find /usr/lib/x86_64-linux-gnu -name "libOpenCL*.so*"
   95  # Look for the library using the loader
   96  ldconfig -p | grep -i opencl
   97  cd ~/CLBlast/build
   98  rm -rf *
   99  make -j$(nproc)
  100  cmake -DOPENCL_INCLUDE_DIRS=/opt/rocm-6.3.3/include -DOPENCL_LIBRARIES=/usr/lib/x86_64-linux-gnu/libOpenCL.so.1 ..
  101  make -j$(nproc)
  102  sudo make install
  103  cd ..
  104  cd Ollama
  105  ls
  106  rm -rf build
  107  AMDGPU_TARGETS=gfx803 GPU_TARGETS=gfx803 ROCM_PATH=/opt/rocm CLBlast_DIR=/usr/local/lib/cmake/CLBlast/ go generate ./... && go build
  108  sudo apt install golang-go
  109  sudo apt-mark hold rocblas rocblas-dev
  110  sudo apt update
  111  sudo apt upgrade
  112  sudo apt-mark hold rocm-hip-libraries rocm-hip-sdk
  113  sudo apt upgrade
  114  chmod +x '/media/heathen-admin/BLUE USB/rocblas-apt-fix.sh' 
  115  sudo bash '/media/heathen-admin/BLUE USB/rocblas-apt-fix.sh' 
  116  sudo apt update
  117  sudo apt upgrade
  118  sudo apt install golang-go
  119  # Most likely location if built from source
  120  ls -la /usr/local/lib/cmake/CLBlast/
  121  # Alternative system locations
  122  ls -la /usr/lib/cmake/CLBlast/
  123  ls -la /usr/lib/x86_64-linux-gnu/cmake/CLBlast/
  124  # Search the entire system if needed
  125  sudo find / -name "CLBlast*.cmake" 2>/dev/null
  126  AMDGPU_TARGETS=gfx803 GPU_TARGETS=gfx803 ROCM_PATH=/opt/rocm CLBlast_DIR=/usr/lib/x86_64-linux-gnu/cmake/CLBlast/ go generate ./... && go build
  127  radeontop
  128  history
heathen-admin@LLM:~$ 
