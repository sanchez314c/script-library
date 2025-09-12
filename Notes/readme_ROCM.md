#!/bin/bash
# Force ROCm to build for gfx803



####instal rocm, etc

sudo apt install libopenmpi3 libstdc++-12-dev libdnnl-dev ninja-build libopenblas-dev libpng-dev libjpeg-dev


######ROCBLAS shit


#Clone Ollama Repo
git clone https://github.com/ollama/ollama.git


export CMAKE_C_COMPILER=/opt/rocm-6.3.4/bin/hipcc
export CMAKE_CXX_COMPILER=/opt/rocm-6.3.4/bin/hipcc
export ROCM_PATH=/opt/rocm-6.3.4
export HIP_PLATFORM=amd
export CMAKE_PREFIX_PATH=$ROCM_PATH
 -DROCM_PATH="$ROCM_PATH" CGO_ENABLED=1


############Clone Roberts RocBlas gfx803 Lib
git clone https://github.com/ROCm/rocBLAS.git

sudo ./install.sh -ida gfx803



############Change RocmCompute from 9 to 8

sed -i 's/var RocmComputeMajorMin = "9"/var RocmComputeMajorMin = "8"/' discover/gpu.go

###########Add gfx803 to AMDGPUTARGETs on CMakePresets.json and CMakeLists.txt

sed -i 's/"gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-" /"gfx803;gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-" /g' CMakePresets.json

sed -i 's/"list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(900|94[012]|101[02]|1030|110[012])$")"/"list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(803|900|94[012]|101[02]|1030|110[012])$")"/g' CMakeLists.txt


#########################
##########CMAKE##########
#########################

// There are 3 potential env vars to use to select GPUs.
// ROCR_VISIBLE_DEVICES supports UUID or numeric so is our preferred on linux
// GPU_DEVICE_ORDINAL supports numeric IDs only
// HIP_VISIBLE_DEVICES supports numeric IDs only

#########AMD
cmake -B build -DAMDGPU_TARGETS=gfx803 && cmake --build build -- -j 18

cmake -B build -DOLLAMA_ROCM=ON -DROCM_TARGETS=gfx803
make -C build -j$(nproc)

cmake -B build -DCMAKE_PREFIX_PATH=/opt/rocm -DAMDGPU_TARGETS=gfx803 && cmake --build build -- -j18

cmake -B build -DGGML_HIP=ON -DCMAKE_C_COMPILER=/opt/rocm-6.3.3/bin/hipcc -DCMAKE_CXX_COMPILER=/opt/rocm-6.3.3/bin/hipcc -DCMAKE_PREFIX_PATH=/opt/rocm -DAMDGPU_TARGETS=gfx803 && cmake --build build -- -j18

build && cmake -B build -DGGML_HIPBLAS=ON -DAMDGPU_TARGETS=gfx803 && cmake --build build -- -j18


cmake -B build -DGGML_HIPBLAS=ON -DCMAKE_C_COMPILER=/opt/rocm-6.3.4/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm-6.3.4/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm-6.3.4 -DHIP_ROOT_DIR=/opt/rocm-6.3.4 && cmake --build build -- -j18


cmake -B build -DLLAMA_HIPBLAS=ON -DCMAKE_C_COMPILER=/opt/rocm-6.3.4/bin/hipcc -DCMAKE_CXX_COMPILER=/opt/rocm-6.3.4/bin/hipcc -DCMAKE_PREFIX_PATH=/opt/rocm-6.3.4 -DAMDGPU_TARGETS=gfx803 && cmake --build build -- -j18


########nVidia
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="35;37;50;52" && cmake --build build -- -j18


###########################
###########BUILD###########
###########################

go generate ./... &&  go build .


###########################
###########SERVE###########
###########################

./ollama serve&


###########################
###########RUN#############
###########################

./ollama run tinyllama



