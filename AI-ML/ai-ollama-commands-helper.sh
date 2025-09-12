cd ~
rm -rf ollama-for-amd
git clone https://github.com/likelovewant/ollama-for-amd.git
cd ollama-for-amd
mkdir build
cd build
cmake -G Ninja -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 -DROCM_PATH=/opt/rocm -DGGML_CUDA=OFF ..
ninja -j10
export HSA_OVERRIDE_GFX_VERSION=8.0.3
./ollama serve
