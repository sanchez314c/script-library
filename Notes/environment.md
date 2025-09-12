##### ETC/ENVIRONMENT #####

# General Environment Variables
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PYTHONENCODING=UTF-8
export PIP_ROOT_USER_ACTION=ignore

# ROCm Paths
export ROCM_PATH=/opt/rocm-6.3.4
export HIP_PATH=/opt/rocm-6.3.4/hip

# CUDA Paths
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda  

# Combined PATH
export PATH=$HOME/bin:/usr/local/cuda/bin:/usr/local/go/bin:$ROCM_PATH/bin:$ROCM_PATH/hip/bin:$PATH

export PATH="$HOME/.cargo/bin:$PATH"

# Combined LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$LD_LIBRARY_PATH


##### ROCM CONDA VARS #####

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

# PyTorch & ROCm-specific Variables
export PYTORCH_ROCM_ARCH=gfx803
export ROCM_ARCH=gfx803
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_ENABLED=1

# Ollama Environment Variables
export num_gpu=1
export OLLAMA_GPU_OVERHEAD=0.2
export AMD_LOG_LEVEL=3
export LLAMA_HIPBLAS=1
export OLLAMA_LLM_LIBRARY=rocm_v6
export OLLAMA_DEBUG=true
export OLLAMA_FLASH_ATTENTION=true
export GPU_MAX_HEAP_SIZE=100
export GPU_USE_SYNC_OBJECTS=1
export GPU_MAX_ALLOC_PERCENT=100
export GPU_SINGLE_ALLOC_PERCENT=100
export OLLAMA_NUM_THREADS=20


##### CUDA CONDA VARS #####

# CUDA0 Environment Variables
export CUDA_VISIBLE_DEVICES=0,1,2
export TORCH_CUDA_ARCH_LIST="3.5;3.7"
export CUDA_COMPUTE_MAJOR=3
export CUDA_COMPUTE_MINOR=5
export FORCE_CUDA=1
export CUDA_CACHE_PATH=~/.nv/ComputeCache
export OLLAMA_NUM_THREADS=20

# PyTorch CUDA-specific Variables
export TORCH_CUDNN_V8_API_ENABLED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_BENCHMARK=1
