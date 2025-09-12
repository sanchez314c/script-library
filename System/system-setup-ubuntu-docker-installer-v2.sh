#!/bin/bash
###############################################################
#     ____   ___   ____ _  _______ ____                      #
#    |  _ \ / _ \ / ___| |/ / ____|  _ \                     #
#    | | | | | | | |   | ' /|  _| | |_) |                    #
#    | |_| | |_| | |___| . \| |___|  _ <                     #
#    |____/ \___/ \____|_|\_\_____|_| \_\                    #
#                                                            #
###############################################################
#
# Docker Environment Setup Script with NVIDIA CUDA and ROCm Support
# Version: 3.0.0
# Date: April 15, 2025
# Description: Sets up Docker with NVIDIA CUDA, ROCm PyTorch, and Ollama for RX580/K80

set -e  # Exit immediately if a command exits with non-zero status

# Create log file
REAL_USER="${SUDO_USER:-$USER}"
LOG_FILE="/home/${REAL_USER}/Desktop/docker_setup_$(date +%Y%m%d_%H%M%S).log"
touch "$LOG_FILE"
chown $REAL_USER:$REAL_USER "$LOG_FILE"
chmod 644 "$LOG_FILE"

log() {
    echo "$@" | tee -a "$LOG_FILE"
}

log "🚀 Starting Docker Environment Setup..."

check_root() {
    log "🔍 Checking for root privileges..."
    if [ "$(id -u)" -ne 0 ]; then
        log "❌ This script must be run with sudo"
        exit 1
    fi
    log "✅ Running as root"
}

detect_gpu() {
    log "🔍 Detecting GPUs..."
    
    # Check for NVIDIA GPU
    NVIDIA_GPU=false
    if [ -x "/usr/bin/nvidia-smi" ] && nvidia-smi &>/dev/null; then
        NVIDIA_GPU=true
        GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1)
        log "✅ NVIDIA GPU detected: $GPU_INFO"
    else
        log "⚠️ No NVIDIA GPU detected with working drivers"
    fi
    
    # Check for ROCm
    AMD_GPU=false
    ROCM_PATH=""
    
    # Check multiple common ROCm paths
    for path in /opt/rocm /opt/rocm-* /usr/local/rocm; do
        if [ -d "$path" ]; then
            ROCM_PATH="$path"
            break
        fi
    done
    
    if [ -n "$ROCM_PATH" ]; then
        if [ -x "$ROCM_PATH/bin/rocminfo" ] && "$ROCM_PATH/bin/rocminfo" &>/dev/null; then
            AMD_GPU=true
            ROCM_VERSION=$(basename "$ROCM_PATH" | sed 's/rocm-//')
            log "✅ ROCm installation detected: $ROCM_PATH (version $ROCM_VERSION)"
            
            # Check specifically for RX580 (gfx803)
            if "$ROCM_PATH/bin/rocminfo" | grep -q "gfx803"; then
                log "✅ AMD RX580 GPU detected (gfx803)"
            else
                log "ℹ️ AMD GPU detected, but not RX580 (gfx803)"
            fi
        else
            log "⚠️ ROCm directory found at $ROCM_PATH but rocminfo failed"
        fi
    else
        log "⚠️ No ROCm installation detected"
    fi
    
    log "GPU detection summary: NVIDIA: $NVIDIA_GPU, AMD: $AMD_GPU"
    
    # Verify at least one GPU type is detected
    if [ "$NVIDIA_GPU" = false ] && [ "$AMD_GPU" = false ]; then
        log "⚠️ No supported GPUs detected. Installation will continue but GPU features will be limited."
    fi
}

clean_docker() {
    log "🧹 Cleaning up Docker..."
    
    # Check if Docker is already installed correctly
    if docker --version &>/dev/null; then
        DOCKER_VERSION=$(docker --version)
        log "🔍 Docker already installed: $DOCKER_VERSION"
        
        read -p "Docker is already installed. Remove and reinstall? (y/N): " REINSTALL
        if [[ ! "$REINSTALL" =~ ^[Yy]$ ]]; then
            log "✅ Keeping existing Docker installation"
            return
        fi
        log "🧹 Removing existing Docker installation..."
    fi
    
    # Stop all Docker services first
    systemctl stop docker docker.socket containerd 2>/dev/null || true
    pkill -f dockerd 2>/dev/null || true
    
    # Remove Docker packages
    apt-get remove --purge -y docker.io docker-ce docker-ce-cli containerd.io \
        docker-compose docker-compose-plugin nvidia-container-toolkit || true
    
    # Remove Docker directories and files
    rm -rf /var/lib/docker /etc/docker /run/docker.sock /var/run/docker.sock
    rm -rf /home/${REAL_USER}/.docker
    
    # Reload systemd to recognize removed services
    systemctl daemon-reload
    
    # Remove Docker group membership
    gpasswd -d "${REAL_USER}" docker 2>/dev/null || true
    
    log "✅ Docker cleanup completed"
}

install_docker() {
    log "🐳 Installing Docker..."
    
    # Add Docker repository and key
    log "🔑 Adding Docker repository key..."
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    
    # Add Docker repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Update package list
    apt-get update || { 
        log "❌ Apt update failed, checking repository setup..."
        # Handle Ubuntu version codename issue
        if ! grep -q "VERSION_CODENAME" /etc/os-release; then
            log "⚠️ No VERSION_CODENAME found in /etc/os-release, using lsb_release instead"
            CODENAME=$(lsb_release -cs)
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $CODENAME stable" | \
                tee /etc/apt/sources.list.d/docker.list > /dev/null
            apt-get update || { log "❌ Apt update still failed after repository fix"; exit 1; }
        else
            log "❌ Apt update failed after repository fix"; 
            exit 1; 
        }
    }
    
    # Install Docker packages
    log "📦 Installing Docker packages..."
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin || {
        log "❌ Docker installation failed";
        exit 1;
    }
    
    # Verify docker installation
    if ! docker --version; then
        log "❌ Docker installation verification failed"
        exit 1
    fi
    
    # Install Docker Compose
    log "📦 Installing Docker Compose..."
    apt-get install -y docker-compose || {
        log "⚠️ Docker Compose package installation failed, trying alternative method..."
        COMPOSE_VERSION="v2.18.1"
        curl -SL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
        ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
    }
    
    # Verify Docker Compose installation
    if ! docker-compose --version; then
        log "⚠️ Docker Compose installation verification failed, but continuing..."
    fi
    
    # Start and enable Docker services
    systemctl daemon-reload
    systemctl start containerd || { log "❌ Containerd start failed"; exit 1; }
    systemctl enable containerd
    
    log "🔄 Starting Docker service..."
    systemctl start docker || {
        log "❌ Docker start failed, trying manual startup for debugging..."
        dockerd --debug > /tmp/dockerd-debug.log 2>&1 & 
        sleep 5
        pkill -f dockerd
        systemctl start docker || { 
            log "❌ Docker still failed to start"; 
            log "Debug log content:"
            cat /tmp/dockerd-debug.log | tee -a "$LOG_FILE"
            exit 1; 
        }
    }
    
    # Enable Docker service
    systemctl enable docker
    
    # Add user to Docker group
    usermod -aG docker "${REAL_USER}"
    
    # Reset any failed services
    systemctl reset-failed docker.service docker.socket containerd.service 2>/dev/null || true
    
    log "✅ Docker and Docker Compose installed successfully"
}

configure_gpu_support() {
    log "🔧 Configuring GPU support for Docker..."

    # NVIDIA Container Toolkit
    if [ "$NVIDIA_GPU" = true ]; then
        log "📦 Installing NVIDIA Container Toolkit..."
        
        if ! dpkg -l | grep -q nvidia-container-toolkit; then
            # Add NVIDIA Container Toolkit repository and key
            install -m 0755 -d /etc/apt/keyrings
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
                
            # Update package list and install
            apt-get update
            apt-get install -y nvidia-container-toolkit || { 
                log "❌ NVIDIA Container Toolkit installation failed";
                log "⚠️ Continuing without NVIDIA GPU support";
                NVIDIA_GPU=false;
            }
            
            # Configure Docker for NVIDIA
            if [ "$NVIDIA_GPU" = true ]; then
                nvidia-ctk runtime configure --runtime=docker
                log "✅ NVIDIA Container Toolkit configured"
            fi
        else
            log "✅ NVIDIA Container Toolkit already installed"
        fi
    fi

    # ROCm Docker Support
    if [ "$AMD_GPU" = true ]; then
        log "🔧 Configuring ROCm support for Docker..."
        
        # Create Docker daemon.json if it doesn't exist
        mkdir -p /etc/docker
        
        # Handle daemon.json properly - preserve existing content if present
        if [ ! -f "/etc/docker/daemon.json" ] || [ ! -s "/etc/docker/daemon.json" ]; then
            echo '{}' > /etc/docker/daemon.json
        fi
        
        log "ℹ️ ROCm will use device pass-through (--device=/dev/kfd --device=/dev/dri)"
        
        # Create environment variable file
        if [ ! -d "/home/${REAL_USER}/docker-env" ]; then
            mkdir -p "/home/${REAL_USER}/docker-env"
        fi
        
        # Create ROCm environment file
        cat > "/home/${REAL_USER}/docker-env/rocm_env.sh" << 'EOL'
#!/bin/bash
# ROCm Environment Variables for RX580
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
EOL
        
        chmod +x "/home/${REAL_USER}/docker-env/rocm_env.sh"
        chown ${REAL_USER}:${REAL_USER} -R "/home/${REAL_USER}/docker-env"
        
        log "✅ ROCm support configured with environment variables"
    fi

    # Restart Docker if GPU support was added
    if [ "$NVIDIA_GPU" = true ] || [ "$AMD_GPU" = true ]; then
        log "🔄 Restarting Docker with GPU support..."
        systemctl restart docker || { 
            log "❌ Docker restart failed";
            exit 1;
        }
        log "✅ Docker restarted with GPU support"
    fi
}

configure_docker_containers() {
    log "🐳 Configuring Docker containers..."
    
    # AMD ROCm containers
    if [ "$AMD_GPU" = true ]; then
        log "📥 Setting up ROCm containers..."
        
        # Ensure git is installed
        apt-get install -y git || { log "❌ Git installation failed"; }
        
        # Create and prepare working directory
        log "🔧 Creating temporary build directory..."
        ROCM_DOCKER_DIR=$(mktemp -d -t rocm-docker-XXXXXXXXXX)
        log "🔧 Working in directory: $ROCM_DOCKER_DIR"
        
        # Create ROCm PyTorch Dockerfile
        log "📄 Creating ROCm PyTorch Dockerfile..."
        cat > "$ROCM_DOCKER_DIR/Dockerfile" << 'EOL'
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    build-essential \
    git \
    python3 \
    python3-pip \
    python3-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install ROCm dependencies for PyTorch
RUN pip3 install --upgrade pip
# Use ROCm 5.4.2 PyTorch as it's stable for RX580
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

# Add common ML libraries and test utilities
RUN pip3 install numpy matplotlib pandas scikit-learn jupyter ipykernel

# Create workspace directory
WORKDIR /workspace

# ROCm Environment Variables for RX580
ENV HSA_ENABLE_SDMA=0
ENV ROC_ENABLE_PRE_VEGA=1
ENV HSA_OVERRIDE_GFX_VERSION=8.0.3
ENV HIP_VISIBLE_DEVICES=0
ENV HIP_PLATFORM=amd
ENV ROCR_VISIBLE_DEVICES=0
ENV HSA_NO_SCRATCH=1
ENV HIP_FORCE_DEV=0
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
ENV USE_CUDA=0
ENV USE_ROCM=1

# PyTorch & ROCm-specific Variables
ENV PYTORCH_ROCM_ARCH=gfx803
ENV ROCM_ARCH=gfx803
ENV TORCH_BLAS_PREFER_HIPBLASLT=0
ENV PYTORCH_TUNABLEOP_ENABLED=1

CMD ["/bin/bash"]
EOL

        # Create Ollama Dockerfile
        log "📄 Creating Ollama Dockerfile..."
        cat > "$ROCM_DOCKER_DIR/Dockerfile_rocm_ollama" << 'EOL'
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    build-essential \
    git \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install web UI dependencies
RUN pip3 install fastapi uvicorn jinja2 pydantic aiohttp

# Create workspace
WORKDIR /workspace

# ROCm Environment Variables for RX580
ENV HSA_ENABLE_SDMA=0
ENV ROC_ENABLE_PRE_VEGA=1
ENV HSA_OVERRIDE_GFX_VERSION=8.0.3
ENV HIP_VISIBLE_DEVICES=0
ENV HIP_PLATFORM=amd
ENV ROCR_VISIBLE_DEVICES=0
ENV HSA_NO_SCRATCH=1
ENV HIP_FORCE_DEV=0
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
ENV USE_CUDA=0
ENV USE_ROCM=1

# PyTorch & ROCm-specific Variables
ENV PYTORCH_ROCM_ARCH=gfx803
ENV ROCM_ARCH=gfx803
ENV TORCH_BLAS_PREFER_HIPBLASLT=0
ENV PYTORCH_TUNABLEOP_ENABLED=1

# Ollama Environment Variables
ENV num_gpu=1
ENV OLLAMA_GPU_OVERHEAD=0.2
ENV AMD_LOG_LEVEL=3
ENV LLAMA_HIPBLAS=1
ENV OLLAMA_LLM_LIBRARY=rocm_v6
ENV OLLAMA_DEBUG=true
ENV OLLAMA_FLASH_ATTENTION=true
ENV GPU_MAX_HEAP_SIZE=100
ENV GPU_USE_SYNC_OBJECTS=1
ENV GPU_MAX_ALLOC_PERCENT=100
ENV GPU_SINGLE_ALLOC_PERCENT=100
ENV OLLAMA_NUM_THREADS=20

EXPOSE 11434
CMD ["ollama", "serve"]
EOL

        # Build Docker images
        log "🔨 Building ROCm PyTorch container..."
        docker build -f "$ROCM_DOCKER_DIR/Dockerfile" "$ROCM_DOCKER_DIR" -t rocm_pt:latest || { 
            log "❌ ROCm PyTorch build failed";
            log "⚠️ Continuing with installation";
        }
        
        if docker images | grep -q "rocm_pt"; then
            docker tag rocm_pt:latest rocm
            log "✅ Tagged rocm_pt:latest as rocm"
        fi
        
        log "🔨 Building Ollama container..."
        docker build -f "$ROCM_DOCKER_DIR/Dockerfile_rocm_ollama" "$ROCM_DOCKER_DIR" -t rocm_ollama:latest || { 
            log "❌ Ollama build failed";
            log "⚠️ Continuing with installation";
        }
        
        if docker images | grep -q "rocm_ollama"; then
            docker tag rocm_ollama:latest ollama
            log "✅ Tagged rocm_ollama:latest as ollama"
        fi
        
        # Create docker-compose.yml for Ollama and WebUI
        log "📄 Creating docker-compose.yml for Ollama..."
        mkdir -p "/home/${REAL_USER}/docker-ollama"
        cat > "/home/${REAL_USER}/docker-ollama/docker-compose.yml" << 'EOL'
version: '3.8'

services:
  ollama:
    image: rocm_ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ./ollama-data:/root/.ollama
    restart: unless-stopped
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
      - render
    environment:
      - HSA_OVERRIDE_GFX_VERSION=8.0.3
      - OLLAMA_GPU_OVERHEAD=0.2
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_LLM_LIBRARY=rocm_v6

  webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: webui
    depends_on:
      - ollama
    ports:
      - "3000:8080"
    volumes:
      - ./webui-data:/app/backend/data
    restart: unless-stopped
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - WEBUI_AUTH=optional

volumes:
  ollama-data:
  webui-data:
EOL
        chown -R ${REAL_USER}:${REAL_USER} "/home/${REAL_USER}/docker-ollama"
        
        # Clean up temporary directory
        rm -rf "$ROCM_DOCKER_DIR"
        log "✅ ROCm containers configured"
    fi
    
    # NVIDIA CUDA containers
    if [ "$NVIDIA_GPU" = true ]; then
        log "📥 Pulling NVIDIA CUDA containers..."
        docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 && \
            docker tag nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 cuda0 && \
            docker tag nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 cuda1
            
        # Create environment variable file for CUDA
        cat > "/home/${REAL_USER}/docker-env/cuda_env.sh" << 'EOL'
#!/bin/bash
# CUDA Environment Variables for K80
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
EOL
        
        chmod +x "/home/${REAL_USER}/docker-env/cuda_env.sh"
        chown ${REAL_USER}:${REAL_USER} "/home/${REAL_USER}/docker-env/cuda_env.sh"
        
        # Create CUDA Dockerfile for specific environment variables
        log "📄 Creating CUDA Dockerfile..."
        CUDA_DOCKER_DIR=$(mktemp -d -t cuda-docker-XXXXXXXXXX)
        cat > "$CUDA_DOCKER_DIR/Dockerfile_cuda_ollama" << 'EOL'
FROM ollama/ollama:latest

# CUDA Environment Variables for K80
ENV CUDA_VISIBLE_DEVICES=0,1,2
ENV TORCH_CUDA_ARCH_LIST="3.5;3.7"
ENV CUDA_COMPUTE_MAJOR=3
ENV CUDA_COMPUTE_MINOR=5
ENV FORCE_CUDA=1
ENV CUDA_CACHE_PATH=/root/.nv/ComputeCache
ENV OLLAMA_NUM_THREADS=20

# PyTorch CUDA-specific Variables
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV CUDNN_BENCHMARK=1

EXPOSE 11434
CMD ["ollama", "serve"]
EOL

        # Build CUDA Ollama image
        log "🔨 Building CUDA Ollama container..."
        docker build -f "$CUDA_DOCKER_DIR/Dockerfile_cuda_ollama" "$CUDA_DOCKER_DIR" -t cuda_ollama:latest || { 
            log "❌ CUDA Ollama build failed";
            log "⚠️ Using default ollama/ollama image instead";
        }
        
        if docker images | grep -q "cuda_ollama"; then
            log "✅ Built cuda_ollama:latest with custom environment variables"
        else
            log "⚠️ Using default ollama/ollama image"
        fi
        
        # Create docker-compose.yml for CUDA Ollama
        log "📄 Creating docker-compose.yml for CUDA Ollama..."
        mkdir -p "/home/${REAL_USER}/docker-ollama-cuda"
        cat > "/home/${REAL_USER}/docker-ollama-cuda/docker-compose.yml" << 'EOL'
version: '3.8'

services:
  ollama:
    image: cuda_ollama:latest
    container_name: ollama-cuda
    ports:
      - "11435:11434"
    volumes:
      - ./ollama-data:/root/.ollama
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - OLLAMA_HOST=0.0.0.0
      - CUDA_VISIBLE_DEVICES=0,1,2
      - TORCH_CUDA_ARCH_LIST=3.5;3.7
      - CUDA_COMPUTE_MAJOR=3
      - CUDA_COMPUTE_MINOR=5
      - FORCE_CUDA=1
      - OLLAMA_NUM_THREADS=20
      - TORCH_CUDNN_V8_API_ENABLED=1
      - CUBLAS_WORKSPACE_CONFIG=:4096:8
      - CUDNN_BENCHMARK=1

  webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: webui-cuda
    depends_on:
      - ollama
    ports:
      - "3001:8080"
    volumes:
      - ./webui-data:/app/backend/data
    restart: unless-stopped
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - WEBUI_AUTH=optional

volumes:
  ollama-data:
  webui-data:
EOL
        chown -R ${REAL_USER}:${REAL_USER} "/home/${REAL_USER}/docker-ollama-cuda"
        log "✅ NVIDIA CUDA containers configured"
    fi
    
    # Add convenience alias for ROCm containers
    if [ "$AMD_GPU" = true ]; then
        BASHRC_FILE="/home/${REAL_USER}/.bashrc"
        if [ -f "$BASHRC_FILE" ] && ! grep -q "alias drun" "$BASHRC_FILE"; then
            DRUN_ALIAS="alias drun='docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G -v \$HOME/dockerx:/workspace -w /workspace'"
            echo "$DRUN_ALIAS" >> "$BASHRC_FILE"
            log "✅ Added 'drun' alias for ROCm containers to ${BASHRC_FILE}"
            
            # Add environment sourcing
            if ! grep -q "source ~/docker-env/rocm_env.sh" "$BASHRC_FILE"; then
                echo "# Source ROCm environment variables" >> "$BASHRC_FILE"
                echo "[ -f ~/docker-env/rocm_env.sh ] && source ~/docker-env/rocm_env.sh" >> "$BASHRC_FILE"
                log "✅ Added ROCm environment sourcing to ${BASHRC_FILE}"
            fi
        fi
    fi
    
    # Add convenience alias for CUDA containers
    if [ "$NVIDIA_GPU" = true ]; then
        BASHRC_FILE="/home/${REAL_USER}/.bashrc"
        if [ -f "$BASHRC_FILE" ] && ! grep -q "alias cudadrun" "$BASHRC_FILE"; then
            CUDADRUN_ALIAS="alias cudadrun='docker run -it --gpus all --network=host --ipc=host --shm-size 8G -v \$HOME/dockerx-cuda:/workspace -w /workspace'"
            echo "$CUDADRUN_ALIAS" >> "$BASHRC_FILE"
            log "✅ Added 'cudadrun' alias for CUDA containers to ${BASHRC_FILE}"
            
            # Add environment sourcing
            if ! grep -q "source ~/docker-env/cuda_env.sh" "$BASHRC_FILE"; then
                echo "# Source CUDA environment variables" >> "$BASHRC_FILE"
                echo "[ -f ~/docker-env/cuda_env.sh ] && source ~/docker-env/cuda_env.sh" >> "$BASHRC_FILE"
                log "✅ Added CUDA environment sourcing to ${BASHRC_FILE}"
            fi
        fi
    fi
    
    # Add convenience script to start containers
    log "📄 Creating container starter scripts..."
    if [ "$AMD_GPU" = true ]; then
        cat > "/home/${REAL_USER}/start-ollama-rocm.sh" << 'EOL'
#!/bin/bash
# Source ROCm environment variables
[ -f ~/docker-env/rocm_env.sh ] && source ~/docker-env/rocm_env.sh

# Start containers
cd ~/docker-ollama
docker-compose up -d

echo "Ollama with ROCm started at http://localhost:3000"
echo "API available at http://localhost:11434"
echo "Environment variables loaded from ~/docker-env/rocm_env.sh"
EOL
        chmod +x "/home/${REAL_USER}/start-ollama-rocm.sh"
        chown ${REAL_USER}:${REAL_USER} "/home/${REAL_USER}/start-ollama-rocm.sh"
    fi
    
    if [ "$NVIDIA_GPU" = true ]; then
        cat > "/home/${REAL_USER}/start-ollama-cuda.sh" << 'EOL'
#!/bin/bash
# Source CUDA environment variables
[ -f ~/docker-env/cuda_env.sh ] && source ~/docker-env/cuda_env.sh

# Start containers
cd ~/docker-ollama-cuda
docker-compose up -d

echo "Ollama with CUDA started at http://localhost:3001"
echo "API available at http://localhost:11435"
echo "Environment variables loaded from ~/docker-env/cuda_env.sh"
EOL
        chmod +x "/home/${REAL_USER}/start-ollama-cuda.sh"
        chown ${REAL_USER}:${REAL_USER} "/home/${REAL_USER}/start-ollama-cuda.sh"
    fi
    
    log "✅ Docker containers and utilities configured"
}

verify_docker() {
    log "🔍 Verifying Docker setup..."
    
    # Check Docker and Docker Compose versions
    log "Checking Docker version:"
    docker --version | tee -a "$LOG_FILE" || log "❌ Docker verification failed"
    
    log "Checking Docker Compose version:"
    docker-compose --version | tee -a "$LOG_FILE" || log "❌ Docker Compose verification failed"
    
    # Check GPU access in Docker
    if [ "$NVIDIA_GPU" = true ]; then
        log "Testing NVIDIA GPU access in Docker..."
        if docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 nvidia-smi; then
            log "✅ NVIDIA GPU test passed"
        else
            log "⚠️ NVIDIA GPU test failed. Check nvidia-container-toolkit installation."
        fi
    fi
    
    if [ "$AMD_GPU" = true ]; then
        log "Testing AMD GPU access in Docker..."
        if docker run --rm --device=/dev/kfd --device=/dev/dri --group-add=video ubuntu:20.04 ls -la /dev/dri; then
            log "✅ AMD GPU test passed"
        else
            log "⚠️ AMD GPU device access test failed. Check permissions."
        fi
    fi
    
    log "✅ Docker verification complete"
}

show_final_information() {
    # Get IP address for convenience
    IP_ADDRESS=$(hostname -I | awk '{print $1}')
    
    log "
✨ Docker Setup Complete! ✨

🐳 Docker Information:
- Docker Engine and Docker Compose installed
- GPU Support: NVIDIA ($NVIDIA_GPU), AMD ($AMD_GPU)
"

    if [ "$AMD_GPU" = true ]; then
        log "
🔧 ROCm Docker Setup:
- Container images: rocm_pt (pytorch), rocm_ollama (ollama)
- Convenient alias 'drun' added to run ROCm containers
- Ollama + WebUI setup in ~/docker-ollama/
- Start with: ~/start-ollama-rocm.sh
- WebUI will be available at: http://$IP_ADDRESS:3000
- Environment variables in ~/docker-env/rocm_env.sh
"
    fi

    if [ "$NVIDIA_GPU" = true ]; then
        log "
🔧 NVIDIA Docker Setup:
- Container images: cuda_ollama, nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 (cuda0, cuda1)
- Convenient alias 'cudadrun' added to run CUDA containers
- Ollama + WebUI setup in ~/docker-ollama-cuda/
- Start with: ~/start-ollama-cuda.sh
- WebUI will be available at: http://$IP_ADDRESS:3001
- Environment variables in ~/docker-env/cuda_env.sh
"
    fi

    log "
⚠️ Important Notes:
- To use docker without sudo, log out and log back in
- All environment variables for both GPU types have been set up
- Environment scripts will be automatically sourced from .bashrc
- Check $LOG_FILE for detailed installation log
- For any issues, verify GPU drivers first
"
}

main() {
    check_root
    detect_gpu
    clean_docker
    install_docker
    configure_gpu_support
    configure_docker_containers
    verify_docker
    show_final_information
}

main