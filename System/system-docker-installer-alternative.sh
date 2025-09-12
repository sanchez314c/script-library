#!/bin/bash

# Main installation script for Ollama with ROCm and Open WebUI
# This script creates the directory structure and copies all necessary files

set -e

echo "🚀 Starting installation of Ollama with ROCm and Open WebUI"

# Create the required directory structure
BASE_DIR="/home/heathen-admin/AI"
DOCKER_DIR="${BASE_DIR}/Docker"
WEBUI_DIR="${BASE_DIR}/OpenWeb-UI"

echo "📁 Creating directory structure..."
mkdir -p "${DOCKER_DIR}"
mkdir -p "${WEBUI_DIR}"

# Create all necessary files in the Docker directory
echo "✍️ Creating Dockerfile.rocm..."
cat > "${DOCKER_DIR}/Dockerfile.rocm" << 'EOL'
# Docker Buildfile for ROCm 6.3 to use Ollama with a RX570 / Polaris / gfx803 AMD GPU
# created, build and compiled by Robert Rosenbusch at January 2025
# include llm-benchmnark and open-webui 

FROM rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0

ENV WEBGUI_PORT=8080 \
    OLLAMA_PORT=11434 \
    OLLAMA_LISTENIP=0.0.0.0 \
    COMMANDLINE_ARGS='' \
    ### how many CPUCores are using while compiling
    MAX_JOBS=14 \ 
    ### Settings for AMD GPU RX570/RX580/RX590 GPU
    HSA_OVERRIDE_GFX_VERSION=8.0.3 \ 
    PYTORCH_ROCM_ARCH=gfx803 \
    ROCM_ARCH=gfx803 \ 
    TORCH_BLAS_PREFER_HIPBLASLT=0\ 
    ROC_ENABLE_PRE_VEGA=1 \
    USE_CUDA=0 \  
    USE_ROCM=1 \ 
    USE_NINJA=1 \
    FORCE_CUDA=1 \ 
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONENCODING=UTF-8\      
    REQS_FILE='requirements.txt' \
    PIP_ROOT_USER_ACTION='ignore' \
    COMMANDLINE_ARGS='' 

## Write the Environment VARSs to global... to compile later with while you use #docker save# or #docker commit#
RUN echo OLLAMA_HOST=${OLLAMA_LISTENIP}:${OLLAMA_PORT} >> /etc/environment && \ 
    echo MAX_JOBS=${MAX_JOBS} >> /etc/environment && \ 
    echo HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION} >> /etc/environment && \ 
    echo PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} >> /etc/environment && \ 
    echo ROCM_ARCH=${ROCM_ARCH} >> /etc/environment && \ 
    echo TORCH_BLAS_PREFER_HIPBLASLT=${TORCH_BLAS_PREFER_HIPBLASLT} >> /etc/environment && \ 
    echo ROC_ENABLE_PRE_VEGA=${ROC_ENABLE_PRE_VEGA} >> /etc/environment && \
    echo USE_CUDA=${USE_CUDA} >> /etc/environment && \
    echo USE_ROCM=${USE_ROCM} >> /etc/environment && \
    echo USE_NINJA=${USE_NINJA} >> /etc/environment && \
    echo FORCE_CUDA=${FORCE_CUDA} >> /etc/environment && \
    echo PIP_ROOT_USER_ACTION=${PIP_ROOT_USER_ACTION} >> /etc/environment && \
    true

## Export the AMD Stuff
RUN export OLLAMA_HOST=${OLLAMA_LISTENIP}:${OLLAMA_PORT} && \
    export MAX_JOBS=${MAX_JOBS} && \ 
    export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION} && \
    export ROC_ENABLE_PRE_VEGA=${ROC_ENABLE_PRE_VEGA} && \
    export PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} && \
    export ROCM_ARCH=${ROCM_ARCH} && \
    export TORCH_BLAS_PREFER_HIPBLASLT=${TORCH_BLAS_PREFER_HIPBLASLT} && \    
    export USE_CUDA=${USE_CUDA}  && \
    export USE_ROCM=${USE_ROCM}  && \
    export USE_NINJA=${USE_NINJA} && \
    export FORCE_CUDA=${FORCE_CUDA} && \
    export PIP_ROOT_USER_ACTION=${PIP_ROOT_USER_ACTION} && \
    true

## Update System and install golang for ollama and python virtual Env
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends virtualenv google-perftools ccache tmux mc pigz plocate golang && \
    pip install --upgrade pip wheel && \
    pip install cmake mkl mkl-include && \ 
    # symlink for Ollama
    ln -s /opt/rocm-6.3.0 /opt/rocm && \
    true

## Compile rocBLAS for gfx803    
ENV ROCBLAS_GIT_VERSION="rocm-6.3.0"
RUN echo "Checkout ROCBLAS: ${ROCBLAS_GIT_VERSION}" && \
    git clone https://github.com/ROCm/rocBLAS.git -b ${ROCBLAS_GIT_VERSION} /rocblas && \
    true

WORKDIR /rocblas
RUN echo "BUILDING rocBLAS with ARCH: ${ROCM_ARCH} and JOBS: ${MAX_JOBS}" && \
    ./install.sh -ida ${ROCM_ARCH} -j ${MAX_JOBS} && \
    true

## Checkout interactive LLM Benchmark for Ollama
RUN git clone https://github.com/willybcode/llm-benchmark.git /llm-benchmark && \
    true 

WORKDIR /llm-benchmark
RUN pip install -r requirements.txt && \
    sed -i 's/return \[model\["name"\] for model in models/return \[model\["model"\] for model in models/' benchmark.py&& \ 
    sed -i 's/return OllamaResponse.model_validate(last_element)/return last_element/' benchmark.py &&\
    true

## Install Open WebUI    
WORKDIR /    
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    pip install open-webui && \
    true
 
## Checkout Ollama
ENV OLLAMA_GIT_VERSION="v0.5.12"
RUN echo "Checkout OLLAMA: ${OLLAMA_GIT_VERSION}" && \
    git clone https://github.com/ollama/ollama.git -b ${OLLAMA_GIT_VERSION} /ollama && \
    true

## Replace gfx803 on Ollama    
WORKDIR /ollama
RUN echo "REPLACE Ollama Source for gfx803"  && \
    sed -i 's/var RocmComputeMajorMin = "9"/var RocmComputeMajorMin = "8"/' discover/gpu.go && \
    sed -i 's/"gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-" /"gfx803;gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-" /g' CMakePresets.json && \
    sed -i 's/"list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(900|94[012]|101[02]|1030|110[012])$")"/"list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(803|900|94[012]|101[02]|1030|110[012])$")"/g' CMakeLists.txt && \
    true

## Compile Ollama    
RUN echo "BUILDING Ollama for gfx803" && \
    cmake -B build -DAMDGPU_TARGETS="${ROCM_ARCH}" && \
    cmake --build build && \    
    go generate ./... && \
    go build . && \
    true

EXPOSE ${WEBGUI_PORT} ${OLLAMA_PORT}

ENTRYPOINT ["/ollama/ollama"]
CMD ["serve"]
EOL

echo "✍️ Creating docker-compose.yml..."
cat > "${DOCKER_DIR}/docker-compose.yml" << 'EOL'
version: '3.8'

services:
  ollama:
    build:
      context: .
      dockerfile: Dockerfile.rocm
    container_name: ollama
    restart: always
    ports:
      - "11434:11434"
    volumes:
      - ollama:/root/.ollama
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    environment:
      - HSA_OVERRIDE_GFX_VERSION=8.0.3
      - PYTORCH_ROCM_ARCH=gfx803
      - ROCM_ARCH=gfx803
      - ROC_ENABLE_PRE_VEGA=1
      - USE_ROCM=1
      - USE_CUDA=0
      - OLLAMA_HOST=0.0.0.0:11434
      
  webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    restart: always
    ports:
      - "3000:8080"
    volumes:
      - /home/heathen-admin/AI/OpenWeb-UI:/app/backend/data
    depends_on:
      - ollama
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      - OLLAMA_API_BASE_URL=http://ollama:11434
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      
  llm-benchmark:
    build:
      context: .
      dockerfile: Dockerfile.rocm
    container_name: llm-benchmark
    entrypoint: ["/bin/bash", "-c"]
    command: ["cd /llm-benchmark && python benchmark.py"]
    volumes:
      - ollama:/root/.ollama:ro
    depends_on:
      - ollama
    profiles:
      - benchmark
      
volumes:
  ollama:
EOL

echo "✍️ Creating .env file..."
cat > "${DOCKER_DIR}/.env" << 'EOL'
# Environment configuration for Ollama with ROCm

# GPU Architecture - Default is for RX570/RX580/RX590
HSA_OVERRIDE_GFX_VERSION=8.0.3
ROCM_ARCH=gfx803
ROC_ENABLE_PRE_VEGA=1

# Compilation settings
MAX_JOBS=14  # Adjust based on your CPU cores

# Network settings
OLLAMA_PORT=11434
WEBUI_PORT=3000

# OpenAI API Key (optional, for Open WebUI)
OPENAI_API_KEY=

# Ollama version to use
OLLAMA_VERSION=v0.5.12
EOL

echo "✍️ Creating setup.sh..."
cat > "${DOCKER_DIR}/setup.sh" << 'EOL'
#!/bin/bash

# Helper script for setting up Ollama with ROCm and Open WebUI

set -e

BASE_DIR="/home/heathen-admin/AI"
DOCKER_DIR="${BASE_DIR}/Docker"
WEBUI_DIR="${BASE_DIR}/OpenWeb-UI"

echo "🚀 Setting up Ollama with ROCm and Open WebUI"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check for AMD GPU
if ! command -v rocminfo &> /dev/null; then
    echo "⚠️ ROCm tools not found. Make sure ROCm is installed on your system."
    echo "  Continuing anyway, but GPU acceleration may not work."
else
    echo "✅ ROCm tools found."
    
    # Check for gfx803 architecture
    if rocminfo | grep -q "gfx803"; then
        echo "✅ Found gfx803 GPU architecture (RX570/RX580/RX590)."
    else
        echo "⚠️ gfx803 architecture not detected. This setup is optimized for RX570/RX580/RX590."
        echo "  You may need to modify the Dockerfile and docker-compose.yml for your GPU architecture."
    fi
fi

# Start building and running the containers
echo "🔄 Building and starting containers..."
docker compose build
docker compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
if docker compose ps | grep -q "ollama"; then
    echo "✅ Ollama service is running."
else
    echo "❌ Ollama service failed to start. Check logs with 'docker compose logs ollama'."
fi

if docker compose ps | grep -q "open-webui"; then
    echo "✅ Open WebUI service is running."
else
    echo "❌ Open WebUI service failed to start. Check logs with 'docker compose logs webui'."
fi

# Display information
echo ""
echo "📋 Setup Summary:"
echo "  - Ollama API: http://localhost:11434"
echo "  - Open WebUI: http://localhost:3000"
echo "  - Configuration files location: ${DOCKER_DIR}"
echo "  - OpenWeb-UI data location: ${WEBUI_DIR}"
echo ""
echo "🔍 To run the LLM benchmark:"
echo "  cd ${DOCKER_DIR} && docker compose --profile benchmark up llm-benchmark"
echo ""
echo "📝 To view logs:"
echo "  cd ${DOCKER_DIR} && docker compose logs -f"
echo ""
echo "🛑 To stop services:"
echo "  cd ${DOCKER_DIR} && docker compose down"

echo ""
echo "✨ Setup complete! ✨"
EOL

echo "📋 Creating README.md..."
cat > "${DOCKER_DIR}/README.md" << 'EOL'
# Ollama with ROCm and Open WebUI

This project provides a Docker Compose setup for running Ollama with AMD ROCm GPU support (specifically for RX570/RX580/RX590 with gfx803 architecture) and Open WebUI.

## Prerequisites

- Docker and Docker Compose installed
- AMD GPU with RX570/RX580/RX590 (gfx803 architecture)
- ROCm drivers properly installed on your host system

## Getting Started

1. Navigate to the Docker directory:
   ```bash
   cd /home/heathen-admin/AI/Docker
   ```

2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Access Open WebUI at:
   ```
   http://localhost:3000
   ```

## Services

### Ollama with ROCm Support
- Custom-built Ollama with support for RX570/RX580/RX590 GPUs
- Runs on port 11434
- Modified to support gfx803 architecture
- Includes rocBLAS compiled specifically for gfx803

### Open WebUI
- Web-based UI for interacting with Ollama
- Runs on port 3000
- Automatically connects to the Ollama service
- Data stored in /home/heathen-admin/AI/OpenWeb-UI

### LLM Benchmark (Optional)
- Tool for benchmarking LLM performance
- Can be started with:
  ```bash
  docker compose --profile benchmark up llm-benchmark
  ```

## Configuration

- By default, the system is configured for gfx803 architecture (RX570/RX580/RX590)
- You can modify environment variables in the .env file if needed

## Troubleshooting

If you encounter issues with GPU detection:

1. Verify your ROCm installation on the host:
   ```bash
   rocminfo
   ```

2. Check if the devices are properly passed to the container:
   ```bash
   docker exec -it ollama rocminfo
   ```

3. If you have a different GPU architecture, modify the `HSA_OVERRIDE_GFX_VERSION` and `ROCM_ARCH` environment variables in both the Dockerfile and docker-compose.yml.

## Building from Scratch

If you need to modify the build:

```bash
# Rebuild Ollama container
docker compose build ollama

# Restart services
docker compose down
docker compose up -d
```
EOL

# Make the setup script executable
chmod +x "${DOCKER_DIR}/setup.sh"

echo "📦 Creating a master script to run the setup..."
cat > "${BASE_DIR}/run-ollama-setup.sh" << EOL
#!/bin/bash
cd "${DOCKER_DIR}"
./setup.sh
EOL

chmod +x "${BASE_DIR}/run-ollama-setup.sh"

echo ""
echo "✅ Installation complete!"
echo ""
echo "📋 Summary:"
echo "  - Docker files location: ${DOCKER_DIR}"
echo "  - OpenWeb-UI data location: ${WEBUI_DIR}"
echo ""
echo "🚀 To set up and start the services, run:"
echo "  ${BASE_DIR}/run-ollama-setup.sh"
echo ""
echo "This will build and start Ollama with ROCm support and Open WebUI."
