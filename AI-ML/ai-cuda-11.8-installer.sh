#!/bin/bash





######
#####
#####
# CHANGE TO CUDA VERSION 11.4
# Exit on any error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check command success
check_success() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Success: $1${NC}"
    else
        echo -e "${RED}❌ Error: $1 failed${NC}"
        exit 1
    fi
}

# Update package list and install dependencies
echo "🔄 Updating package list and installing dependencies..."
sudo apt update -y && sudo apt install -y build-essential
check_success "System dependencies installed"

# Install CUDA function
install_cuda() {
    echo "🌐 Installing CUDA 11.8 system-wide..."

    # Check disk space (~10GB needed)
    echo "Checking disk space (need ~10GB free)..."
    FREE_SPACE=$(df -h /usr/local | awk 'NR==2 {print $4}' | sed 's/G//')
    echo "Free space: $FREE_SPACE GB"
    if [ "$(echo "$FREE_SPACE < 10" | bc)" -eq 1 ]; then
        echo -e "${RED}❌ Error: Insufficient disk space ($FREE_SPACE GB < 10 GB)${NC}"
        exit 1
    fi
    check_success "$FREE_SPACE GB free—proceeding"

    # Check GCC version
    echo "Checking GCC version..."
    GCC_VERSION=$(gcc --version | head -n1 | awk '{print $3}')
    echo "Current GCC version: $GCC_VERSION"
    if [[ "$GCC_VERSION" > "11.2" ]]; then
        echo -e "${YELLOW}⚠️ Warning: GCC $GCC_VERSION is too new for CUDA 11.8—installing GCC 11${NC}"
        sudo apt install -y gcc-11 g++-11
        check_success "GCC 11 installed"
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
        sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
        check_success "GCC 11 set as default"
    fi

    # Download and install CUDA
    echo "Downloading CUDA 11.8 runfile installer..."
    wget -q --show-progress https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run -O cuda.run
    check_success "CUDA installer downloaded"
    echo "Running CUDA 11.8 installer..."
    sudo sh cuda.run --silent --toolkit --no-opengl-libs --override
    # Uncomment below to install drivers too
    # sudo sh cuda.run --silent --driver --toolkit --no-opengl-libs --override
    check_success "CUDA installed"

    # Verify nvcc exists
    if [ ! -f /usr/local/cuda-11.8/bin/nvcc ]; then
        echo -e "${RED}❌ Error: nvcc not found—installation incomplete${NC}"
        exit 1
    fi
    check_success "CUDA toolkit verified"

    # Clean up
    echo "Cleaning up installer..."
    rm -fv cuda.run
    check_success "Installer cleaned up"

    # Configure library paths
    echo "Configuring library paths..."
    sudo bash -c 'echo "/usr/local/cuda-11.8/lib64" > /etc/ld.so.conf.d/cuda-11.8.conf'
    sudo ldconfig
    check_success "Library paths configured"

    # Update environment
    echo "Setting CUDA_HOME in /etc/environment..."
    if ! grep -q "CUDA_HOME=/usr/local/cuda-11.8" /etc/environment; then
        sudo bash -c 'echo "CUDA_HOME=/usr/local/cuda-11.8" >> /etc/environment'
    fi
    check_success "CUDA_HOME set"

    # Update PATH in user's .bashrc and apply immediately
    echo "Updating PATH in ~/.bashrc..."
    BASHRC="$HOME/.bashrc"
    if [ ! -f "$BASHRC" ]; then
        echo "Creating $BASHRC..."
        touch "$BASHRC"
    fi
    if ! grep -q "/usr/local/cuda-11.8/bin" "$BASHRC"; then
        echo 'export PATH="$PATH:/usr/local/cuda-11.8/bin"' >> "$BASHRC"
    fi
    export PATH="$PATH:/usr/local/cuda-11.8/bin"  # Apply to current session
    check_success "PATH updated"
}

# Run the installation
install_cuda

# Final verification
if command -v nvcc >/dev/null 2>&1; then
    echo -e "${GREEN}🎉 CUDA 11.8 installation complete! Version: $(nvcc --version)${NC}"
else
    echo -e "${RED}❌ Error: nvcc not found after install—check /usr/local/cuda-11.8/bin${NC}"
    exit 1
fi
