#!/bin/bash

# Ollama ROCm Installer Script for gfx803
# Version: 1.0.0 - Built by Grok 3 (xAI) for Jason
# Date: March 1, 2025 - Supports RX 580 (gfx803) with ROCm 6.3

set -x
set -e

LOG_FILE="/home/$USER/ollama_rocm_install.log"
echo "🚀 Starting Ollama ROCm Installation for gfx803..." | tee -a "$LOG_FILE"

# Directory and path definitions
BASE_DIR="/home/$USER"
BINDIR="/usr/local/bin"
ROCM_PATH="/opt/rocm"
OLLAMA_DIR="$BASE_DIR/ollama-rocm"

check_root() {
    echo "Checking for root privileges..." | tee -a "$LOG_FILE"
    if [ "$(id -u)" != "0" ]; then
        echo "Error: Script requires root privileges. Run with sudo." | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "Success: Running as root" | tee -a "$LOG_FILE"
}

check_prereqs() {
    echo "Checking prerequisites..." | tee -a "$LOG_FILE"
    command -v git >/dev/null 2>&1 || { echo "Error: git not installed" | tee -a "$LOG_FILE"; exit 1; }
    command -v cmake >/dev/null 2>&1 || { echo "Error: cmake not installed" | tee -a "$LOG_FILE"; exit 1; }
    command -v ninja >/dev/null 2>&1 || { echo "Error: ninja not installed" | tee -a "$LOG_FILE"; exit 1; }
    command -v go >/dev/null 2>&1 || { echo "Error: go not installed" | tee -a "$LOG_FILE"; exit 1; }
    [ -d "$ROCM_PATH" ] || { echo "Error: ROCm not found at $ROCM_PATH" | tee -a "$LOG_FILE"; exit 1; }
    echo "Success: Prerequisites met" | tee -a "$LOG_FILE"
}

clone_and_patch() {
    echo "Cloning latest Ollama and patching for gfx803..." | tee -a "$LOG_FILE"
    rm -rf "$OLLAMA_DIR" || echo "No old ollama-rocm directory to remove" | tee -a "$LOG_FILE"
    git clone https://github.com/ollama/ollama.git "$OLLAMA_DIR" || { echo "Error: Git clone failed" | tee -a "$LOG_FILE"; exit 1; }
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed" | tee -a "$LOG_FILE"; exit 1; }

    # Step 1: Verify ROCm detects gfx803
    echo "Checking ROCm detection of gfx803..." | tee -a "$LOG_FILE"
    rocminfo | grep gfx803 || {
        echo "Warning: gfx803 not detected. Trying HSA override..." | tee -a "$LOG_FILE"
        export HSA_OVERRIDE_GFX_VERSION=8.0.3
        rocminfo | grep gfx803 || {
            echo "Error: gfx803 still not detected. Check ROCm installation." | tee -a "$LOG_FILE"
            exit 1
        }
    }

    # Step 2: Modify discover/gpu.go
    echo "Modifying discover/gpu.go..." | tee -a "$LOG_FILE"
    sed -i 's/var RocmComputeMajorMin = "9"/var RocmComputeMajorMin = "8"/' "$OLLAMA_DIR/discover/gpu.go" || {
        echo "Error: Failed to patch gpu.go" | tee -a "$LOG_FILE"
        exit 1
    }
    grep "RocmComputeMajorMin" "$OLLAMA_DIR/discover/gpu.go" | tee -a "$LOG_FILE" || echo "Warning: Patch applied but not verified" | tee -a "$LOG_FILE"

    # Step 3: Update CMakePresets.json
    echo "Updating CMakePresets.json..." | tee -a "$LOG_FILE"
    sed -i 's/"gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-"/"gfx803;gfx900;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-"/' "$OLLAMA_DIR/CMakePresets.json" || {
        echo "Error: Failed to patch CMakePresets.json" | tee -a "$LOG_FILE"
        exit 1
    }
    grep "AMDGPU_TARGETS" "$OLLAMA_DIR/CMakePresets.json" | grep "gfx803" | tee -a "$LOG_FILE" || {
        echo "Error: Failed to update CMakePresets.json" | tee -a "$LOG_FILE"
        exit 1
    }

    # Step 4: Update CMakeLists.txt regex
    echo "Updating CMakeLists.txt regex..." | tee -a "$LOG_FILE"
    sed -i '100s/.*/        list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(803|900(:xnack-)?|902|90c(:xnack-)?|1010(:xnack-)?|1011|1012(:xnack-)?|103[0-6]|110[0-3]|1150)$")/' "$OLLAMA_DIR/CMakeLists.txt" || {
        echo "Error: Failed to patch CMakeLists.txt regex" | tee -a "$LOG_FILE"
        exit 1
    }
    grep "AMDGPU_TARGETS INCLUDE" "$OLLAMA_DIR/CMakeLists.txt" | grep "gfx803" | tee -a "$LOG_FILE" || {
        echo "Error: Failed to update CMakeLists.txt regex" | tee -a "$LOG_FILE"
        exit 1
    }

    # Step 5: Comment out duplicate ggml-hip
    echo "Commenting out duplicate ggml-hip inclusion..." | tee -a "$LOG_FILE"
    sed -i '106s/add_subdirectory/#add_subdirectory/' "$OLLAMA_DIR/CMakeLists.txt" || {
        echo "Error: Failed to comment ggml-hip" | tee -a "$LOG_FILE"
        exit 1
    }
    grep "#add_subdirectory.*ggml-hip" "$OLLAMA_DIR/CMakeLists.txt" | tee -a "$LOG_FILE" || {
        echo "Error: Failed to comment out ggml-hip line" | tee -a "$LOG_FILE"
        exit 1
    }

    echo "Success: Ollama cloned and patched" | tee -a "$LOG_FILE"
}

build_ollama() {
    echo "Building Ollama with ROCm (gfx803) support..." | tee -a "$LOG_FILE"
    cd "$OLLAMA_DIR"
    rm -rf build
    mkdir build && cd build
    cmake -GNinja \
        -DAMDGPU_TARGETS="gfx803" \
        -DGGML_HIP=ON \
        -DGGML_CUDA=OFF \
        -DCMAKE_CUDA_COMPILER="" \
        -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
        .. > cmake.log 2>&1 || { echo "Error: CMake failed" | tee -a "$LOG_FILE"; cat cmake.log | tail -n 50 | tee -a "$LOG_FILE"; exit 1; }
    ninja -j$(nproc) > ninja.log 2>&1 || { echo "Error: Ninja build failed" | tee -a "$LOG_FILE"; cat ninja.log | tail -n 50 | tee -a "$LOG_FILE"; exit 1; }
    # Build the Go CLI binary
    cd "$OLLAMA_DIR"
    go build -o ollama-rocm . || { echo "Error: Go build failed" | tee -a "$LOG_FILE"; exit 1; }
    [ -f ollama-rocm ] || { echo "Error: ollama-rocm binary not found" | tee -a "$LOG_FILE"; exit 1; }
    sudo cp ollama-rocm "$BINDIR/ollama-rocm" || { echo "Error: Failed to install ollama-rocm" | tee -a "$LOG_FILE"; exit 1; }
    sudo mkdir -p /usr/local/lib/ollama
    sudo cp build/lib/ollama/libggml-hip.so /usr/local/lib/ollama/ || { echo "Error: Failed to install libggml-hip.so" | tee -a "$LOG_FILE"; exit 1; }
    echo "Success: Ollama built and installed" | tee -a "$LOG_FILE"
}

setup_service() {
    echo "Setting up Ollama ROCm service..." | tee -a "$LOG_FILE"
    sudo systemctl stop ollama-rocm 2>/dev/null || true
    sudo tee /etc/systemd/system/ollama-rocm.service > /dev/null << EOF || { echo "Error: Service file creation failed" | tee -a "$LOG_FILE"; exit 1; }
[Unit]
Description=Ollama Service (ROCm gfx803)
After=network-online.target

[Service]
ExecStart=$BINDIR/ollama-rocm serve
User=$USER
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=/usr/share/ollama-rocm/.ollama/models"
Environment="OLLAMA_HOST=127.0.0.1:11435"
Environment="HSA_OVERRIDE_GFX_VERSION=8.0.3"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="OLLAMA_DEBUG=true"

[Install]
WantedBy=default.target
EOF
    sudo systemctl daemon-reload || { echo "Error: Daemon reload failed" | tee -a "$LOG_FILE"; exit 1; }
    sudo systemctl enable ollama-rocm || { echo "Error: Service enable failed" | tee -a "$LOG_FILE"; exit 1; }
    sudo systemctl start ollama-rocm || { echo "Error: Service start failed" | tee -a "$LOG_FILE"; exit 1; }
    sleep 2
    sudo systemctl status ollama-rocm --no-pager | tee -a "$LOG_FILE"
    "$BINDIR/ollama-rocm" list || { echo "Error: Ollama not responding" | tee -a "$LOG_FILE"; journalctl -u ollama-rocm -n 50 -l | tee -a "$LOG_FILE"; exit 1; }
    ss -tuln | grep 11435 || { echo "Error: Port 11435 not listening" | tee -a "$LOG_FILE"; exit 1; }
    echo "Success: ROCm service set up and running" | tee -a "$LOG_FILE"
}

main() {
    echo "Starting main function..." | tee -a "$LOG_FILE"
    check_root
    check_prereqs
    clone_and_patch
    build_ollama
    setup_service
    echo "
✨ Ollama ROCm Installation Complete!
- Binary: ollama-rocm (RX 580, port 11435)
- Commands: ollama-rocm list, ollama-rocm run <model>
- Logs: journalctl -u ollama-rocm
- Models: /usr/share/ollama-rocm/.ollama/models
- Monitor: rocm-smi
- Full log: $LOG_FILE
    " | tee -a "$LOG_FILE"
}

trap 'echo "Error: Script failed at line $LINENO with exit code $?" | tee -a "$LOG_FILE"' ERR

main
