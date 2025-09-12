#!/bin/bash 😈

# Ollama ROCm Source Build Script for RX 580 (HIP-Only) 🚀
# Version: 1.4.7 - Built by Cortana (via Grok 3, xAI) for Jason 💪
# Date: February 28, 2025 📅

set -x
set -v
set -e

echo "Starting Ollama ROCm Source Build for RX 580... 🌟"

OLLAMA_DIR="/home/$USER/ollama-rocm"
BINDIR="/usr/local/bin"
ROCM_PATH="/opt/rocm-6.3.3"
GO_VERSION="1.22.0"
LOG_FILE="/home/$USER/Desktop/ollama-build.log"

# Determine number of CPU cores for parallel builds
N_CORES=$(nproc)
echo "Detected $N_CORES CPU cores—using all for parallel builds ⚡"

check_root() {
    echo "Checking for root privileges... 🔒"
    if [ "$(id -u)" != "0" ]; then
        echo "Error: Script requires root privileges. Run with sudo. 😕"
        exit 1
    fi
    echo "Success: Running as root ✅"
}

setup_repository() {
    echo "Cloning Ollama repository (v0.5.12)... 🌀"
    rm -rf "$OLLAMA_DIR" || echo "No old directory to remove 😶"
    git clone --branch v0.5.12 https://github.com/ollama/ollama.git "$OLLAMA_DIR" || { echo "Error: Git clone failed 😢"; exit 1; }
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed 😢"; exit 1; }
    echo "Verifying clone contents... 👀"
    ls -l llm/ || { echo "Error: llm/ directory not found—clone may have failed 😢"; exit 1; }
    echo "Success: Repository cloned ✅"
}

setup_build_env() {
    echo "Setting up build environment... 🛠️"
    apt update || { echo "Error: Apt update failed 😢"; exit 1; }
    apt install -y libstdc++-12-dev cmake gcc-11 g++-11 git librocprim-dev || { echo "Error: Build deps install failed 😢"; exit 1; }  # Upgraded to GCC 11 for AVX-512 VNNI
    apt remove -y golang-go golang || true
    rm -rf /usr/local/go /usr/bin/go /usr/local/bin/go || true
    wget -v "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -O go.tar.gz || { echo "Error: Go download failed 😢"; exit 1; }
    tar -C /usr/local -xzf go.tar.gz || { echo "Error: Go extraction failed 😢"; exit 1; }
    rm -fv go.tar.gz
    echo "Installed Go version: $(/usr/local/go/bin/go version) ✅"
    export PATH="/usr/local/go/bin:$PATH"
    export CC="/usr/bin/gcc-11"  # Updated to GCC 11
    export CXX="/usr/bin/g++-11"  # Updated to G++ 11

    # Check GCC version and adjust AVX flags
    GCC_VERSION=$(/usr/bin/gcc-11 --version | head -n1 | cut -d" " -f4 | cut -d. -f1)
    if [ "$GCC_VERSION" -lt 11 ]; then
        echo "Warning: GCC < 11 detected—unexpected, forcing -mavx512vnni ⚠️"
        sed -i 's/-mavxvnni/-mavx512vnni/g' CMakeLists.txt
    else
        echo "GCC 11 or newer detected—keeping -mavx512vnni as is ✅"
    fi

    echo "Setting HIP-specific environment variables... 💾"
    export CGO_CFLAGS="-I$ROCM_PATH/include -I$ROCM_PATH/hip/include"
    export CGO_LDFLAGS="-L$ROCM_PATH/lib -L$ROCM_PATH/lib64 -L$ROCM_PATH/hip/lib -lamdhip64 -lhiprtc -lrocm_smi64 -lroctx64 -lrocfft -lhipblas -lggml -lllama -lclip"  # Added GGML, LLaMA, Clip libraries
    export GOFLAGS="-tags=hip"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib"
    export HSA_OVERRIDE_GFX_VERSION="8.0.3"
    export ROC_ENABLE_PRE_VEGA="1"
    export PYTORCH_ROCM_ARCH="gfx803"
    export HIP_VISIBLE_DEVICES="0"
    echo "CGO_CFLAGS=$CGO_CFLAGS"
    echo "CGO_LDFLAGS=$CGO_LDFLAGS"
    echo "GOFLAGS=$GOFLAGS"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "Success: Build environment configured for ROCm (HIP-only) ✅"
}

patch_ollama() {
    echo "Patching Ollama to accept gfx803... 🛠️"
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed 😢"; exit 1; }
    
    # Find the AMD GPU detection file
    GPU_FILE=$(find . -type f -name "*.go" -exec grep -l "amdgpu" {} + | head -1)
    if [ -z "$GPU_FILE" ]; then
        echo "Error: Could not locate GPU detection file with 'amdgpu' 😢"
        ls -lR "$OLLAMA_DIR" > "$OLLAMA_DIR/dir_listing.txt"
        exit 1
    fi
    
    echo "Found GPU check file: $GPU_FILE 👀"
    
    # First, let's understand the structure of the file
    echo "Examining context of GPU detection file... 🔍"
    grep -n "func " "$GPU_FILE" | head -10
    
    # Display the relevant code section
    LINE_NUM=$(grep -n "amdgpu too old" "$GPU_FILE" | cut -d ':' -f 1)
    if [ -z "$LINE_NUM" ]; then
        echo "Error: Could not locate 'amdgpu too old' line 😢"
        exit 1
    fi
    
    echo "Found 'amdgpu too old' at line $LINE_NUM ✅"
    
    # Look at the surrounding code for context
    echo "Code context around rejection: 📜"
    CONTEXT_START=$((LINE_NUM - 10))
    [ "$CONTEXT_START" -lt 1 ] && CONTEXT_START=1
    CONTEXT_END=$((LINE_NUM + 10))
    sed -n "${CONTEXT_START},${CONTEXT_END}p" "$GPU_FILE"
    
    # Find the function that contains this code
    FUNC_START=$(grep -n "^func" "$GPU_FILE" | awk -F: '$1 < '$LINE_NUM' {last=$1} END {print last}')
    if [ -z "$FUNC_START" ]; then
        echo "Error: Could not locate function beginning 😢"
        exit 1
    fi
    
    echo "Containing function starts at line $FUNC_START ✅"
    
    # Create a backup of the original file
    cp "$GPU_FILE" "${GPU_FILE}.bak"
    
    # Now apply a direct patch to disable the age check and fix the 'reason' undefined error
    echo "Applying patch to force accept gfx803 and fix 'reason' undefined... 🛠️"
    
    # Fix the 'reason' undefined by adding a default reason string or variable
    echo "Fixing 'undefined: reason' in $GPU_FILE... 🛠️"
    if ! grep -q "const reason string" "$GPU_FILE"; then
        sed -i '/^import.*$/a\const reason = "gfx803 compatibility issue"' "$GPU_FILE" || \
        sed -i '/^package .*$/a\const reason = "gfx803 compatibility issue"' "$GPU_FILE"
    fi
    sed -i '/amdgpu too old/{s/return nil, err/return gpus, fmt.Errorf("AMD GPU detected but too old for gfx803: %v", reason)/}' "$GPU_FILE"
    
    # Rather than trying to modify a complex return statement, let's add a short-circuit at the top
    # of the function that processes AMD GPUs to accept gfx803
    FUNC_LINE=$(sed -n "${FUNC_START}p" "$GPU_FILE")
    FUNC_NAME=$(echo "$FUNC_LINE" | grep -o "func [a-zA-Z0-9_]*" | cut -d' ' -f2)
    echo "Function name: $FUNC_NAME ✅"
    
    # Comment out the age check and force acceptance for gfx803
    sed -i "${LINE_NUM}s/.*amdgpu too old.*/\t\/\/ Bypassing old gfx check for gfx803/" "$GPU_FILE"
    
    # Now find the line that returns nil, err right after the "too old" warning
    RETURN_LINE=$((LINE_NUM + 1))
    RETURN_CONTENT=$(sed -n "${RETURN_LINE}p" "$GPU_FILE")
    
    if [[ "$RETURN_CONTENT" == *"return nil, err"* ]]; then
        echo "Found return statement, replacing with acceptance... ✅"
        sed -i "${RETURN_LINE}s/return nil, err/return gpus, nil/" "$GPU_FILE"
    else
        echo "Return statement not found at expected line, searching nearby... 🔍"
        for i in {1..5}; do
            RETURN_LINE=$((LINE_NUM + i))
            RETURN_CONTENT=$(sed -n "${RETURN_LINE}p" "$GPU_FILE")
            if [[ "$RETURN_CONTENT" == *"return nil, err"* ]]; then
                echo "Found return at line $RETURN_LINE, replacing... ✅"
                sed -i "${RETURN_LINE}s/return nil, err/return gpus, nil/" "$GPU_FILE"
                break
            fi
        done
    fi
    
    # Compare original and patched files to verify changes
    echo "Patch changes: 🔄"
    diff -u "${GPU_FILE}.bak" "$GPU_FILE" || echo "Files differ as expected ✅"
    
    echo "Success: Patched Ollama source to accept gfx803 and fix 'reason' undefined ✅"
}

build_ollama() {
    echo "Building Ollama with ROCm (HIP-only) support... 💪"
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed 😢"; exit 1; }
    
    echo "Running CMake build with explicit GGML/LLaMA libraries... 🛠️"
    cmake -B build -DAMDGPU_TARGETS=gfx803 -DLLAMA_BUILD_SHARED=ON -DGGML_BUILD_SHARED=ON || { echo "Error: CMake configuration failed 😢"; exit 1; }
    cmake --build build -j$N_CORES || { echo "Error: CMake build failed 😢"; exit 1; }  # Use all CPU cores
    
    echo "Generating Go files with HIP... 🌀"
    /usr/local/go/bin/go generate ./... || { echo "Error: Go generate failed 😢"; exit 1; }
    
    echo "Building Ollama with HIP tags and explicit linking... 🚀"
    CGO_ENABLED=1 /usr/local/go/bin/go build -v -x -p $N_CORES -o ollama-rocm -ldflags "-L$ROCM_PATH/lib -L$ROCM_PATH/lib64 -L$ROCM_PATH/hip/lib -lggml -lllama -lclip -lamdhip64 -lhiprtc -lrocm_smi64 -lroctx64 -lrocfft -lhipblas" . || { echo "Error: Go build failed—check above for cgo/link errors 😢"; exit 1; }
    
    if [ ! -f ollama-rocm ]; then
        echo "Error: Build failed—ollama-rocm binary not found 😢"
        exit 1
    fi
    mv -v ollama-rocm "$BINDIR/ollama-rocm" || { echo "Error: Failed to move binary 😢"; exit 1; }
    echo "Success: Ollama-rocm built and installed ✅"
}

create_service() {
    echo "Creating systemd service... ⚙️"
    tee /etc/systemd/system/ollama-rocm.service > /dev/null << EOF || { echo "Error: Service file creation failed 😢"; exit 1; }
[Unit]
Description=Ollama Service (ROCm HIP-Only)
After=network-online.target

[Service]
ExecStart=$BINDIR/ollama-rocm serve
User=$USER
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=/usr/share/ollama-rocm/.ollama/models"
Environment="OLLAMA_HOST=127.0.0.1:11435"
Environment="ROCM_PATH=$ROCM_PATH"
Environment="LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib"
Environment="HSA_OVERRIDE_GFX_VERSION=8.0.3"
Environment="ROC_ENABLE_PRE_VEGA=1"
Environment="PYTORCH_ROCM_ARCH=gfx803"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="OLLAMA_DEBUG=true"

[Install]
WantedBy=default.target
EOF
    systemctl daemon-reload || { echo "Error: Daemon reload failed 😢"; exit 1; }
    systemctl enable ollama-rocm || { echo "Error: Service enable failed 😢"; exit 1; }
    systemctl restart ollama-rocm || { echo "Error: Service restart failed 😢"; exit 1; }
    echo "Success: Service created and started ✅"
}

verify_installation() {
    echo "Verifying installation... 🵡‍♀️"
    sleep 2
    $ROCM_PATH/bin/rocminfo | grep -A 5 "Name:.*gfx803" || { echo "Error: rocminfo failed or RX580 not detected 😢"; exit 1; }
    $ROCM_PATH/bin/rocm-smi || { echo "Error: rocm-smi failed 😢"; exit 1; }
    systemctl status ollama-rocm --no-pager || { echo "Error: Service status check failed 😢"; exit 1; }
    $BINDIR/ollama-rocm list || { echo "Error: Ollama test failed 😢"; exit 1; }
    echo "Checking HIP usage in logs... 🔍"
    journalctl -u ollama-rocm --since "2 minutes ago" -l | grep -i "hip" && echo "Success: HIP detected in logs ✅" || { echo "Warning: No HIP usage detected—checking for AMD references ⚠️"; }
    
    # Check for any AMD-related activity in logs
    journalctl -u ollama-rocm --since "2 minutes ago" -l | grep -i "amd\|rocm\|gfx803" && echo "Success: AMD references in logs ✅" || { echo "Warning: No AMD references found in logs ⚠️"; }
    
    echo "Testing GPU activity... ⚡"
    # Run in background and capture output
    $BINDIR/ollama-rocm run llama2 "Hello world" > /tmp/ollama_test_output.txt 2>&1 &
    OLLAMA_PID=$!
    
    echo "Waiting for model to load and utilize GPU (15 seconds)... ⏳"
    sleep 15
    
    # Check GPU utilization 
    GPU_USAGE=$($ROCM_PATH/bin/rocm-smi | grep -o "GPU.*[1-9][0-9]*%" || echo "0%")
    echo "Current GPU usage: $GPU_USAGE"
    
    # Clean up test process
    if ps -p $OLLAMA_PID > /dev/null; then
        echo "Terminating test process... 🚫"
        kill -TERM $OLLAMA_PID || kill -KILL $OLLAMA_PID
    fi
    
    # Check test output
    cat /tmp/ollama_test_output.txt
    
    # Final verification based on logs
    echo "Checking recent logs for GPU activity... 🵡‍♀️"
    journalctl -u ollama-rocm --since "1 minute ago" -l | grep -i "gpu\|rocm\|amd\|hip" && echo "Success: GPU-related activity in logs ✅" || echo "Warning: No GPU-related activity found in recent logs ⚠️"
    
    echo "Verification complete - monitor GPU usage with: rocm-smi ✅"
}

main() {
    echo "Entering main function... 🎯"
    check_root
    setup_repository
    setup_build_env
    patch_ollama
    build_ollama
    create_service
    verify_installation
    echo "
Ollama ROCm Build Complete! 🎉
- Built with HIP-only support for RX 580 (gfx803) 🚀
- Binary: $BINDIR/ollama-rocm
- Service: ollama-rocm (port 11435)
Commands:
- ollama-rocm list : List models 📋
- ollama-rocm run <model> : Run model 🎮
- journalctl -u ollama-rocm : View logs 📜
Notes:
- Models stored in /usr/share/ollama-rocm/.ollama/models 📁
- Runs standalone—no CUDA or Conda dependency 🚫
- If you still don't see GPU utilization, try:
  systemctl restart ollama-rocm ⚙️
  rocm-smi --reset 🔄
    "
}

trap 'echo "Error: Script failed at line $LINENO with exit code $? 😢"' ERR

main
