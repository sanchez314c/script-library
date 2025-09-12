 Script works and builds but Ollama isn't detecting ROCm due to too old at the application level so exports don't effect it. Help?


#!/bin/bash

# Ollama ROCm Source Build Script for RX 580 (HIP-Only)
# Version: 1.2.23 - Built by Cortana (via Grok 3, xAI) for Jason
# Date: February 25, 2025

set -x
set -e

echo "Starting Ollama ROCm Source Build for RX 580..."

OLLAMA_DIR="/home/$USER/ollama-rocm"
BINDIR="/usr/local/bin"
ROCM_PATH="/opt/rocm-6.3.3"
GO_VERSION="1.23.4"
LOG_FILE="/home/$USER/Desktop/ollama-build.log"

check_root() {
    echo "Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    echo "Success: Running as root"
}

setup_repository() {
    echo "Cloning Ollama repository (v0.5.12)..."
    rm -rf "$OLLAMA_DIR" || echo "No old directory to remove"
    git clone --branch v0.5.12 https://github.com/ollama/ollama.git "$OLLAMA_DIR" || { echo "Error: Git clone failed"; exit 1; }
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed"; exit 1; }
    echo "Verifying clone contents..."
    ls -l llm/ || { echo "Error: llm/ directory not found—clone may have failed"; exit 1; }
    echo "Success: Repository cloned"
}

setup_build_env() {
    echo "Setting up build environment..."
    apt update || { echo "Error: Apt update failed"; exit 1; }
    apt install -y libstdc++-12-dev cmake gcc-10 g++-10 git librocprim-dev || { echo "Error: Build deps install failed"; exit 1; }
    apt remove -y golang-go golang || true
    rm -rf /usr/local/go /usr/bin/go /usr/local/bin/go || true
    wget -v https://go.dev/dl/go$GO_VERSION.linux-amd64.tar.gz -O go.tar.gz || { echo "Error: Go download failed"; exit 1; }
    tar -C /usr/local -xzf go.tar.gz || { echo "Error: Go extraction failed"; exit 1; }
    rm -fv go.tar.gz
    echo "Installed Go version: $(/usr/local/go/bin/go version)"
    export PATH="/usr/local/go/bin:$PATH"
    export CC="/usr/bin/gcc-10"
    export CXX="/usr/bin/g++-10"

    echo "Setting HIP-specific environment variables..."
    export CGO_CFLAGS="-I$ROCM_PATH/include -I$ROCM_PATH/hip/include"
    export CGO_LDFLAGS="-L$ROCM_PATH/lib -L$ROCM_PATH/lib64 -L$ROCM_PATH/hip/lib -lamdhip64 -lhiprtc -lrocm_smi64 -lroctx64 -lrocfft"
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
    echo "Success: Build environment configured for ROCm (HIP-only)"
}

patch_ollama() {
    echo "Patching Ollama to accept gfx803..."
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed"; exit 1; }
    
    # Find and patch the AMD GPU detection file
    GPU_FILE=$(find . -type f -name "*.go" -exec grep -l "amdgpu" {} + | head -1)
    if [ -z "$GPU_FILE" ]; then
        echo "Error: Could not locate GPU detection file with 'amdgpu'"
        ls -lR "$OLLAMA_DIR" > "$OLLAMA_DIR/dir_listing.txt"
        exit 1
    fi
    echo "Patching $GPU_FILE..."
    sed -i '/amdgpu too old/{s/return nil, err/return g, nil/}' "$GPU_FILE" || { echo "Error: Failed to patch $GPU_FILE"; exit 1; }
    
    # Find the main llama implementation file where the get_compiler needs to be added
    # In v0.5.12, the structure may have changed
    echo "Searching for appropriate file to add get_compiler shim..."
    
    # Look for llm implementation files in the repository
    LLM_FILES=$(find . -type f -name "*.go" -path "*/llm/*" | grep -v "_test.go")
    
    # Try to identify the main llm file for Linux
    LLM_LINUX_FILE=""
    for file in $LLM_FILES; do
        if grep -q "linux" "$file" || grep -q "unix" "$file"; then
            echo "Found potential Linux implementation file: $file"
            LLM_LINUX_FILE="$file"
            break
        fi
    done
    
    # If no specific Linux file found, use a general llm file
    if [ -z "$LLM_LINUX_FILE" ]; then
        LLM_LINUX_FILE=$(echo "$LLM_FILES" | head -1)
        echo "No specific Linux implementation found, using: $LLM_LINUX_FILE"
    fi
    
    if [ -z "$LLM_LINUX_FILE" ]; then
        echo "Error: Could not find any llm implementation files"
        exit 1
    fi
    
    # Create a new file for our get_compiler shim
    echo "Creating get_compiler shim in a separate file..."
    SHIM_FILE="llm/getcompiler_amd.go"
    mkdir -p $(dirname "$SHIM_FILE")
    
    # Determine the package name from the LLM_LINUX_FILE
    PACKAGE_NAME=$(grep "package" "$LLM_LINUX_FILE" | head -1 | awk '{print $2}')
    if [ -z "$PACKAGE_NAME" ]; then
        PACKAGE_NAME="llm"  # Default package name
    fi
    
    # Write the shim to a separate file
    cat > "$SHIM_FILE" << EOF
// Package $PACKAGE_NAME provides LLM functionality
package $PACKAGE_NAME

/*
#include <stdlib.h>
*/
import "C"

//export get_compiler
func get_compiler() *C.char {
    return C.CString("hipcc")  // Shim to bypass linker
}
EOF
    
    echo "Success: Created get_compiler shim in $SHIM_FILE"
    
    # Patch CMakeLists.txt if it exists, to support gfx803
    if [ -f "CMakeLists.txt" ]; then
        echo "Patching CMakeLists.txt for gfx803 support..."
        sed -i 's/set(CMAKE_CUDA_ARCHITECTURES .*/set(CMAKE_CUDA_ARCHITECTURES 37)/' CMakeLists.txt || echo "Warning: CMakeLists.txt patch failed (may be harmless)"
    fi
    
    echo "Success: Patched Ollama source"
}

build_ollama() {
    echo "Building Ollama with ROCm (HIP-only) support..."
    cd "$OLLAMA_DIR" || { echo "Error: Directory change failed"; exit 1; }
    echo "Generating Go files with HIP..."
    /usr/local/go/bin/go generate ./... || { echo "Error: Go generate failed"; exit 1; }
    echo "Building Ollama with HIP tags..."
    CGO_ENABLED=1 /usr/local/go/bin/go build -v -x -o ollama-rocm . || { echo "Error: Go build failed—check above for cgo errors"; exit 1; }
    if [ ! -f ollama-rocm ]; then
        echo "Error: Build failed—ollama-rocm binary not found"
        exit 1
    fi
    mv -v ollama-rocm "$BINDIR/ollama-rocm" || { echo "Error: Failed to move binary"; exit 1; }
    echo "Success: Ollama-rocm built and installed"
}

create_service() {
    echo "Creating systemd service..."
    tee /etc/systemd/system/ollama-rocm.service > /dev/null << EOF || { echo "Error: Service file creation failed"; exit 1; }
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
    systemctl daemon-reload || { echo "Error: Daemon reload failed"; exit 1; }
    systemctl enable ollama-rocm || { echo "Error: Service enable failed"; exit 1; }
    systemctl restart ollama-rocm || { echo "Error: Service restart failed"; exit 1; }
    echo "Success: Service created and started"
}

verify_installation() {
    echo "Verifying installation..."
    sleep 2
    $ROCM_PATH/bin/rocminfo | grep -A 5 "Name:.*gfx803" || { echo "Error: rocminfo failed or RX580 not detected"; exit 1; }
    $ROCM_PATH/bin/rocm-smi || { echo "Error: rocm-smi failed"; exit 1; }
    systemctl status ollama-rocm --no-pager || { echo "Error: Service status check failed"; exit 1; }
    $BINDIR/ollama-rocm list || { echo "Error: Ollama test failed"; exit 1; }
    echo "Checking HIP usage in logs..."
    journalctl -u ollama-rocm --since "2 minutes ago" -l | grep -i "hip" && echo "Success: HIP detected in logs" || { echo "Error: No HIP usage detected—running on CPU"; exit 1; }
    echo "Testing GPU activity..."
    $BINDIR/ollama-rocm run llama2 "Hello world" & sleep 5; $ROCM_PATH/bin/rocm-smi | grep -q "[1-9]%" || { echo "Error: No GPU activity detected"; exit 1; }
    echo "Success: Verification complete—HIP confirmed"
}

main() {
    echo "Entering main function..."
    check_root
    setup_repository
    setup_build_env
    patch_ollama
    build_ollama
    create_service
    verify_installation
    echo "
Ollama ROCm Build Complete!
- Built with HIP-only support for RX 580 (gfx803)
- Binary: $BINDIR/ollama-rocm
- Service: ollama-rocm (port 11435)
Commands:
- ollama-rocm list : List models
- ollama-rocm run <model> : Run model
- journalctl -u ollama-rocm : View logs
Notes:
- Models stored in /usr/share/ollama-rocm/.ollama/models
- Runs standalone—no CUDA or Conda dependency
    "
}

trap 'echo "Error: Script failed at line $LINENO with exit code $?"' ERR

main






0.1\th1:fxVm/GzAzEWqLHuvctI91KS9hhNmmWOoWu0XTYJS7CA=\ndep\tgorgonia.org/vecf32\tv0.9.0\th1:PClazic1r+JVJ1dEzRXgeiVl4g1/Hf/w+wUSqnco1Xg=\ndep\tgorgonia.org/vecf64\tv0.9.0\th1:bgZDP5x0OzBF64PjMGC3EvTdOoMEcmfAh1VCUnZFm1A=\nbuild\t-buildmode=exe\nbuild\t-compiler=gc\nbuild\t-tags=hip\nbuild\tCGO_ENABLED=1\nbuild\tCGO_CFLAGS=\"-I/opt/rocm-6.3.3/include -I/opt/rocm-6.3.3/hip/include\"\nbuild\tCGO_CPPFLAGS=\nbuild\tCGO_CXXFLAGS=\nbuild\tCGO_LDFLAGS=\"-L/opt/rocm-6.3.3/lib -L/opt/rocm-6.3.3/lib64 -L/opt/rocm-6.3.3/hip/lib -lamdhip64 -lhiprtc -lrocm_smi64 -lroctx64 -lrocfft\"\nbuild\tGOARCH=amd64\nbuild\tGOOS=linux\nbuild\tGOAMD64=v1\nbuild\tvcs=git\nbuild\tvcs.revision=8c13cfa4dd35a79c983eb19b5ec2be7ffa220b69\nbuild\tvcs.time=2025-02-24T03:13:53Z\nbuild\tvcs.modified=true\n\xf92C1\x86\x18 r\x00\x82B\x10A\x16\xd8\xf2"
EOF
mkdir -p $WORK/b001/exe/
cd .
GOROOT='/usr/local/go' /usr/local/go/pkg/tool/linux_amd64/link -o $WORK/b001/exe/a.out -importcfg $WORK/b001/importcfg.link -buildmode=exe -buildid=WYvGSu2zg-rY107PYzUo/Oq0ehDZHVDKwaYfR8K9t/3LunfNWeVWAICupiG5eG/WYvGSu2zg-rY107PYzUo -extld=/usr/bin/g++-10 $WORK/b001/_pkg_.a
/usr/local/go/pkg/tool/linux_amd64/buildid -w $WORK/b001/exe/a.out # internal
mv $WORK/b001/exe/a.out ollama-rocm
rm -rf $WORK/b001/
+ '[' '!' -f ollama-rocm ']'
+ mv -v ollama-rocm /usr/local/bin/ollama-rocm
renamed 'ollama-rocm' -> '/usr/local/bin/ollama-rocm'
+ echo 'Success: Ollama-rocm built and installed'
Success: Ollama-rocm built and installed
+ create_service
+ echo 'Creating systemd service...'
Creating systemd service...
+ tee /etc/systemd/system/ollama-rocm.service
+ systemctl daemon-reload
+ systemctl enable ollama-rocm
+ systemctl restart ollama-rocm
+ echo 'Success: Service created and started'
Success: Service created and started
+ verify_installation
+ echo 'Verifying installation...'
Verifying installation...
+ sleep 2
+ /opt/rocm-6.3.3/bin/rocminfo
+ grep -A 5 'Name:.*gfx803'
  Name:                    gfx803                             
  Uuid:                    GPU-XX                             
  Marketing Name:          Radeon RX 580 Series               
  Vendor Name:             AMD                                
  Feature:                 KERNEL_DISPATCH                    
  Profile:                 BASE_PROFILE                       
--
      Name:                    amdgcn-amd-amdhsa--gfx803          
      Machine Models:          HSA_MACHINE_MODEL_LARGE            
      Profiles:                HSA_PROFILE_BASE                   
      Default Rounding Mode:   NEAR                               
      Default Rounding Mode:   NEAR                               
      Fast f16:                TRUE                               
+ /opt/rocm-6.3.3/bin/rocm-smi


========================================== ROCm System Management Interface ==========================================
==================================================== Concise Info ====================================================
Device  Node  IDs              Temp    Power     Partitions          SCLK     MCLK    Fan  Perf  PwrCap  VRAM%  GPU%  
              (DID,     GUID)  (Edge)  (Socket)  (Mem, Compute, ID)                                                   
======================================================================================================================
0       1     0x67df,   49227  55.0°C  15.188W   N/A, N/A, 0         1167Mhz  300Mhz  0%   auto  135.0W  10%    0%    
======================================================================================================================
================================================ End of ROCm SMI Log =================================================
+ systemctl status ollama-rocm --no-pager
● ollama-rocm.service - Ollama Service (ROCm HIP-Only)
     Loaded: loaded (/etc/systemd/system/ollama-rocm.service; enabled; preset: enabled)
     Active: active (running) since Wed 2025-02-26 01:11:02 EST; 2s ago
   Main PID: 689693 (ollama-rocm)
      Tasks: 14 (limit: 153961)
     Memory: 14.6M (peak: 17.6M)
        CPU: 151ms
     CGroup: /system.slice/ollama-rocm.service
             └─689693 /usr/local/bin/ollama-rocm serve

Feb 26 01:11:02 HackPr07.1 ollama-rocm[689693]: time=2025-02-26T01:11:02.713-05:00 level=DEBUG source=amd_linux.go:101 msg="evaluating amdgpu node /sys/…/properties"
Feb 26 01:11:02 HackPr07.1 ollama-rocm[689693]: time=2025-02-26T01:11:02.713-05:00 level=DEBUG source=amd_linux.go:121 msg="detected CPU /sys/class/kfd/…/properties"
Feb 26 01:11:02 HackPr07.1 ollama-rocm[689693]: time=2025-02-26T01:11:02.713-05:00 level=DEBUG source=amd_linux.go:101 msg="evaluating amdgpu node /sys/…/properties"
Feb 26 01:11:02 HackPr07.1 ollama-rocm[689693]: time=2025-02-26T01:11:02.713-05:00 level=DEBUG source=amd_linux.go:206 msg="mapping amdgpu to drm sysfs … unique_id=0
Feb 26 01:11:02 HackPr07.1 ollama-rocm[689693]: time=2025-02-26T01:11:02.714-05:00 level=DEBUG source=amd_linux.go:240 msg=matched amdgpu=/sys/class/kfd…card1/device
Feb 26 01:11:02 HackPr07.1 ollama-rocm[689693]: time=2025-02-26T01:11:02.714-05:00 level=WARN source=amd_linux.go:309 msg="amdgpu too old gfx803" gpu=0
Feb 26 01:11:02 HackPr07.1 ollama-rocm[689693]: time=2025-02-26T01:11:02.714-05:00 level=INFO source=amd_linux.go:402 msg="no compatible amdgpu devices detected"
Feb 26 01:11:02 HackPr07.1 ollama-rocm[689693]: time=2025-02-26T01:11:02.714-05:00 level=INFO source=gpu.go:377 msg="no compatible GPUs were discovered"
Feb 26 01:11:02 HackPr07.1 ollama-rocm[689693]: releasing cuda driver library
Feb 26 01:11:02 HackPr07.1 ollama-rocm[689693]: time=2025-02-26T01:11:02.714-05:00 level=INFO source=types.go:130 msg="inference compute" id=0 library=c…="119.2 GiB"
Hint: Some lines were ellipsized, use -l to show in full.
+ /usr/local/bin/ollama-rocm list
NAME                ID              SIZE      MODIFIED     
llama2:latest       78e26419b446    3.8 GB    28 hours ago    
llama3.2:3b         a80c4f17acd5    2.0 GB    28 hours ago    
tinyllama:latest    2644915ede35    637 MB    28 hours ago    
+ echo 'Checking HIP usage in logs...'
Checking HIP usage in logs...
+ journalctl -u ollama-rocm --since '2 minutes ago' -l
+ grep -i hip
Feb 26 01:11:02 HackPr07.1 systemd[1]: Stopping ollama-rocm.service - Ollama Service (ROCm HIP-Only)...
Feb 26 01:11:02 HackPr07.1 systemd[1]: Stopped ollama-rocm.service - Ollama Service (ROCm HIP-Only).
Feb 26 01:11:02 HackPr07.1 systemd[1]: Started ollama-rocm.service - Ollama Service (ROCm HIP-Only).
Feb 26 01:11:02 HackPr07.1 ollama-rocm[689693]: 2025/02/26 01:11:02 routes.go:1205: INFO server config env="map[CUDA_VISIBLE_DEVICES: GPU_DEVICE_ORDINAL: HIP_VISIBLE_DEVICES:0 HSA_OVERRIDE_GFX_VERSION:8.0.3 HTTPS_PROXY: HTTP_PROXY: NO_PROXY: OLLAMA_DEBUG:true OLLAMA_FLASH_ATTENTION:false OLLAMA_GPU_OVERHEAD:0 OLLAMA_HOST:http://127.0.0.1:11435 OLLAMA_INTEL_GPU:false OLLAMA_KEEP_ALIVE:5m0s OLLAMA_KV_CACHE_TYPE: OLLAMA_LLM_LIBRARY: OLLAMA_LOAD_TIMEOUT:5m0s OLLAMA_MAX_LOADED_MODELS:0 OLLAMA_MAX_QUEUE:512 OLLAMA_MODELS:/usr/share/ollama-rocm/.ollama/models OLLAMA_MULTIUSER_CACHE:false OLLAMA_NEW_ENGINE:false OLLAMA_NOHISTORY:false OLLAMA_NOPRUNE:false OLLAMA_NUM_PARALLEL:0 OLLAMA_ORIGINS:[http://localhost https://localhost http://localhost:* https://localhost:* http://127.0.0.1 https://127.0.0.1 http://127.0.0.1:* https://127.0.0.1:* http://0.0.0.0 https://0.0.0.0 http://0.0.0.0:* https://0.0.0.0:* app://* file://* tauri://* vscode-webview://*] OLLAMA_SCHED_SPREAD:false ROCR_VISIBLE_DEVICES: http_proxy: https_proxy: no_proxy:]"
Feb 26 01:11:02 HackPr07.1 ollama-rocm[689693]: time=2025-02-26T01:11:02.586-05:00 level=DEBUG source=gpu.go:525 msg="gpu library search" globs="[/usr/local/lib/ollama/libcuda.so* /libcuda.so* /opt/rocm-6.3.3/lib/libcuda.so* /opt/rocm-6.3.3/lib64/libcuda.so* /opt/rocm-6.3.3/hip/lib/libcuda.so* /opt/rocm-6.3.3/lib/libcuda.so* /opt/rocm-6.3.3/lib64/libcuda.so* /opt/rocm-6.3.3/hip/lib/libcuda.so* /usr/local/cuda*/targets/*/lib/libcuda.so* /usr/lib/*-linux-gnu/nvidia/current/libcuda.so* /usr/lib/*-linux-gnu/libcuda.so* /usr/lib/wsl/lib/libcuda.so* /usr/lib/wsl/drivers/*/libcuda.so* /opt/cuda/lib*/libcuda.so* /usr/local/cuda/lib*/libcuda.so* /usr/lib*/libcuda.so* /usr/local/lib*/libcuda.so*]"
+ echo 'Success: HIP detected in logs'
Success: HIP detected in logs
+ echo 'Testing GPU activity...'
Testing GPU activity...
+ sleep 5
+ /usr/local/bin/ollama-rocm run llama2 'Hello world'

Hello there! It's nice to meet you. How can I help you+ /opt/rocm-6.3.3/bin/rocm-smi
+ grep -q '[1-9]%'
 today+ echo 'Error: No GPU activity detected'
Error: No GPU activity detected
+ exit 1
(darkpool-rocm) heathen-admin@HackPr07:~$ 

