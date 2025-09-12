#!/bin/bash

# Ollama ROCm Install Script for RX 580 (HIP-Only)
# Version: 1.2.31-FIXED - Fixed Original with ROCm 6.3.3
# Date: March 6, 2025

set -x
set -e

echo "Starting Ollama ROCm Install for RX 580..."

OLLAMA_DIR="/usr/share/ollama"
BINDIR="/usr/bin"
ROCM_PATH="/opt/rocm-6.3.3"
LOG_FILE="/home/$SUDO_USER/Desktop/ollama-install.log"

check_root() {
    echo "Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    # Fix: Handle unset $SUDO_USER
    if [ -z "$SUDO_USER" ] || [ "$SUDO_USER" = "root" ]; then
        SUDO_USER="ollama"
        adduser --system --group --home /home/ollama ollama || true
        LOG_FILE="/home/ollama/Desktop/ollama-install.log"
    fi
    echo "Success: Running as root with user $SUDO_USER"
}

install_ollama() {
    echo "Installing Ollama with ROCm support..."
    
    # Install the standard Ollama binary first
    echo "Installing standard Ollama binary..."
    curl -fsSL https://ollama.com/install.sh | sh || { echo "Error: Official install script failed"; exit 1; }
    
    # Verify the binary location
    echo "Verifying binary location..."
    which ollama || { echo "Error: Ollama binary not found after install"; exit 1; }
    
    # Move binary to /usr/bin if needed
    if [ ! -f "$BINDIR/ollama" ]; then
        OLLAMA_PATH=$(which ollama)
        if [ -n "$OLLAMA_PATH" ]; then
            echo "Moving ollama from $OLLAMA_PATH to $BINDIR"
            cp -v $OLLAMA_PATH $BINDIR/ollama
            chmod +x $BINDIR/ollama
        else
            echo "Error: Ollama binary not found"
            exit 1
        fi
    fi
    
    # Download the ROCm libraries
    echo "Downloading ROCm libraries..."
    curl -L https://ollama.com/download/ollama-linux-amd64-rocm.tgz -o /tmp/ollama-linux-amd64-rocm.tgz || { echo "Error: Download of ROCm libraries failed"; exit 1; }
    
    # Extract the libraries
    echo "Extracting ROCm libraries..."
    tar -C /usr -xzf /tmp/ollama-linux-amd64-rocm.tgz || { echo "Error: Extraction failed"; exit 1; }
    
    # Create Ollama models directory with proper permissions
    echo "Creating data directories with proper permissions..."
    mkdir -p /home/${SUDO_USER}/.ollama/models
    chown -R ${SUDO_USER}:${SUDO_USER} /home/${SUDO_USER}/.ollama
    
    echo "Success: Ollama installed with ROCm libraries"
}

setup_service() {
    echo "Setting up Ollama service..."
    
    # Create main service file
    echo "Creating main service file..."
    tee /etc/systemd/system/ollama.service > /dev/null << EOF
[Unit]
Description=Ollama Service (ROCm HIP-Only)
After=network-online.target

[Service]
ExecStart=${BINDIR}/ollama serve
User=${SUDO_USER}
Group=${SUDO_USER}
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF

    # Create override with all environment variables
    echo "Creating environment variables override..."
    mkdir -p /etc/systemd/system/ollama.service.d/
    tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null << EOF
[Service]
Environment="OLLAMA_HOME=/home/${SUDO_USER}/.ollama"
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="ROCM_PATH=${ROCM_PATH}"
Environment="LD_LIBRARY_PATH=${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${ROCM_PATH}/hip/lib:/usr/lib/ollama/rocm"
Environment="HSA_OVERRIDE_GFX_VERSION=8.0.3"
Environment="ROC_ENABLE_PRE_VEGA=1"
Environment="PYTORCH_ROCM_ARCH=gfx803"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="CUDA_VISIBLE_DEVICES="
Environment="OLLAMA_DEBUG=true"
Environment="OLLAMA_LLM_LIBRARY=rocm"
Environment="HIP_IGNORE_HW_VERSION=1"
Environment="HSA_ENABLE_SDMA=0"
Environment="OLLAMA_CUSTOM_LIBRARIES_PATH=/usr/lib/ollama"
EOF

    # Reload systemd
    systemctl daemon-reload
    
    # Enable and start service
    systemctl enable ollama
    systemctl restart ollama
    
    echo "Waiting for service to start..."
    sleep 5
    
    echo "Success: Service setup complete"
}

verify_installation() {
    echo "Verifying installation..."
    
    # Check if ROCm sees the RX 580
    echo "Checking if RX 580 is detected..."
    ${ROCM_PATH}/bin/rocminfo | grep -A 5 "Name:.*gfx803" || { echo "Error: RX580 not detected"; exit 1; }
    
    # Check current GPU status
    echo "Current GPU status:"
    ${ROCM_PATH}/bin/rocm-smi
    
    # Check service status
    echo "Checking service status..."
    systemctl status ollama --no-pager
    
    # Verify Ollama is responding
    echo "Checking if Ollama is responding..."
    # Fix: Configurable timeout
    TIMEOUT=${TIMEOUT:-60}
    for i in {1..20}; do
        if ${BINDIR}/ollama list; then
            echo "Success: Ollama is responding!"
            break
        fi
        echo "Attempt $i: Waiting for Ollama to start (3s)..."
        sleep 3
        
        if [ $i -eq 5 ]; then
            echo "Restarting service..."
            systemctl restart ollama
        fi
        
        if [ $((i * 3)) -ge $TIMEOUT ]; then
            echo "Warning: Ollama not responding after multiple attempts"
            echo "Checking logs:"
            journalctl -u ollama --no-pager -n 50
            
            echo "Trying manual startup..."
            systemctl stop ollama
            sudo -u ${SUDO_USER} bash -c "cd /home/${SUDO_USER} && OLLAMA_HOME=/home/${SUDO_USER}/.ollama HSA_OVERRIDE_GFX_VERSION=8.0.3 ROC_ENABLE_PRE_VEGA=1 HIP_VISIBLE_DEVICES=0 OLLAMA_LLM_LIBRARY=rocm HIP_IGNORE_HW_VERSION=1 HSA_ENABLE_SDMA=0 OLLAMA_CUSTOM_LIBRARIES_PATH=/usr/lib/ollama LD_LIBRARY_PATH=${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${ROCM_PATH}/hip/lib:/usr/lib/ollama/rocm ${BINDIR}/ollama serve &"
            sleep 5
            
            if ! ${BINDIR}/ollama list; then
                echo "Error: Manual startup also failed"
                exit 1
            fi
        fi
    done
    
    # Test a model 
    echo "Testing model..."
    ${BINDIR}/ollama pull tinyllama:latest
    ${BINDIR}/ollama run tinyllama "Hello, this is a test for RX 580 GPU acceleration" &
    TEST_PID=$!
    sleep 5
    ${ROCM_PATH}/bin/rocm-smi
    kill $TEST_PID 2>/dev/null || true
    wait $TEST_PID 2>/dev/null || true
    
    echo "Installation verification complete!"
}

main() {
    echo "Entering main function..."
    # Fix: Add ROCm pre-check
    echo "Checking ROCm installation..."
    [ -d "$ROCM_PATH" ] || { echo "Error: ROCm 6.3.3 not found at $ROCM_PATH"; exit 1; }
    $ROCM_PATH/bin/rocminfo || { echo "Error: ROCm not functional"; exit 1; }
    # Fix: Log to file
    exec > >(tee -a "$LOG_FILE") 2>&1
    check_root
    install_ollama
    setup_service
    verify_installation
    echo "
Ollama ROCm Install Complete!
- Built with HIP-only support for RX 580 (gfx803)
- Binary: ${BINDIR}/ollama
- ROCm Libraries: /usr/lib/ollama/rocm
- Service: ollama (port 11434)
Commands:
- ollama list : List models
- ollama run <model> : Run model
- journalctl -u ollama : View logs
    "
}

trap 'echo "Error: Script failed at line $LINENO with exit code $?"' ERR

main