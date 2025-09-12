#!/bin/bash

# PyGPT Installation Script
# Version: 1.0.0 - Built by Cortana for Jason
# Date: March 8, 2025

# Enable command tracing
set -x

echo "Starting PyGPT Installation on Ubuntu..."

# Log file for installation
LOG_FILE="/tmp/pygpt_install_log.txt"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

check_root() {
    echo "Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "Error: Requires root. Run with sudo."
        exit 1
    fi
    echo "Success: Running as root"
}

install_prerequisites() {
    echo "Installing prerequisites..."
    apt update || { echo "WARNING: apt update failed, but continuing..."; }
    apt install -y snapd || { echo "ERROR: Failed to install snapd. Attempting to continue anyway."; }
}

install_pygpt() {
    echo "Installing PyGPT via snap..."
    snap install pygpt
    
    if ! command -v pygpt &> /dev/null; then
        echo "Snap install failed. Trying alternate method..."
        # Hypothetical fallback: GitHub source install (adjust URL if real by 2025)
        apt install -y python3-pip git
        git clone https://github.com/pygpt/pygpt.git /tmp/pygpt
        cd /tmp/pygpt
        pip3 install -r requirements.txt
        python3 setup.py install
        rm -rf /tmp/pygpt
    fi
    
    if command -v pygpt &> /dev/null; then
        echo "SUCCESS: PyGPT installed"
    else
        echo "ERROR: PyGPT installation failed"
        exit 1
    fi
}

main() {
    check_root
    install_prerequisites
    install_pygpt
    
    echo "
PyGPT Installation Complete!
- If installed via snap, run 'pygpt' to start
- If installed via source, ensure Python environment is set up
- Check $LOG_FILE for details
"
}

main
