#!/bin/bash

# Open WebUI Bare-Metal Install Script for darklake Conda Env
# Created by Cortana for Jason
# Installs Open WebUI from source in darklake env on Ubuntu 24.04.1
# Date: February 23, 2025

set -e

echo "🚀 Starting Open WebUI Install in darklake Conda Environment..."

# Prerequisites
install_prerequisites() {
    echo "🛠️ Installing system prerequisites..."
    sudo apt update
    sudo apt install -y \
        build-essential git curl libssl-dev libffi-dev \
        ffmpeg  # For audio/video features
}

# Activate darklake env and install deps
setup_darklake_env() {
    echo "🐍 Activating darklake Conda environment..."
    source "$HOME/miniconda3/bin/activate" darklake
    if [ $? -ne 0 ]; then
        echo "❌ Failed to activate darklake env. Ensure AI-ML-SETUP has run!"
        exit 1
    fi
    pip install --upgrade pip
}

# Install Open WebUI
install_open_webui() {
    echo "📦 Installing Open WebUI from source in darklake..."
    mkdir -p ~/open-webui
    cd ~/open-webui
    git clone https://github.com/open-webui/open-webui.git src
    cd src
    pip install -r requirements.txt --no-cache-dir
    pip install ".[ffmpeg]"  # Optional audio/video support
}

# Configure environment
configure_env() {
    echo "⚙️ Configuring Open WebUI..."
    cd ~/open-webui/src
    cp .env.example .env
    sed -i 's|^#DATA_DIR=.*|DATA_DIR=../data|' .env
    sed -i 's|^#OLLAMA_API_BASE_URL=.*|OLLAMA_API_BASE_URL="http://localhost:11434"|' .env
    echo "WEBUI_PORT=3000" >> .env
}

# Create systemd service
create_service() {
    echo "🌐 Setting up Open WebUI as a systemd service..."
    sudo tee /etc/systemd/system/open-webui.service << EOF
[Unit]
Description=Open WebUI AI Interface
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/open-webui/src
Environment="PATH=$HOME/miniconda3/envs/darklake/bin:$HOME/miniconda3/bin:$PATH"
ExecStart=$HOME/miniconda3/envs/darklake/bin/python main.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable open-webui
}

# Verify and start
verify_and_start() {
    echo "🔍 Verifying and starting Open WebUI..."
    sudo systemctl start open-webui
    sleep 5
    if systemctl is-active --quiet open-webui; then
        echo "✅ Open WebUI running at http://localhost:3000"
    else
        echo "❌ Failed to start Open WebUI. Check logs with: journalctl -u open-webui"
        exit 1
    fi
    echo "
✨ Open WebUI Installation Complete! ✨
- Access: http://localhost:3000
- Manage: systemctl {start|stop|restart|status} open-webui
- Logs: journalctl -u open-webui
- Config: ~/open-webui/src/.env
- Data: ~/open-webui/data

Running in darklake Conda env, connected to Ollama at http://localhost:11434.
Edit .env to tweak settings!
"
}

# Main
main() {
    install_prerequisites
    setup_darklake_env
    install_open_webui
    configure_env
    create_service
    verify_and_start
    conda deactivate
}

main
