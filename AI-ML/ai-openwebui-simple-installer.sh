#!/bin/bash

# Open WebUI Bare-Metal Install Script for darklake Conda Env
# Created by Cortana for Jason
# Date: February 23, 2025
set -e

echo "🚀 Starting Open WebUI Install in darklake Conda Environment..."

check_root() {
    echo "🔍 Checking for root privileges..."
    [ "$(id -u)" != "0" ] && { echo "❌ Requires root. Run with sudo."; exit 1; }
    echo "✅ Running as root"
}

install_prerequisites() {
    echo "🛠️ Installing system prerequisites..."
    sudo apt update
    sudo apt install -y ffmpeg  # Audio/video support
}

setup_darklake_env() {
    # Determine the correct username: prefer SUDO_USER, then LOGNAME, then whoami
    TARGET_USER="${SUDO_USER:-${LOGNAME:-$(whoami)}}"
    if [ "$TARGET_USER" = "root" ]; then
        echo "⚠️ Warning: Running as root, trying to guess the real user..."
        # Try to find a non-root user from /home directory
        FIRST_USER=$(ls -1 /home | head -n 1)
        if [ -n "$FIRST_USER" ]; then
            TARGET_USER="$FIRST_USER"
            echo "ℹ️ Using first user found in /home: $TARGET_USER"
        fi
    fi

    echo "🐍 Activating darklake Conda environment for user $TARGET_USER..."
    sudo -u "$TARGET_USER" bash -c "source /home/$TARGET_USER/miniconda3/etc/profile.d/conda.sh && conda activate darklake" || {
        echo "❌ Failed to activate darklake env. Run 04-ai-ml-conda-installer.sh first!"
        exit 1
    }
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/pip" install --upgrade pip
}

install_open_webui() {
    echo "📦 Installing Open WebUI from source..."
    sudo -u "$TARGET_USER" mkdir -p "/home/$TARGET_USER/AI/OpenWeb-UI"
    cd "/home/$TARGET_USER/AI/OpenWeb-UI"
    sudo -u "$TARGET_USER" git clone https://github.com/open-webui/open-webui.git src
    cd src
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/pip" install -r requirements.txt --no-cache-dir
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/pip" install ".[ffmpeg]"
}

configure_env() {
    echo "⚙️ Configuring Open WebUI for multi-GPU Ollama..."
    cd "/home/$TARGET_USER/AI/OpenWeb-UI/src"
    sudo -u "$TARGET_USER" cp .env.example .env
    sudo -u "$TARGET_USER" sed -i 's|^#DATA_DIR=.*|DATA_DIR=../data|' .env
    # Point to EXO-managed Ollama instances
    sudo -u "$TARGET_USER" bash -c "cat << 'EOF' >> .env
OLLAMA_API_BASE_URL=\"http://localhost:11434,http://localhost:11435,http://localhost:11436\"
WEBUI_PORT=3000
EOF"
}

create_service() {
    echo "🌐 Setting up Open WebUI systemd service..."
    sudo tee /etc/systemd/system/open-webui.service > /dev/null << EOF
[Unit]
Description=Open WebUI AI Interface
After=network.target ollama-rocm.service ollama-k80-gpu0.service ollama-k80-gpu1.service

[Service]
User=$TARGET_USER
WorkingDirectory=/home/$TARGET_USER/AI/OpenWeb-UI/src
Environment="PATH=/home/$TARGET_USER/miniconda3/envs/darklake/bin:$PATH"
ExecStart=/home/$TARGET_USER/miniconda3/envs/darklake/bin/python main.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable open-webui
}

verify_and_start() {
    echo "🔍 Verifying and starting Open WebUI..."
    sudo systemctl start open-webui
    sleep 5
    systemctl is-active --quiet open-webui || {
        echo "❌ Failed to start Open WebUI. Check logs: journalctl -u open-webui"
        exit 1
    }
    echo "
✨ Open WebUI Installation Complete! ✨
- Access: http://localhost:3000
- Manage: systemctl {start|stop|restart|status} open-webui
- Logs: journalctl -u open-webui
- Config: /home/$TARGET_USER/AI/OpenWeb-UI/src/.env

Connected to Ollama instances:
- RX580: http://localhost:11434
- K80 GPU0: http://localhost:11435
- K80 GPU1: http://localhost:11436
Edit .env to adjust URLs!
"
}

main() {
    check_root
    install_prerequisites
    setup_darklake_env
    install_open_webui
    configure_env
    create_service
    verify_and_start
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/conda" deactivate
}

main