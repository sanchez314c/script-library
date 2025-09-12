#!/bin/bash

# EXO Labs AI Cluster Setup Script
# Created by Cortana for Jason
# Pools RX580 (ROCm) and K80 (CUDA) GPUs via EXO
# Date: February 23, 2025
set -e

echo "🚀 Starting EXO Labs AI Cluster Setup..."

check_root() {
    echo "🔍 Checking for root privileges..."
    [ "$(id -u)" != "0" ] && { echo "❌ Requires root. Run with sudo."; exit 1; }
    echo "✅ Running as root"
}

install_dependencies() {
    echo "📦 Installing EXO dependencies..."
    sudo apt update
    sudo apt install -y libzmq3-dev protobuf-compiler
    sudo -u "${SUDO_USER:-$USER}" "/home/${SUDO_USER:-$USER}/miniconda3/bin/pip" install pyzmq protobuf --user
    echo "✅ Dependencies installed"
}

install_exo() {
    echo "🔧 Installing EXO Labs..."
    sudo -u "${SUDO_USER:-$USER}" mkdir -p "/home/${SUDO_USER:-$USER}/AI/EXO"
    cd "/home/${SUDO_USER:-$USER}/AI/EXO"
    sudo -u "${SUDO_USER:-$USER}" git clone https://github.com/exo-explore/exo.git
    cd exo
    sudo -u "${SUDO_USER:-$USER}" "/home/${SUDO_USER:-$USER}/miniconda3/bin/pip" install -e . --user
    sudo -u "${SUDO_USER:-$USER}" mkdir -p "/home/${SUDO_USER:-$USER}/.config/exo" "/home/${SUDO_USER:-$USER}/AI/EXO/logs"
    echo "✅ EXO installed"
}

configure_exo() {
    echo "⚙️ Configuring EXO cluster..."
    sudo -u "${SUDO_USER:-$USER}" bash -c "cat << 'EOF' > /home/${SUDO_USER:-$USER}/.config/exo/config.yaml
cluster:
  name: \"jason-ai-cluster\"
  discovery:
    method: \"auto\"
    port: 29500
  networking:
    interface: \"auto\"
    protocol: \"tcp\"
  
compute:
  cuda:
    enabled: true
    visible_devices: \"0,1\"  # K80 GPUs
    memory_limit: \"90%\"
  rocm:
    enabled: true
    visible_devices: \"0\"    # RX580
    memory_limit: \"90%\"
    
scheduling:
  strategy: \"round-robin\"
  priority:
    - \"gpu\"
    - \"cpu\"
    
models:
  cache_dir: \"~/AI/EXO/models\"
  default_backend: \"auto\"
  
monitoring:
  enabled: true
  port: 8080
  metrics:
    - \"gpu_utilization\"
    - \"memory_usage\"
    - \"network_bandwidth\"
    
logging:
  level: \"info\"
  file: \"~/AI/EXO/logs/exo.log\"
EOF"

    sudo tee /etc/systemd/system/exo-cluster.service > /dev/null << EOF
[Unit]
Description=EXO Labs AI Cluster
After=network.target

[Service]
User=${SUDO_USER:-$USER}
Environment="PATH=/home/${SUDO_USER:-$USER}/miniconda3/bin:/usr/local/cuda-11.4/bin:/opt/rocm-6.3.3/bin:$PATH"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/opt/rocm-6.3.3/lib:$LD_LIBRARY_PATH"
WorkingDirectory=/home/${SUDO_USER:-$USER}/AI/EXO
ExecStart=/home/${SUDO_USER:-$USER}/.local/bin/exo-cluster start
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable exo-cluster
    sudo systemctl restart exo-cluster
    echo "✅ EXO configured"
}

sync_ollama() {
    echo "🧠 Syncing with existing Ollama instances..."
    # Use pre-built Ollama binaries from prior scripts
    sudo systemctl stop ollama-rocm ollama-k80-gpu0 ollama-k80-gpu1 2>/dev/null || true
    sudo systemctl disable ollama-rocm ollama-cuda0 ollama-cuda1 2>/dev/null || true  # Disable any old env-based services

    sudo tee /etc/systemd/system/ollama-rocm.service > /dev/null << EOF
[Unit]
Description=Ollama ROCm Instance (RX580)
After=network.target exo-cluster.service

[Service]
User=${SUDO_USER:-$USER}
WorkingDirectory=/home/${SUDO_USER:-$USER}/AI/EXO
ExecStart=/usr/local/bin/ollama-rocm serve
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_MODELS=/usr/share/ollama-rocm/.ollama/models"

[Install]
WantedBy=multi-user.target
EOF

    sudo tee /etc/systemd/system/ollama-k80-gpu0.service > /dev/null << EOF
[Unit]
Description=Ollama CUDA Instance (K80 GPU0)
After=network.target exo-cluster.service

[Service]
User=${SUDO_USER:-$USER}
WorkingDirectory=/home/${SUDO_USER:-$USER}/AI/EXO
ExecStart=/usr/local/bin/ollama-k80-gpu0 serve
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0:11435"
Environment="OLLAMA_MODELS=/usr/share/ollama-k80-gpu0/.ollama/models"
Environment="CUDA_VISIBLE_DEVICES=0"

[Install]
WantedBy=multi-user.target
EOF

    sudo tee /etc/systemd/system/ollama-k80-gpu1.service > /dev/null << EOF
[Unit]
Description=Ollama CUDA Instance (K80 GPU1)
After=network.target exo-cluster.service

[Service]
User=${SUDO_USER:-$USER}
WorkingDirectory=/home/${SUDO_USER:-$USER}/AI/EXO
ExecStart=/usr/local/bin/ollama-k80-gpu1 serve
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0:11436"
Environment="OLLAMA_MODELS=/usr/share/ollama-k80-gpu1/.ollama/models"
Environment="CUDA_VISIBLE_DEVICES=1"

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable ollama-rocm ollama-k80-gpu0 ollama-k80-gpu1
    sudo systemctl restart ollama-rocm ollama-k80-gpu0 ollama-k80-gpu1
    echo "✅ Ollama synced"
}

create_utility_scripts() {
    echo "📝 Creating utility scripts..."
    sudo -u "${SUDO_USER:-$USER}" bash -c "cat << 'EOF' > /home/${SUDO_USER:-$USER}/AI/EXO/manage-cluster.sh
#!/bin/bash
case \"\$1\" in
    start)
        sudo systemctl start exo-cluster ollama-rocm ollama-k80-gpu0 ollama-k80-gpu1
        ;;
    stop)
        sudo systemctl stop exo-cluster ollama-rocm ollama-k80-gpu0 ollama-k80-gpu1
        ;;
    restart)
        sudo systemctl restart exo-cluster ollama-rocm ollama-k80-gpu0 ollama-k80-gpu1
        ;;
    status)
        sudo systemctl status exo-cluster ollama-rocm ollama-k80-gpu0 ollama-k80-gpu1 --no-pager
        echo \"Cluster Nodes:\"; exo-cli list-nodes || echo \"❌ EXO CLI failed\"
        echo \"GPU Status:\"; exo-cli gpu-status || echo \"❌ EXO CLI failed\"
        ;;
    monitor)
        xdg-open http://localhost:8080 || echo \"❌ Browser launch failed\"
        ;;
    *)
        echo \"Usage: \$0 {start|stop|restart|status|monitor}\"
        exit 1
        ;;
esac
EOF"
    sudo -u "${SUDO_USER:-$USER}" chmod +x "/home/${SUDO_USER:-$USER}/AI/EXO/manage-cluster.sh"

    sudo -u "${SUDO_USER:-$USER}" bash -c "cat << 'EOF' > /home/${SUDO_USER:-$USER}/AI/EXO/deploy-model.sh
#!/bin/bash
[ -z \"\$1\" ] && { echo \"Usage: \$0 <model_name> [num_gpus]\"; echo \"Example: \$0 llama3-70b auto\"; exit 1; }
MODEL_NAME=\$1
NUM_GPUS=\${2:-\"auto\"}
exo-cli deploy --model \"\$MODEL_NAME\" --gpus \"\$NUM_GPUS\" --optimize-memory --enable-monitoring
EOF"
    sudo -u "${SUDO_USER:-$USER}" chmod +x "/home/${SUDO_USER:-$USER}/AI/EXO/deploy-model.sh"
    echo "✅ Utility scripts created"
}

verify_installation() {
    echo "🔍 Verifying installation..."
    sleep 2
    sudo -u "${SUDO_USER:-$USER}" exo-cli list-gpus || { echo "❌ EXO CLI failed—check install"; exit 1; }
    systemctl status ollama-rocm --no-pager || echo "❌ Ollama ROCm failed"
    systemctl status ollama-k80-gpu0 --no-pager || echo "❌ Ollama K80 GPU0 failed"
    systemctl status ollama-k80-gpu1 --no-pager || echo "❌ Ollama K80 GPU1 failed"
    echo "
✨ EXO Labs AI Cluster Setup Complete! ✨
- RX580 (ROCm): 0.0.0.0:11434
- K80 GPU0 (CUDA): 0.0.0.0:11435
- K80 GPU1 (CUDA): 0.0.0.0:11436
- EXO pools all GPUs

Commands:
- ~/AI/EXO/manage-cluster.sh status : Check status
- ~/AI/EXO/deploy-model.sh <model> : Deploy model
- ~/AI/EXO/manage-cluster.sh monitor : Open dashboard

Config: ~/.config/exo/config.yaml
Logs: ~/AI/EXO/logs/exo.log
"
}

main() {
    check_root
    install_dependencies
    install_exo
    configure_exo
    sync_ollama
    create_utility_scripts
    verify_installation
}

main