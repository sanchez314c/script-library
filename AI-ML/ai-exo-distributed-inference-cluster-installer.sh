#!/bin/bash

# EXO Labs AI Cluster Setup Script
# Created by Cortana for Jason
# This script should be run AFTER CUDA and ROCm are properly installed and verified

set -e

echo "🚀 Starting EXO Labs AI Cluster Setup..."

# Required dependencies
install_dependencies() {
    echo "📦 Installing core dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
        python3-pip \
        python3-venv \
        git \
        build-essential \
        cmake \
        pkg-config \
        libzmq3-dev \
        protobuf-compiler \
        golang-go

    # Install Python packages
    python3 -m pip install --upgrade pip
    pip3 install \
        torch \
        transformers \
        accelerate \
        protobuf \
        pyzmq \
        numpy \
        psutil
}

# Install EXO Labs
install_exo() {
    echo "🔧 Installing EXO Labs..."
    
    # Create installation directory
    mkdir -p ~/exo-cluster
    cd ~/exo-cluster
    
    # Clone the repository
    git clone https://github.com/exo-explore/exo.git
    cd exo
    
    # Install EXO
    pip3 install -e .
    
    # Create configuration directory
    mkdir -p ~/.config/exo
}

# Configure EXO cluster
configure_exo() {
    echo "⚙️ Configuring EXO cluster..."
    
    # Create cluster configuration
    cat << 'EOF' > ~/.config/exo/config.yaml
cluster:
  name: "jason-ai-cluster"
  discovery:
    method: "auto"
    port: 29500
  networking:
    interface: "auto"
    protocol: "tcp"
  
compute:
  cuda:
    enabled: true
    visible_devices: "all"
    memory_limit: "90%"
  rocm:
    enabled: true
    visible_devices: "all"
    memory_limit: "90%"
    
scheduling:
  strategy: "round-robin"
  priority:
    - "gpu"
    - "cpu"
    
models:
  cache_dir: "~/exo-cluster/models"
  default_backend: "auto"
  
monitoring:
  enabled: true
  port: 8080
  metrics:
    - "gpu_utilization"
    - "memory_usage"
    - "network_bandwidth"
    
logging:
  level: "info"
  file: "~/exo-cluster/logs/exo.log"
EOF

    # Create systemd service
    sudo tee /etc/systemd/system/exo-cluster.service << 'EOF'
[Unit]
Description=EXO Labs AI Cluster
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
Environment="PATH=/usr/local/cuda/bin:/opt/rocm/bin:$PATH"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/rocm/lib"
WorkingDirectory=~/exo-cluster
ExecStart=/usr/local/bin/exo-cluster start
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
}

# Create utility scripts
create_utility_scripts() {
    echo "📝 Creating utility scripts..."
    
    # Create cluster management script
    cat << 'EOF' > ~/exo-cluster/manage-cluster.sh
#!/bin/bash

case "$1" in
    start)
        sudo systemctl start exo-cluster
        ;;
    stop)
        sudo systemctl stop exo-cluster
        ;;
    restart)
        sudo systemctl restart exo-cluster
        ;;
    status)
        sudo systemctl status exo-cluster
        echo "Cluster Nodes:"
        exo-cli list-nodes
        echo "GPU Status:"
        exo-cli gpu-status
        ;;
    monitor)
        echo "Opening monitoring dashboard..."
        xdg-open http://localhost:8080
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|monitor}"
        exit 1
        ;;
esac
EOF

    chmod +x ~/exo-cluster/manage-cluster.sh
    
    # Create model deployment script
    cat << 'EOF' > ~/exo-cluster/deploy-model.sh
#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <model_name> [num_gpus]"
    echo "Example: $0 llama3-70b 4"
    exit 1
fi

MODEL_NAME=$1
NUM_GPUS=${2:-"auto"}

echo "Deploying $MODEL_NAME across $NUM_GPUS GPUs..."
exo-cli deploy \
    --model $MODEL_NAME \
    --gpus $NUM_GPUS \
    --optimize-memory \
    --enable-monitoring
EOF

    chmod +x ~/exo-cluster/deploy-model.sh
}

# Verify installation
verify_installation() {
    echo "🔍 Verifying installation..."
    
    # Check EXO installation
    if ! command -v exo-cli &> /dev/null; then
        echo "❌ EXO CLI not found!"
        exit 1
    fi
    
    # Test cluster formation
    echo "Testing cluster formation..."
    exo-cli test-cluster
    
    # Verify GPU detection
    echo "Verifying GPU detection..."
    exo-cli list-gpus
    
    echo "
✨ EXO Labs AI Cluster Setup Complete! ✨

Your cluster is now ready to distribute AI workloads across all available GPUs.

Quick Start Commands:
1. Start cluster:    ~/exo-cluster/manage-cluster.sh start
2. Check status:     ~/exo-cluster/manage-cluster.sh status
3. Deploy model:     ~/exo-cluster/deploy-model.sh <model_name> [num_gpus]
4. Monitor cluster:  ~/exo-cluster/manage-cluster.sh monitor

Configuration file location: ~/.config/exo/config.yaml
Logs location: ~/exo-cluster/logs/exo.log

Example model deployment:
$ ~/exo-cluster/deploy-model.sh llama3-70b auto

The cluster will automatically:
- Discover all available GPUs (CUDA and ROCm)
- Distribute workloads optimally
- Handle inter-GPU communication
- Provide monitoring and metrics

To add more nodes to the cluster, run this same script on other machines
and they will auto-discover each other on the local network.

Remember: The first run of a model will download it to the cache directory.
"
}

# Main installation
main() {
    install_dependencies
    install_exo
    configure_exo
    create_utility_scripts
    verify_installation
}

# Start installation
main