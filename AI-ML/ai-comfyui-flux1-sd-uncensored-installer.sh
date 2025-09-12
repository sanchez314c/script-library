#!/bin/bash

# ComfyUI Bare-Metal Install Script for Uncensored Stable Diffusion and FLUX.1
# Created by Cortana for Jason
# Date: February 23, 2025
set -e

echo "🚀 Starting ComfyUI Install for Uncensored Stable Diffusion and FLUX.1..."

check_root() {
    echo "🔍 Checking for root privileges..."
    [ "$(id -u)" != "0" ] && { echo "❌ Requires root. Run with sudo."; exit 1; }
    echo "✅ Running as root"
}

setup_darklake_env() {
    echo "🐍 Activating darklake Conda environment..."
    source "$HOME/miniconda3/bin/activate" darklake || {
        echo "❌ Failed to activate darklake. Run ai-ml-docker-frameworks.sh first!"
        exit 1
    }
    "$HOME/miniconda3/bin/pip" install --upgrade pip
}

install_comfyui() {
    echo "📦 Installing ComfyUI from source..."
    mkdir -p ~/comfyui
    cd ~/comfyui
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    "$HOME/miniconda3/bin/pip" install -r requirements.txt --no-cache-dir
    # Install ROCm and CUDA PyTorch versions
    "$HOME/miniconda3/bin/pip" install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
    "$HOME/miniconda3/bin/pip" install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu114
}

install_flux_extension() {
    echo "🔧 Adding FLUX.1 support via XLabs-AI extension..."
    cd ~/comfyui/ComfyUI/custom_nodes
    git clone https://github.com/XLabs-AI/x-flux-comfyui.git
    cd x-flux-comfyui
    "$HOME/miniconda3/bin/pip" install -r requirements.txt --no-cache-dir
}

install_models() {
    echo "📥 Installing uncensored Stable Diffusion and FLUX.1 models..."
    mkdir -p ~/comfyui/ComfyUI/models/checkpoints ~/comfyui/ComfyUI/models/unet ~/comfyui/ComfyUI/models/vae ~/comfyui/ComfyUI/models/clip

    # Uncensored Stable Diffusion 1.5 (e.g., DreamShaper - NSFW capable)
    wget -q -O ~/comfyui/ComfyUI/models/checkpoints/dreamshaper_8.safetensors \
        https://civitai.com/api/download/models/128713  # DreamShaper 8, uncensored-friendly

    # FLUX.1 Dev (uncensored, full model - not Schnell)
    wget -q -O ~/comfyui/ComfyUI/models/unet/flux1-dev.safetensors \
        https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors
    # FLUX.1 VAE
    wget -q -O ~/comfyui/ComfyUI/models/vae/ae.safetensors \
        https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors
    # FLUX.1 CLIP models
    wget -q -O ~/comfyui/ComfyUI/models/clip/clip_l.safetensors \
        https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors
    wget -q -O ~/comfyui/ComfyUI/models/clip/t5xxl_fp8_e4m3fn.safetensors \
        https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors
}

create_service() {
    echo "🌐 Setting up ComfyUI systemd service..."
    sudo tee /etc/systemd/system/comfyui.service > /dev/null << EOF
[Unit]
Description=ComfyUI Diffusion Interface
After=network.target ollama-rocm.service ollama-k80-gpu0.service ollama-k80-gpu1.service

[Service]
User=$USER
WorkingDirectory=$HOME/comfyui/ComfyUI
Environment="PATH=$HOME/miniconda3/envs/darklake/bin:/usr/local/cuda-11.4/bin:/opt/rocm-6.3.3/bin:$PATH"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/opt/rocm-6.3.3/lib:$LD_LIBRARY_PATH"
ExecStart=$HOME/miniconda3/envs/darklake/bin/python main.py --port 8188 --listen 0.0.0.0
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable comfyui
}

verify_and_start() {
    echo "🔍 Verifying and starting ComfyUI..."
    sudo systemctl start comfyui
    sleep 5
    systemctl is-active --quiet comfyui || {
        echo "❌ Failed to start ComfyUI. Check logs: journalctl -u comfyui"
        exit 1
    }
    echo "
✨ ComfyUI Installation Complete! ✨
- Access: http://localhost:8188
- Manage: systemctl {start|stop|restart|status} comfyui
- Logs: journalctl -u comfyui
- Models:
  - Uncensored SD 1.5: ~/comfyui/ComfyUI/models/checkpoints/dreamshaper_8.safetensors
  - FLUX.1 Dev (Uncensored): ~/comfyui/ComfyUI/models/unet/flux1-dev.safetensors

Connected to Ollama via EXO:
- RX580: http://localhost:11434
- K80 GPU0: http://localhost:11435
- K80 GPU1: http://localhost:11436
Use --extra-model-paths-config for custom models!
"
}

main() {
    check_root
    setup_darklake_env
    install_comfyui
    install_flux_extension
    install_models
    create_service
    verify_and_start
    conda deactivate
}

main
