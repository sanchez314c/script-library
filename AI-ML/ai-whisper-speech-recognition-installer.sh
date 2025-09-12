#!/bin/bash

# Whisper Bare-Metal Install Script for darklake Conda Env
# Created by Cortana for Jason
# Date: February 23, 2025
set -e

echo "🚀 Starting Whisper Install for Local Audio Processing..."

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

install_whisper() {
    echo "📦 Installing Whisper from source..."
    "$HOME/miniconda3/bin/pip" install git+https://github.com/openai/whisper.git --no-cache-dir
    # Optional: Faster-Whisper for better performance
    "$HOME/miniconda3/bin/pip" install faster-whisper --no-cache-dir
}

create_transcribe_script() {
    echo "📝 Creating Whisper transcription script..."
    cat << 'EOF' > ~/whisper_transcribe.py
#!/usr/bin/env python3
import sys
import whisper
import faster_whisper
import torch

def transcribe_audio(audio_path, model_size="medium", use_faster=False):
    print(f"Loading {model_size} model...")
    if use_faster:
        model = faster_whisper.WhisperModel(model_size, device="auto", compute_type="float16")
        segments, info = model.transcribe(audio_path, beam_size=5)
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        text = " ".join(segment.text for segment in segments)
    else:
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path)
        text = result["text"]
        print(f"Detected language: {result['language']}")
    print("Transcription:", text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python whisper_transcribe.py <audio_file> [model_size] [faster]")
        print("Example: python whisper_transcribe.py audio.mp3 medium true")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "medium"
    use_faster = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False
    
    transcribe_audio(audio_file, model_size, use_faster)
EOF
    chmod +x ~/whisper_transcribe.py
    sudo ln -sf ~/whisper_transcribe.py /usr/local/bin/whisper_transcribe
}

create_service() {
    echo "🌐 Setting up Whisper systemd service (optional REST API)..."
    sudo tee /etc/systemd/system/whisper.service > /dev/null << EOF
[Unit]
Description=Whisper Speech-to-Text Server
After=network.target

[Service]
User=$USER
WorkingDirectory=$HOME
Environment="PATH=$HOME/miniconda3/envs/darklake/bin:/usr/local/cuda-11.4/bin:/opt/rocm-6.3.3/bin:$PATH"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/opt/rocm-6.3.3/lib:$LD_LIBRARY_PATH"
ExecStart=$HOME/miniconda3/envs/darklake/bin/python -m faster_whisper.server --port 5000 --model medium
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable whisper
}

verify_and_start() {
    echo "🔍 Verifying Whisper installation..."
    source "$HOME/miniconda3/bin/activate" darklake
    python -c "import whisper; print('Whisper OK:', whisper.__version__)" || {
        echo "❌ Whisper install failed!"
        exit 1
    }
    python -c "import faster_whisper; print('Faster-Whisper OK:', faster_whisper.__version__)" || {
        echo "❌ Faster-Whisper install failed!"
        exit 1
    }
    
    # Optional: Start service
    sudo systemctl start whisper
    sleep 5
    systemctl is-active --quiet whisper && echo "✅ Whisper server running at http://localhost:5000" || echo "⚠️ Whisper server not started—run manually if needed"
    
    echo "
✨ Whisper Installation Complete! ✨
- CLI: whisper_transcribe <audio_file> [model_size] [faster]
  - Example: whisper_transcribe audio.mp3 medium true
- Server (optional): http://localhost:5000 (REST API via Faster-Whisper)
- Manage: systemctl {start|stop|restart|status} whisper
- Logs: journalctl -u whisper

Models: tiny, base, small, medium, large (default: medium)
Uses RX580 (ROCm 6.3.3) and K80s (CUDA 11.4) automatically!
"
}

main() {
    check_root
    setup_darklake_env
    install_whisper
    create_transcribe_script
    create_service
    verify_and_start
    conda deactivate
}

main
