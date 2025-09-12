#!/bin/bash

# Whisper TTS/STT Bare-Metal Install Script for darklake Conda Env
# Version: 1.0.1 - Built by Cortana (via Grok 3, xAI) for Jason
# Date: February 24, 2025

# Enable verbosity and error handling
set -x  # Trace every command
set -e  # Exit on any error

echo "🚀 Starting Whisper TTS/STT Install in darklake Conda Environment..."

check_root() {
    echo "🔍 Checking for root privileges..."
    if [ "$(id -u)" != "0" ]; then
        echo "❌ Error: Script requires root privileges. Run with sudo."
        exit 1
    fi
    echo "✅ Success: Running as root"
}

setup_darklake_env() {
    CONDA_HOME="/home/${SUDO_USER:-$USER}/miniconda3"  # Matches user’s Conda install
    echo "🐍 Activating darklake Conda environment..."
    echo "Sourcing Conda activate from $CONDA_HOME/bin/activate..."
    sudo -u "${SUDO_USER:-$USER}" bash -c "source $CONDA_HOME/bin/activate darklake" || { echo "❌ Error: Failed to activate darklake—run ai-ml-docker-frameworks.sh first!"; exit 1; }
    echo "Upgrading pip in darklake..."
    sudo -u "${SUDO_USER:-$USER}" "$CONDA_HOME/bin/pip" install --upgrade pip || { echo "❌ Error: Pip upgrade failed"; exit 1; }
    echo "✅ Success: darklake environment activated and pip upgraded"
}

install_whisper() {
    CONDA_HOME="/home/${SUDO_USER:-$USER}/miniconda3"
    echo "📦 Installing Whisper and Faster-Whisper..."
    echo "Installing Whisper from git..."
    sudo -u "${SUDO_USER:-$USER}" "$CONDA_HOME/bin/pip" install git+https://github.com/openai/whisper.git --no-cache-dir || { echo "❌ Error: Whisper install failed"; exit 1; }
    echo "Installing Faster-Whisper..."
    sudo -u "${SUDO_USER:-$USER}" "$CONDA_HOME/bin/pip" install faster-whisper --no-cache-dir || { echo "❌ Error: Faster-Whisper install failed"; exit 1; }
    echo "✅ Success: Whisper and Faster-Whisper installed"
}

install_piper() {
    CONDA_HOME="/home/${SUDO_USER:-$USER}/miniconda3"
    echo "📦 Installing Piper TTS..."
    echo "Installing piper-tts..."
    sudo -u "${SUDO_USER:-$USER}" "$CONDA_HOME/bin/pip" install piper-tts --no-cache-dir || { echo "❌ Error: Piper-tts install failed"; exit 1; }
    echo "Creating Piper models directory at /home/${SUDO_USER:-$USER}/AI/Piper/models..."
    sudo -u "${SUDO_USER:-$USER}" mkdir -pv "/home/${SUDO_USER:-$USER}/AI/Piper/models" || { echo "❌ Error: Failed to create Piper models directory"; exit 1; }
    echo "Downloading Piper model: en_US-libritts-high.onnx..."
    sudo -u "${SUDO_USER:-$USER}" wget -v -O "/home/${SUDO_USER:-$USER}/AI/Piper/models/en_US-libritts-high.onnx" \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx || { echo "❌ Error: Piper model download failed"; exit 1; }
    echo "Downloading Piper config: en_US-libritts-high.onnx.json..."
    sudo -u "${SUDO_USER:-$USER}" wget -v -O "/home/${SUDO_USER:-$USER}/AI/Piper/models/en_US-libritts-high.onnx.json" \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx.json || { echo "❌ Error: Piper config download failed"; exit 1; }
    echo "✅ Success: Piper TTS installed with en_US-libritts-high voice"
}

create_transcribe_script() {
    echo "📝 Creating Whisper TTS/STT script..."
    echo "Writing script to /home/${SUDO_USER:-$USER}/AI/Whisper/whisper_tts_stt.py..."
    sudo -u "${SUDO_USER:-$USER}" mkdir -p "/home/${SUDO_USER:-$USER}/AI/Whisper"
    sudo -u "${SUDO_USER:-$USER}" bash -c "cat << 'EOF' > /home/${SUDO_USER:-$USER}/AI/Whisper/whisper_tts_stt.py" || { echo "❌ Error: Failed to write whisper_tts_stt.py"; exit 1; }
#!/usr/bin/env python3
import sys
import whisper
import faster_whisper
from piper import PiperVoice

def transcribe_audio(audio_path, model_size="medium", use_faster=False):
    print(f"Loading STT model: {model_size}...")
    if use_faster:
        model = faster_whisper.WhisperModel(model_size, device="auto", compute_type="float16")
        segments, info = model.transcribe(audio_path, beam_size=5)
        text = " ".join(segment.text for segment in segments)
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    else:
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path)
        text = result["text"]
        print(f"Detected language: {result['language']}")
    print("Transcription:", text)
    return text

def synthesize_speech(text, output_path="output.wav"):
    print("Synthesizing speech with Piper...")
    voice = PiperVoice.load("~/AI/Piper/models/en_US-libritts-high.onnx")
    with open(output_path, "wb") as f:
        voice.synthesize(text, f)
    print(f"Speech saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python whisper_tts_stt.py <audio_file> [model_size] [faster] [tts_output]")
        print("Example: python whisper_tts_stt.py audio.mp3 medium true output.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "medium"
    use_faster = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False
    tts_output = sys.argv[4] if len(sys.argv) > 4 else None
    
    text = transcribe_audio(audio_file, model_size, use_faster)
    if tts_output:
        synthesize_speech(text, tts_output)
EOF
    echo "Making script executable..."
    sudo -u "${SUDO_USER:-$USER}" chmod +x "/home/${SUDO_USER:-$USER}/AI/Whisper/whisper_tts_stt.py" || { echo "❌ Error: Failed to make script executable"; exit 1; }
    echo "Creating symlink in /usr/local/bin..."
    ln -sfv "/home/${SUDO_USER:-$USER}/AI/Whisper/whisper_tts_stt.py" /usr/local/bin/whisper-tts-stt || { echo "❌ Error: Failed to create symlink"; exit 1; }
    echo "✅ Success: Whisper TTS/STT script created and linked"
}

create_service() {
    echo "🌐 Setting up Whisper STT server (optional REST API)..."
    echo "Writing service file to /etc/systemd/system/whisper-stt.service..."
    sudo tee /etc/systemd/system/whisper-stt.service > /dev/null << EOF || { echo "❌ Error: Failed to write service file"; exit 1; }
[Unit]
Description=Whisper Speech-to-Text Server
After=network.target

[Service]
User=${SUDO_USER:-$USER}
WorkingDirectory=/home/${SUDO_USER:-$USER}
Environment="PATH=/home/${SUDO_USER:-$USER}/miniconda3/envs/darklake/bin:/usr/local/cuda-11.4/bin:/opt/rocm-6.3.3/bin:$PATH"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/opt/rocm-6.3.3/lib:$LD_LIBRARY_PATH"
ExecStart=/home/${SUDO_USER:-$USER}/miniconda3/envs/darklake/bin/python -m faster_whisper.server --port 5000 --model medium
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
    echo "Reloading systemd daemon..."
    systemctl daemon-reload || { echo "❌ Error: Daemon reload failed"; exit 1; }
    echo "Enabling whisper-stt service..."
    systemctl enable whisper-stt || { echo "❌ Error: Service enable failed"; exit 1; }
    echo "✅ Success: Whisper STT service created"
}

verify_and_start() {
    CONDA_HOME="/home/${SUDO_USER:-$USER}/miniconda3"
    echo "🔍 Verifying Whisper TTS/STT installation..."
    echo "Activating darklake for verification..."
    sudo -u "${SUDO_USER:-$USER}" bash -c "source $CONDA_HOME/bin/activate darklake" || { echo "❌ Error: Failed to activate darklake"; exit 1; }
    echo "Checking Whisper..."
    sudo -u "${SUDO_USER:-$USER}" "$CONDA_HOME/envs/darklake/bin/python" -c "import whisper; print('Whisper OK:', whisper.__version__)" || { echo "❌ Error: Whisper install failed"; exit 1; }
    echo "Checking Faster-Whisper..."
    sudo -u "${SUDO_USER:-$USER}" "$CONDA_HOME/envs/darklake/bin/python" -c "import faster_whisper; print('Faster-Whisper OK:', faster_whisper.__version__)" || { echo "❌ Error: Faster-Whisper install failed"; exit 1; }
    echo "Checking Piper..."
    sudo -u "${SUDO_USER:-$USER}" "$CONDA_HOME/envs/darklake/bin/python" -c "from piper import PiperVoice; print('Piper OK')" || { echo "❌ Error: Piper install failed"; exit 1; }
    
    echo "Starting whisper-stt service..."
    systemctl start whisper-stt || { echo "❌ Error: Failed to start whisper-stt service"; exit 1; }
    echo "Waiting 5 seconds for service to stabilize..."
    sleep 5
    echo "Checking service status..."
    if systemctl is-active --quiet whisper-stt; then
        echo "✅ Success: Whisper STT server running at http://localhost:5000"
    else
        echo "⚠️ Warning: STT server not started—run manually if needed"
    fi
    
    echo "
✨ Whisper TTS/STT Installation Complete! ✨
- CLI: whisper-tts-stt <audio_file> [model_size] [faster] [tts_output]
  - Example: whisper-tts-stt audio.mp3 medium true output.wav
- STT Server: http://localhost:5000 (REST API via Faster-Whisper)
- Manage: systemctl {start|stop|restart|status} whisper-stt
- Logs: journalctl -u whisper-stt

Models: tiny, base, small, medium, large (default: medium)
TTS: Piper with en_US-libritts-high voice at ~/AI/Piper/models
Uses RX580 (ROCm 6.3.3) and K80s (CUDA 11.4) automatically!
    "
}

main() {
    echo "🔧 Entering main function..."
    check_root
    setup_darklake_env
    install_whisper
    install_piper
    create_transcribe_script
    create_service
    verify_and_start
    echo "Deactivating darklake environment..."
    sudo -u "${SUDO_USER:-$USER}" "$CONDA_HOME/bin/conda" deactivate || { echo "❌ Error: Failed to deactivate darklake"; exit 1; }
    echo "✅ Success: Installation process complete"
}

# Trap errors with line numbers
trap 'echo "❌ Script failed at line $LINENO with exit code $?"' ERR

main