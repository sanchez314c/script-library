# AI & Machine Learning Tools

Collection of installers and utilities for AI/ML frameworks and applications.

## Directory Structure

### Ollama/
Large Language Model runtime installers for different hardware configurations
- **Official/** - Official Ollama installers from ollama.com
- **ROCm-Builds/** - Custom builds for AMD RX580 GPUs with ROCm support  
- **CUDA-Builds/** - Custom builds for NVIDIA K80 GPUs with CUDA support

### ComfyUI/
Stable Diffusion and FLUX.1 image generation tools

### OpenWebUI/
Web-based interfaces for LLM interaction

### Whisper/
Speech-to-text and text-to-speech tools

### ExO-Distributed/
Distributed AI inference tools for GPU clustering

## Current Available Tools

### Ollama ROCm Builds
- `ai-ollama-rocm-rx580-v51.sh` - Latest ROCm build (v1.5.1)
- `ai-ollama-rocm-rx580-v49.sh` - Stable ROCm build (v1.4.9)

### Official Installers
- `ai-ollama-official-installer-v00.sh` - Official Ollama installer

## Hardware Compatibility

**AMD RX580:** Use ROCm builds in `Ollama/ROCm-Builds/`
**NVIDIA K80:** Use CUDA builds in `Ollama/CUDA-Builds/` 
**General Use:** Use official installers in `Ollama/Official/`

## Security & Quality
- All scripts audited for security issues
- Enhanced error handling and logging added
- Removed duplicates and organized by chronological development
- Version numbering reflects actual development timeline

## Installation Notes
- ROCm builds require ROCm 6.3+ installed first
- CUDA builds require CUDA toolkit 11.4+ 
- Official installers work with system GPU detection
- Monitor GPU usage during compilation phases