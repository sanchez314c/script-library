# 🛠️ Script.Library: Your Digital Lockpick & Swiss Army Chainsaw

<p align="center">
  <img src="https://raw.githubusercontent.com/sanchez314c/Script.Library/main/.images/script-library-hero.png" alt="Script Library Hero" width="600">
</p>

**All Kinds of Scripts for MacOS/Ubuntu, GenAI, Media and other random stuff.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![macOS](https://img.shields.io/badge/macOS-Compatible-blue.svg)](https://www.apple.com/macos/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-Compatible-orange.svg)](https://ubuntu.com/)
[![Shell](https://img.shields.io/badge/Shell-Bash_Zsh-green.svg)](https://www.gnu.org/software/bash/)

## 🎯 Overview

Welcome to Script.Library, the ultimate collection of automation scripts, utilities, and digital tools for power users, developers, and system administrators. This repository is your one-stop arsenal for solving everyday computing challenges, from AI/ML workflows to media processing, system administration, and forensic analysis.

Built by professionals, for professionals who refuse to do things manually when they can be automated intelligently.

## ✨ Script Categories

### 🤖 **AI & Machine Learning**
Advanced scripts for AI development, model training, and LLM operations
- **LLM Comparison Suite**: Benchmark different language models
- **AI Terminal Chat Interface**: Command-line AI interactions
- **Prompt Optimizer**: Enhance AI prompt effectiveness
- **Resume Builder**: AI-powered resume generation and optimization

### 🎵 **Audio Processing**
Professional audio manipulation and automation tools
- **Apple Podcasts Exporter**: Extract podcast libraries
- **Batch Audio Converter**: Mass conversion to M4A
- **Audio Deduplicator**: Find and remove duplicate audio files
- **Spotify Lyrics Integration**: Menubar lyrics display
- **TTS System Voices**: GUI for system voice management

### 📊 **Data Scraping & Analysis**
Intelligent data collection and processing utilities
- **Google Search AI Scraper**: AI-enhanced search result extraction
- **LLM Raw Data Scraper**: Collect training data for models
- **PDF Document Scraper**: Extract structured data from PDFs
- **Web Data Collector**: Advanced web scraping capabilities

### 📄 **Document Processing**
Comprehensive document manipulation and conversion tools
- **PDF Converter**: Universal document to PDF conversion
- **PDF Merger**: Combine multiple PDFs intelligently
- **PDF Review System**: Automated document analysis
- **Batch Document Processor**: Mass document operations

### 🔍 **Digital Forensics**
Professional-grade forensic analysis and investigation tools
- **File Analyzer**: Deep file structure analysis
- **Image Forensics**: Metadata and steganography detection
- **Metadata Comparator**: Cross-reference file metadata
- **Traffic Analyzer**: Network traffic investigation
- **Steganography Detector**: Hidden data detection

### 🎮 **GPU Computing**
High-performance computing and GPU utilization scripts
- **CUDA Information**: Complete GPU capability analysis
- **LLM Testing Suite**: Test Ollama and OpenWebUI deployments
- **MPS Benchmark**: Apple Metal Performance Shaders testing
- **PyTorch Verification**: GPU tensor operation validation
- **TensorFlow GPU**: Mixed precision and performance testing

### 🖼️ **Image & Media Processing**
Professional media manipulation and organization tools
- **Metadata Management**: Complete EXIF/XMP handling
- **Image Conversion**: Batch processing and format conversion
- **Date Extraction**: Intelligent timestamp recovery
- **Rotation Correction**: Automated image orientation fixing
- **Media Organization**: Intelligent file sorting and naming

### 📊 **JSON & Data Processing**
Structured data manipulation and processing utilities
- **ChatGPT Conversation Processor**: Extract and analyze chat logs
- **Google Contacts Cleaner**: Sanitize and organize contacts
- **Google Takeout Processor**: Parse Google data exports
- **JSON Structure Validator**: Data integrity verification

### 📱 **Mobile Integration**
Mobile device interaction and media transfer tools
- **Media Transfer**: Automated mobile to desktop sync
- **Device Detection**: Identify connected mobile devices
- **Wireless Transfer**: Over-the-air file operations

### 🧠 **Natural Language Processing**
Advanced NLP and text processing capabilities
- **Resume Builder Suite**: AI-powered career document generation
- **Skills Extractor**: Parse and categorize professional skills
- **Job Market Analyzer**: AI-driven career market analysis
- **Text Processing**: Advanced linguistic analysis tools

### 🖥️ **System Administration**
Comprehensive system management and automation utilities
- **Docker Management**: Container backup and restoration
- **Network Utilities**: VPN management, speed testing, discovery
- **File Operations**: Advanced file moving, organization, cleanup
- **System Monitoring**: Resource usage and performance tracking
- **Backup Solutions**: Automated backup and recovery systems

### 🎬 **Video Processing**
Professional video manipulation and conversion tools
- **Frame Interpolation**: AI-enhanced video smoothing
- **Format Conversion**: Universal video format support
- **Corruption Detection**: Identify damaged video files
- **Thumbnail Generation**: Automated preview creation
- **Audio Extraction**: Separate audio tracks from video

## 🚀 Quick Start

### Prerequisites
- **macOS 10.15+** or **Ubuntu 18.04+**
- **Python 3.8+** with pip
- **Bash/Zsh** shell environment
- **Git** for repository management

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sanchez314c/Script.Library.git
   cd Script.Library
   ```

2. **Install Python dependencies**
   ```bash
   # Install global requirements
   pip install -r requirements.txt
   
   # Install category-specific requirements
   cd AI-ML && pip install -r requirements.txt
   cd ../Audio && pip install -r requirements.txt
   # Repeat for other categories as needed
   ```

3. **Set up environment**
   ```bash
   # Make scripts executable
   find . -name "*.py" -exec chmod +x {} \;
   find . -name "*.sh" -exec chmod +x {} \;
   
   # Add to PATH (optional)
   echo 'export PATH="$PATH:$(pwd)"' >> ~/.bashrc
   source ~/.bashrc
   ```

4. **Test installation**
   ```bash
   # Test a simple script
   python System/system-inspection.sh
   
   # Test GPU capabilities (if available)
   python GPUs/gpus-mps-verify.py
   ```

## 🎮 Usage Examples

### AI & Machine Learning
```bash
# Compare different LLM models
python AI-ML/ai-llm-comparison-suite.py --models "gpt-4,claude-3,llama-2" --test-prompts prompts.txt

# Generate optimized resume
python NLP/nlps-advanced-ai-resume-generator.py --input resume_data.json --style professional

# Terminal AI chat
python AI-ML/ai-llm-terminal-chat-interface.py --model claude-3-sonnet
```

### Media Processing
```bash
# Batch convert audio files
python Audio/audio-batch-converter-to-m4a.py --input /path/to/audio --quality high

# Fix image metadata and organization
python Images/media-fix-all-metadata-tags.py --directory /path/to/photos --recursive

# Process video collection
python Video/video-frame-interpolator.py --input /path/to/videos --fps 60
```

### System Administration
```bash
# Monitor large file movements
python System/large-folder-move.py --source /old/location --dest /new/location --verify

# Network discovery and analysis
python System/find-network-rokus.py --scan-range 192.168.1.0/24

# Docker container management
bash System/backup-docker-containers.sh --destination /backup/docker
```

### Forensics & Analysis
```bash
# Analyze suspicious files
python Forensics/forensics-file-analyzer.py --target /path/to/file --deep-scan

# Detect steganography
python Forensics/forensics-steganography-detector.py --image suspicious.jpg --algorithm all

# Network traffic analysis
python Forensics/forensics-traffic-analyzer.py --pcap network_capture.pcap
```

## 🏗️ Architecture

```
Script.Library/
├── AI-ML/                    # Artificial Intelligence & Machine Learning
│   ├── LLM Analyzer (NLP)/  # Natural Language Processing tools
│   ├── ai-llm-*.py          # LLM interaction scripts
│   └── ai-prompt-optimizer.py
│
├── Audio/                    # Audio processing and manipulation
│   ├── audio-*.py           # Audio conversion and processing
│   └── audio_utils.py       # Shared audio utilities
│
├── Data Scraping/           # Web scraping and data collection
│   ├── data-*.py           # Various scraping utilities
│   └── link_tree_data.json # Scraped data examples
│
├── Documents/               # Document processing and conversion
│   └── documents-*.py      # PDF and document utilities
│
├── Forensics/              # Digital forensics and analysis
│   └── forensics-*.py     # Investigation and analysis tools
│
├── GPUs/                   # GPU computing and acceleration
│   ├── gpus-*.py          # GPU testing and benchmarking
│   └── gpu-utils-move.sh  # GPU utility management
│
├── Images/                 # Image processing and organization
│   ├── media-*.py         # Image manipulation utilities
│   └── requirements.txt   # Image processing dependencies
│
├── JSON/                   # JSON and structured data processing
│   └── jsons-*.py         # JSON manipulation utilities
│
├── Mobile/                 # Mobile device integration
│   └── mobiles-*.py       # Mobile device utilities
│
├── NLP/                    # Natural Language Processing
│   ├── *-resume-builder*/  # Resume generation suites
│   └── nlps-*.py          # NLP utilities
│
├── System/                 # System administration and automation
│   ├── *.py               # Python system utilities
│   ├── *.sh               # Shell automation scripts
│   └── requirements.txt   # System utility dependencies
│
└── Video/                  # Video processing and conversion
    ├── *.py               # Python video utilities
    ├── *.sh               # Shell video scripts
    └── requirements.txt   # Video processing dependencies
```

## 🔧 Advanced Features

### Multi-Platform Support
- **macOS Optimizations**: Native integration with macOS APIs
- **Ubuntu Compatibility**: Full Linux support with package management
- **Cross-Platform Tools**: Scripts that work seamlessly on both platforms

### Performance Optimization
- **GPU Acceleration**: Leverage CUDA, Metal, and OpenCL
- **Parallel Processing**: Multi-threaded operations for large datasets
- **Memory Management**: Efficient handling of large files and datasets
- **Caching Systems**: Intelligent caching for repeated operations

### Integration Capabilities
```python
# Example: Integrate multiple script categories
from AI_ML.ai_llm_comparison_suite import LLMComparator
from Images.media_metadata_reporter import MetadataReporter
from System.large_folder_move import FolderMover

# Create automated workflow
workflow = ScriptWorkflow([
    LLMComparator(models=['gpt-4', 'claude-3']),
    MetadataReporter(format='json'),
    FolderMover(verify_integrity=True)
])

workflow.execute()
```

### Configuration Management
```yaml
# config.yaml - Global configuration
global_settings:
  temp_directory: "/tmp/script_library"
  log_level: "INFO"
  parallel_workers: 4

ai_ml:
  default_model: "claude-3-sonnet"
  api_timeout: 30
  
media_processing:
  quality_settings: "high"
  preserve_originals: true
  
system:
  backup_retention: 30
  verify_operations: true
```

## 🤝 Contributing

### Development Setup
```bash
# Clone for development
git clone https://github.com/sanchez314c/Script.Library.git
cd Script.Library

# Set up development environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

### Script Development Guidelines

1. **File Naming Convention**
   ```
   category-specific-function-version.py
   Example: ai-llm-comparison-suite.py
   ```

2. **Code Structure**
   ```python
   #!/usr/bin/env python3
   """
   Script Name: [Name]
   Description: [Brief description]
   Author: [Your name]
   Version: [Version number]
   License: MIT
   """
   
   import argparse
   import logging
   from pathlib import Path
   
   def main():
       # Main script logic
       pass
   
   if __name__ == "__main__":
       main()
   ```

3. **Documentation Requirements**
   - Comprehensive docstrings
   - Usage examples in comments
   - Error handling and logging
   - Configuration options
   - Dependencies clearly listed

### Adding New Scripts
```bash
# Create new category (if needed)
mkdir NewCategory
cd NewCategory

# Add requirements file
echo "# Dependencies for NewCategory" > requirements.txt

# Create your script
touch new-awesome-script.py
chmod +x new-awesome-script.py

# Add to README
echo "- **New Awesome Script**: Description of functionality" >> README.md
```

## 📊 Performance & Benchmarks

### Processing Capabilities
| Operation Type | Small Files (1-100) | Medium (100-1K) | Large (1K-10K) | Massive (10K+) |
|----------------|---------------------|------------------|----------------|------------------|
| Image Processing | <30s | 2-5m | 15-30m | 1-3h |
| Video Processing | 1-5m | 15-45m | 2-8h | 8-24h |
| Data Scraping | <10s | 30s-2m | 5-20m | 30m-2h |
| System Operations | <5s | 10-30s | 1-5m | 5-30m |
| AI/ML Tasks | 10s-2m | 2-10m | 10-60m | 1-6h |

### Resource Usage
- **Memory**: 100MB - 8GB depending on operation
- **CPU**: Single-core to full multi-core utilization
- **GPU**: Optional acceleration for compatible operations
- **Storage**: Minimal to several GB for large datasets

## 🔒 Security & Privacy

### Data Handling
- **No Data Collection**: Scripts process locally, no external data transmission
- **Privacy First**: User data remains on local systems
- **Secure Processing**: Encryption for sensitive operations
- **Access Controls**: Respect system permissions and user access levels

### Safe Operations
- **Backup Creation**: Automatic backups before destructive operations
- **Dry Run Mode**: Preview changes before execution
- **Rollback Capability**: Undo operations when possible
- **Integrity Verification**: Checksum validation for file operations

## 🐛 Troubleshooting

### Common Issues

**Permission Errors**
```bash
# Fix script permissions
find . -name "*.py" -exec chmod +x {} \;
find . -name "*.sh" -exec chmod +x {} \;

# Fix file ownership
sudo chown -R $USER:$USER Script.Library/
```

**Dependency Issues**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Install system-specific packages
# macOS
brew install ffmpeg imagemagick exiftool

# Ubuntu
sudo apt-get install ffmpeg imagemagick exiftool
```

**GPU Issues**
```bash
# Test GPU availability
python GPUs/gpus-mps-verify.py

# Check CUDA installation
python GPUs/gpus-cuda-info.py

# Fallback to CPU processing
export USE_GPU=false
```

## 📈 Roadmap

### Upcoming Features
- [ ] **Web Interface**: Browser-based script execution
- [ ] **Mobile Apps**: iOS/Android script runners
- [ ] **Cloud Integration**: AWS/GCP/Azure support
- [ ] **Workflow Builder**: Visual script chaining
- [ ] **Plugin System**: Third-party script integration

### Long-term Goals
- [ ] **AI-Powered Script Generation**: Generate scripts from natural language
- [ ] **Enterprise Edition**: Advanced features for businesses
- [ ] **Script Marketplace**: Community script sharing
- [ ] **Real-time Monitoring**: Live script execution monitoring

## 📞 Support & Community

### Getting Help
- **Documentation**: [Complete Wiki](https://github.com/sanchez314c/Script.Library/wiki)
- **Issues**: [Report Problems](https://github.com/sanchez314c/Script.Library/issues)
- **Discussions**: [Community Forum](https://github.com/sanchez314c/Script.Library/discussions)

### Professional Services
- **Custom Script Development**: Tailored automation solutions
- **Training & Workshops**: Learn advanced scripting techniques
- **Enterprise Consulting**: Large-scale automation strategies

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Open Source Community**: For tools, libraries, and inspiration
- **Contributors**: Everyone who has submitted scripts and improvements
- **Beta Testers**: Users who helped refine and test scripts
- **Platform Providers**: Apple, Ubuntu, and tool maintainers

## 🔗 Related Projects

- [Awesome Shell Scripts](https://github.com/awesome-lists/awesome-bash)
- [Python Automation Tools](https://github.com/python/automation)
- [System Administration Scripts](https://github.com/sysadmin/scripts)

---

<p align="center">
  <strong>Built by power users, for power users</strong><br>
  <sub>Automate everything, script the impossible, achieve the extraordinary.</sub>
</p>

---

**⭐ Star this repository if it makes your digital life easier!**