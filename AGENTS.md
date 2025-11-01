# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Script.Library is a comprehensive collection of automation scripts, utilities, and digital tools for power users, developers, and system administrators. Organized into functional categories, this repository contains over 200+ scripts covering AI/ML workflows, media processing, system administration, digital forensics, and more. All scripts follow standardized naming conventions and include auto-dependency management.

## Development Commands

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/sanchez314c/Script.Library.git
cd Script.Library

# Make scripts executable
find . -name "*.py" -exec chmod +x {} \;
find . -name "*.sh" -exec chmod +x {} \;

# Install global dependencies (optional)
pip install rich customtkinter Pillow

# Category-specific dependencies are auto-installed by individual scripts
```

### Running Scripts

```bash
# Universal launcher (recommended first approach)
python portal_gun_launcher.py --gui              # GUI interface
python portal_gun_launcher.py --list             # List all scripts
python portal_gun_launcher.py --category System  # Filter by category

# Direct execution
python AI-ML/ai-llm-terminal-chat-interface.py
python Images/media-fix-all-metadata-tags.py --directory /path/to/photos
bash System/system-inspection.sh

# Using the master launcher
python portal_gun_launcher.py --find "video"     # Find scripts matching pattern
```

### Testing and Validation

```bash
# Test script functionality
python System/system-inspection.sh               # System diagnostics
python GPUs/gpus-mps-verify.py                    # Verify GPU capabilities

# Test individual categories
python AI-ML/ai-llm-comparison-suite.py --test   # Run with test data

# Verify dependencies
python Images/media-extract-date-from-xmp-inplace.py --check-deps
```

## Architecture Overview

### Repository Structure

The repository is organized into functional categories, each containing specialized scripts:

**Core Categories:**
- **`AI-ML/`** - Artificial Intelligence and Machine Learning tools (LLM interfaces, model comparison, AI workflows)
- **`Audio/`** - Audio processing, conversion, and manipulation tools
- **`Data Scraping/`** - Web scraping and data collection utilities
- **`Documents/`** - PDF processing and document conversion tools
- **`Forensics/`** - Digital forensics and analysis utilities
- **`GPUs/`** - GPU computing, CUDA, Metal Performance Shaders testing
- **`Images/`** - Image processing, metadata management, batch operations
- **`JSON/`** - Structured data processing and manipulation utilities
- **`Mobile/`** - Mobile device integration and media transfer tools
- **`NLP/`** - Natural Language Processing and resume generation suites
- **`System/`** - System administration, automation, and monitoring tools
- **`Video/`** - Video processing, conversion, and manipulation utilities

### Universal Design Patterns

**Standardized Script Headers:**
All Python scripts follow the mandatory header format with:
- ASCII art banner with script information
- Category-specific naming: `[category]-[action]-[target]-[modifier].py`
- Auto-dependency management with graceful installation
- Comprehensive docstrings with usage examples
- Version control and metadata

**Auto-Dependency Management:**
```python
# Common pattern in all scripts
try:
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    print("📦 Installing Rich...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    RICH_AVAILABLE = True
```

**Error Handling and Logging:**
- Standardized emoji status indicators (✅ ❌ ⚠️ 🔄)
- Comprehensive logging with timestamp and context
- Graceful fallbacks for missing dependencies
- Rollback capabilities for destructive operations

### Portal Gun Launcher Architecture

**`portal_gun_launcher.py`** serves as the master control interface:

- **CLI Interface**: Rich-enhanced terminal navigation with tables and panels
- **GUI Interface**: CustomTkinter-based graphical launcher
- **Script Discovery**: Automatic script categorization and metadata extraction
- **Search and Filter**: Pattern-based script discovery
- **Direct Execution**: Launch scripts with arguments through the interface

### Key Technical Components

**Multi-Platform Support:**
- macOS-native integration with system APIs
- Ubuntu compatibility with package manager integration
- Cross-platform scripts with platform-specific optimizations

**Performance Optimization:**
- GPU acceleration via CUDA, Metal, OpenCL where available
- Multiprocessing and threading for large datasets
- Intelligent caching for repeated operations
- Memory-efficient handling of large files

**Security and Privacy:**
- Local-only processing with no external data transmission
- Automatic backup creation before destructive operations
- Dry-run modes for operation preview
- Integrity verification with checksums

## Common Development Patterns

### Script Categories and Capabilities

**AI/ML Scripts:**
- LLM comparison and benchmarking suites
- Terminal chat interfaces for multiple providers
- AI-powered resume and document generation
- Prompt optimization and testing tools

**Media Processing Scripts:**
- Batch image/video conversion and processing
- Metadata extraction and manipulation
- Format conversion with quality optimization
- Automated organization and file management

**System Administration Scripts:**
- Docker container management and backup
- Network monitoring and device discovery
- Large file operations with integrity verification
- System diagnostics and performance monitoring

**Forensics Scripts:**
- File structure analysis and metadata comparison
- Steganography detection and analysis
- Network traffic investigation
- Image forensics with EXIF/XMP analysis

### Dependency Management

**Per-Category Requirements:**
Each category contains its own `requirements.txt` with specific dependencies:
- `Images/requirements.txt` - Pillow, exifread, tqdm, rich
- `System/requirements.txt` - Rich, pandas, system tools
- `Video/requirements.txt` - FFmpeg bindings, OpenCV
- `AI-ML/requirements.txt` - OpenAI, Anthropic, Google AI SDKs

**Auto-Installation Pattern:**
```python
def check_install_dependencies():
    required_packages = ['rich', 'Pillow', 'exifread']
    for package in required_packages:
        if not importlib.util.find_spec(package):
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
```

### Configuration and Customization

**Environment Variables:**
- `USE_GPU=true/false` - Enable/disable GPU acceleration
- `SCRIPT_LIBRARY_LOG_LEVEL` - Set logging verbosity
- Category-specific environment variables for API keys and paths

**Global Configuration:**
Scripts respect common configuration patterns:
- YAML/JSON configuration files in script directories
- Command-line argument parsing with help text
- Interactive configuration prompts for first-time setup

## Important Notes

### Script Execution
- All Python scripts are executable and can be run directly
- Shell scripts include comprehensive error checking
- Most scripts support `--help` for usage information
- Portal gun launcher provides unified interface for all scripts

### Platform-Specific Considerations
- macOS scripts integrate with native APIs (Core Image, AVFoundation)
- Linux scripts use platform-appropriate package managers
- GPU acceleration available on compatible hardware
- System tools auto-installed via Homebrew (macOS) or apt (Ubuntu)

### Data Safety
- Scripts create automatic backups before operations
- Dry-run modes available for testing
- Comprehensive logging for audit trails
- Rollback capabilities where feasible

### Performance Characteristics
- Scripts optimized for both small and large datasets
- Memory usage scales with input size
- GPU acceleration reduces processing time significantly
- Parallel processing used for batch operations

## Standardization Guide

All scripts follow the established standardization guide (`STANDARDIZATION_GUIDE.md`):
- Mandatory ASCII art headers with script metadata
- Consistent naming conventions
- Standardized emoji usage for status indicators
- Comprehensive documentation and error handling
- Version control and author attribution