# Video Processing & Management Suite
## Professional-Grade Video Tools for macOS & Linux

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/sanchez314c/Script.Library)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![macOS](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)](https://www.apple.com/macos/)

A comprehensive collection of professional-grade video processing, analysis, and management tools optimized for performance. This suite provides everything needed for video conversion, quality analysis, metadata management, frame interpolation, and cloud integration.

## 🎯 Overview

This collection includes powerful video processing applications in both Python and Bash:

### **🐍 Python Applications**
- **🎬 Video Wall Systems** - Dynamic multi-video displays and streaming
- **🔧 Video Analysis Tools** - Corruption detection and quality assessment
- **🎞️ Frame Processing** - Advanced interpolation and sequence generation
- **📊 Metadata Management** - Comprehensive metadata extraction and reporting
- **🖼️ Thumbnail Generation** - High-quality preview creation

### **⚡ Bash Scripts**
- **🎨 Aspect Ratio Conversion** - Intelligent cropping and padding
- **✂️ Auto-Crop Detection** - Automatic black border removal
- **☁️ Cloud Integration** - Google Drive upload and organization
- **🔍 Corruption Scanning** - Hardware-accelerated quality verification
- **🎵 Audio Extraction** - High-quality audio format conversion

## ✨ Key Features

- **🍎 Cross-Platform Support** - Native macOS and Linux compatibility
- **⚡ Hardware Acceleration** - VideoToolbox, GPU encoding optimization
- **🔄 Multi-core Processing** - Automatically optimizes CPU usage for maximum performance
- **📦 Auto-dependency Installation** - Scripts automatically install required packages
- **📊 Progress Tracking** - Visual progress indicators for all long-running operations
- **🎯 GUI Integration** - Native dialogs via tkinter and zenity
- **🖥️ Desktop Logging** - All log files automatically placed on desktop for easy access
- **☁️ Cloud Integration** - Built-in Google Drive and streaming support

## 🚀 Quick Start

1. **Clone or download** this Video folder to your local machine
2. **Install system dependencies** (see Installation section below)
3. **Run any script** - Python dependencies install automatically on first use
4. **Check your Desktop** for log files after script execution

### Example Usage

```bash
# Python Applications
python3 video-wall-dynamic.py                    # Launch dynamic video wall
python3 video-corruption-scanner.py              # Scan for corrupted videos
python3 video-frame-interpolator.py              # Interpolate to 120 FPS
python3 video-generate-thumbnail.py              # Generate thumbnails
python3 video-moov-repair.py                     # Repair MP4 metadata

# Bash Scripts
./aspect-ratio-converter.sh                      # Convert 4:3 to 16:9
./auto-crop-detector.sh                          # Auto-detect and crop
./video-gdrive-move.sh                           # Upload to Google Drive
./video-extract-audio.sh                         # Extract high-quality audio
./video-corruption-scanner.sh                    # Hardware-accelerated scanning
```

## 📋 System Requirements

### Core Dependencies

**FFmpeg Suite** (Essential for all video operations):
```bash
# macOS
brew install ffmpeg

# Linux (Ubuntu/Debian)
sudo apt install ffmpeg
```

**GUI Components**:
```bash
# macOS
brew install zenity

# Linux (Ubuntu/Debian)
sudo apt install zenity
```

**Additional Tools**:
```bash
# macOS
brew install parallel bc exiftool rclone

# Linux (Ubuntu/Debian)
sudo apt install parallel bc exiftool rclone
```

### Python Requirements

- **Python 3.8+** (macOS typically includes this)
- **Standard Library Only** - No third-party packages required
- **GUI Support** - tkinter (included with Python)

### Hardware Acceleration

- **macOS**: VideoToolbox (built-in)
- **Linux**: Hardware acceleration auto-detected
- **GPU Support**: NVENC, VideoToolbox, QuickSync

## 🎬 Applications

### 🐍 Python Applications

#### 🎬 Video Wall Systems
**Files:** `video-wall-*.py`

Create dynamic multi-video displays with streaming support and real-time controls.

**Features:**
- Multi-monitor support with native resolution detection
- Real-time video wall configuration and management
- M3U8 streaming integration for live content
- Roku compatibility for casting and remote display
- Hardware-accelerated rendering for smooth playback

**Usage:**
```bash
python3 video-wall-dynamic.py [options]
python3 video-wall-roku.py [--ip ROKU_IP]
python3 video-wall-m3u8.py [--playlist URL]
```

---

#### 🔧 Video Analysis & Quality Control
**Files:** `video-corruption-scanner.py`, `video-ffprobe-error-check.py`, `video-moov-repair.py`

Professional-grade video quality assessment and repair tools.

**Features:**
- Hardware-accelerated corruption detection
- FFprobe integration for comprehensive analysis
- MP4 MOOV atom repair for damaged files
- Batch processing with detailed reporting
- Cross-platform compatibility

**Usage:**
```bash
python3 video-corruption-scanner.py [--directory DIR]
python3 video-ffprobe-error-check.py [--recursive]
python3 video-moov-repair.py [--input FILE]
```

---

#### 🎞️ Frame Processing & Interpolation
**Files:** `video-frame-interpolator.py`, `interpolate_frames.py`, `Video-Image-Sequence-Random.py`

Advanced frame processing for high-quality video enhancement.

**Features:**
- 120 FPS interpolation with motion analysis
- Random image sequence generation
- Frame-by-frame processing with progress tracking
- Quality preservation algorithms
- Memory-efficient processing for large videos

**Usage:**
```bash
python3 video-frame-interpolator.py [--input FILE] [--fps 120]
python3 interpolate_frames.py [--method advanced]
```

---

#### 📊 Metadata & Thumbnail Management
**Files:** `video-generate-thumbnail.py`, `metadata-mover.py`

Comprehensive metadata processing and thumbnail generation.

**Features:**
- High-quality thumbnail generation with customizable settings
- Metadata extraction and organization by date
- Batch processing for large video collections
- Embedded thumbnail support
- Cross-format compatibility

**Usage:**
```bash
python3 video-generate-thumbnail.py [--size 1920x1080]
python3 metadata-mover.py [--organize-by-date]
```

---

#### 🎨 Format Conversion & Processing
**Files:** `png-to-*.py`

Specialized tools for image sequence to video conversion.

**Features:**
- PNG sequence to MP4 conversion
- 4K and DMT quality presets
- Hardware acceleration support
- Batch processing capabilities
- Quality optimization algorithms

**Usage:**
```bash
python3 png-to-4k-video.py [--input-dir DIR]
python3 png-to-dmt-4k.py [--fps 60]
python3 png-to-mp4.py [--quality high]
```

### ⚡ Bash Scripts

#### 🎨 Aspect Ratio & Cropping Tools
**Files:** `aspect-ratio-converter.sh`, `auto-crop-detector.sh`, `video-crop-*.sh`

Professional video cropping and aspect ratio conversion.

**Features:**
- Intelligent 4:3 to 16:9 conversion with padding/cropping options
- Automatic black border detection and removal
- Portrait mode optimization for mobile content
- Quality preservation with hardware acceleration
- Batch processing with progress tracking

**Usage:**
```bash
./aspect-ratio-converter.sh [input_directory]
./auto-crop-detector.sh [video_file]
./video-crop-to-16x9.sh [input_dir]
./video-crop-to-9x16.sh [input_dir]
```

---

#### ☁️ Cloud Integration & File Management
**Files:** `video-gdrive-move.sh`

Automated cloud storage integration with bandwidth management.

**Features:**
- Google Drive upload with automatic organization
- Bandwidth throttling for background uploads
- Progress tracking with ETA calculations
- Folder structure preservation
- Resume capability for interrupted transfers

**Usage:**
```bash
./video-gdrive-move.sh [source_directory] [destination_folder]
```

---

#### 🔍 Quality Control & Analysis
**Files:** `video-corruption-scanner.sh`

Hardware-accelerated video quality verification.

**Features:**
- Multi-threaded corruption detection
- Hardware acceleration for faster processing
- Comprehensive error reporting
- Batch processing with detailed logs
- Multiple scanning algorithms

**Usage:**
```bash
./video-corruption-scanner.sh [directory] [--threads 8]
```

---

#### 🎵 Audio Processing
**Files:** `video-extract-audio.sh`

High-quality audio extraction supporting multiple formats.

**Features:**
- Support for MP3, AAC, WAV, FLAC output formats
- Quality preservation with format-specific optimization
- Batch processing with progress tracking
- Metadata preservation during extraction
- Cross-platform compatibility

**Usage:**
```bash
./video-extract-audio.sh [input_dir] [output_format]
```

---

#### 🛠️ Metadata & Repair Tools
**Files:** `gopro-metadata-fix.sh`, `metadata-mover.sh`

Specialized metadata management and repair utilities.

**Features:**
- GoPro-specific metadata handling
- Date-based file organization using ExifTool
- Comprehensive metadata extraction and reporting
- Batch processing with error recovery
- Filesystem-safe filename conversion

**Usage:**
```bash
./gopro-metadata-fix.sh [gopro_directory]
./metadata-mover.sh [source] [destination]
```

## ⚙️ Configuration

### Hardware Acceleration Setup

**macOS (VideoToolbox):**
```bash
# Automatically detected and enabled
# No additional configuration required
```

**Linux (NVENC/VAAPI):**
```bash
# Install GPU drivers and verify acceleration
ffmpeg -hwaccels  # List available hardware accelerators
```

### Cloud Integration Setup

**Google Drive (rclone):**
```bash
# Configure rclone for Google Drive
rclone config

# Test connection
rclone lsd remote:
```

### Performance Optimization

**CPU Core Detection:**
- Scripts automatically detect available cores
- Optimal thread allocation based on system capabilities
- Memory management for large video files

## 🔧 Performance Optimization

### Multi-core Processing

All scripts automatically optimize processing:

- **CPU-bound tasks:** Uses (cores - 1) to leave one core for system
- **I/O-bound tasks:** Uses (cores × 2) for maximum throughput
- **Memory management:** Processes videos in chunks to prevent memory issues
- **Hardware acceleration:** Automatic detection and utilization

### Memory Efficiency

- Large video files processed in streaming mode
- Automatic garbage collection during batch operations
- Progress tracking with minimal memory footprint
- Intelligent buffering for network operations

## 📊 Logging and Monitoring

### Desktop Logging

All scripts place log files on your Desktop for easy access:

- **Format:** `script-name.log`
- **Content:** Timestamped operations, errors, and results
- **Rotation:** New log file created for each run

### Progress Tracking

Visual progress indicators show:
- Current operation status and file being processed
- Completion percentage with visual progress bars
- Estimated time remaining and processing speed metrics
- Hardware acceleration status and GPU utilization

## 🐛 Troubleshooting

### Common Issues

**FFmpeg not found:**
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

**GUI dialogs not working:**
```bash
# Install zenity for Linux GUI support
sudo apt install zenity

# macOS uses native osascript (built-in)
```

**Hardware acceleration not working:**
```bash
# Check available accelerators
ffmpeg -hwaccels

# Verify GPU drivers are installed
# Scripts will fallback to software encoding
```

**Permission denied errors:**
```bash
# Make bash scripts executable
chmod +x *.sh

# Check file permissions
ls -la /path/to/file
```

**Large file processing errors:**
```bash
# Increase system limits for large files
ulimit -n 4096

# Monitor disk space during processing
df -h
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Python scripts
python3 script-name.py --debug

# Bash scripts
bash -x script-name.sh
```

### Log Analysis

Check desktop log files for detailed error information:

```bash
# View recent log entries
tail -f ~/Desktop/script-name.log

# Search for errors
grep -i error ~/Desktop/script-name.log

# Monitor real-time processing
watch -n 1 'tail -5 ~/Desktop/script-name.log'
```

## 📝 Dependencies

### System Requirements

**Essential (Auto-installed where possible):**
- `ffmpeg` - Video processing and encoding
- `ffprobe` - Video analysis and metadata extraction
- `zenity` - GUI dialogs (Linux) / `osascript` (macOS)

**Additional Tools:**
- `exiftool` - Advanced metadata manipulation
- `rclone` - Cloud storage integration
- `parallel` - GNU parallel for multi-core processing
- `bc` - Mathematical calculations

### Python Packages

**Standard Library Only** (No third-party packages required):
- `tkinter` - GUI dialogs and interfaces
- `subprocess` - System command execution
- `multiprocessing` - Parallel processing
- `concurrent.futures` - Thread/process pool management
- `pathlib` - Modern path handling
- `typing` - Type hints for better code quality

### Installation Commands

**macOS (Homebrew):**
```bash
# Install all dependencies
brew install ffmpeg zenity parallel bc exiftool rclone

# Verify installations
ffmpeg -version && rclone version
```

**Linux (Ubuntu/Debian):**
```bash
# Install all dependencies
sudo apt update
sudo apt install ffmpeg zenity parallel bc exiftool rclone

# Verify installations
ffmpeg -version && rclone version
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

- **Issues:** Report bugs via GitHub Issues
- **Email:** sanchez314c@speedheathens.com
- **Documentation:** Check individual script headers for detailed usage

## 🔗 Related Projects

- [Audio Processing Scripts](../Audio/) - Audio manipulation tools
- [Images Processing Scripts](../Images/) - Image processing and metadata tools
- [System Utilities](../System/) - System administration scripts
- [AI/ML Tools](../AI-ML/) - Machine learning and AI utilities

---

**Author:** sanchez314c@speedheathens.com  
**Version:** 1.0.0  
**Last Updated:** 2025-01-24  
**Platform:** macOS & Linux  

*GET SWIFTY! 🚀*