# 📱 Mobile Media Transfer Station

## 🎯 Overview

Professional-grade mobile device media synchronization suite designed for seamless transfer and organization of photos, videos, and audio files from Android and iOS devices. This enhanced solution features intelligent metadata preservation, automated organization, and comprehensive file management with universal macOS compatibility.

## ✨ Key Features

- **📱 Universal Device Support**: Automatic detection and handling of Android and iOS devices
- **🚀 Multi-threaded Transfers**: High-performance parallel file operations with progress tracking
- **🧠 Intelligent Organization**: Metadata-based sorting with EXIF data analysis and GPS extraction
- **🔍 Advanced Deduplication**: Sophisticated duplicate detection using file hashes and metadata
- **🎨 Professional GUI Interface**: Modern macOS-native interface with real-time monitoring
- **📊 Comprehensive Analytics**: Detailed transfer statistics and quality assessment
- **🔧 Auto-Dependency Management**: Automatic installation and version management
- **⚡ High-Performance Processing**: Optimized for large media collections

## 📋 Script Inventory

### 📱 Mobile Media Transfer Station
**File**: `mobiles-media-transfer.py`
- **Universal Device Support**: Android (ADB) and iOS (libimobiledevice) connectivity
- **Intelligent File Discovery**: Recursive scanning across all device storage locations
- **Metadata Preservation**: EXIF data extraction with GPS coordinates and camera information
- **Advanced Organization**: Automatic sorting by date, type, and quality metrics
- **Professional Interface**: Real-time progress tracking with device status monitoring
- **Batch Processing**: Multi-device simultaneous transfers with conflict resolution

## 🚀 Quick Start Guide

### Prerequisites
- **Operating System**: macOS (Universal compatibility)
- **Python Version**: 3.8 or higher
- **Android Devices**: ADB (Android Debug Bridge) installed
- **iOS Devices**: libimobiledevice framework installed
- **Dependencies**: Auto-installed on first run

### System Setup

#### Android Device Support
```bash
# Install ADB (Android Debug Bridge)
brew install android-platform-tools

# Enable USB Debugging on Android device:
# Settings > Developer Options > USB Debugging
```

#### iOS Device Support
```bash
# Install libimobiledevice
brew install libimobiledevice

# Trust computer on iOS device when prompted
```

### Basic Usage

1. **Make script executable**:
   ```bash
   chmod +x mobiles-media-transfer.py
   ```

2. **Launch GUI interface**:
   ```bash
   ./mobiles-media-transfer.py
   ```

3. **Command-line usage**:
   ```bash
   ./mobiles-media-transfer.py --folder /path/to/destination --no-gui
   ```

### GUI Operation Flow

1. **📁 Select Destination**: Choose folder for organized media files
2. **📱 Connect Devices**: USB or Wi-Fi connected devices appear automatically
3. **🚀 Start Transfer**: Begin automatic media discovery and transfer
4. **📊 Monitor Progress**: Real-time progress with detailed statistics
5. **📁 Auto-Organization**: Files sorted by date, type, and metadata

### Command-Line Interface

```bash
# Basic transfer with GUI
./mobiles-media-transfer.py --folder ~/Media

# Command-line mode without GUI
./mobiles-media-transfer.py --folder ~/Media --no-gui

# Organize existing files only
./mobiles-media-transfer.py --folder ~/Media --organize-only

# Monitor specific device type
./mobiles-media-transfer.py --folder ~/Media --device-type android

# Show help and options
./mobiles-media-transfer.py --help
```

## 📊 Advanced Features

### 🤖 Android Device Handling
- **Comprehensive Discovery**: Scans DCIM, Pictures, Movies, Music, and Download folders
- **Metadata Extraction**: File size, creation date, and storage location analysis
- **Efficient Transfer**: Multi-threaded ADB pull operations with retry mechanisms
- **Storage Optimization**: Automatic detection of available device storage paths
- **Conflict Resolution**: Intelligent handling of duplicate filenames and paths

### 📱 iOS Device Integration
- **Native Framework Support**: Uses libimobiledevice for seamless iOS connectivity
- **Photo Library Access**: Complete photo and video library synchronization
- **Live Photos Support**: Preserves Live Photo relationships and metadata
- **Trust Management**: Automatic handling of device trust relationships
- **Privacy Compliance**: Respects iOS privacy and security requirements

### 🧠 Intelligent Organization System

#### Metadata Analysis
- **EXIF Data Extraction**: Camera make/model, settings, and technical details
- **GPS Coordinate Processing**: Location data with decimal degree conversion
- **Quality Assessment**: Resolution-based quality scoring (25-100 scale)
- **Date/Time Preservation**: Original creation timestamps maintained
- **File Integrity**: MD5 hash verification for duplicate detection

#### Folder Structure
```
Destination/
├── photos/
│   ├── 2024/
│   │   ├── 01-January/
│   │   ├── 02-February/
│   │   └── ...
│   └── 2025/
├── videos/
│   ├── 2024/
│   └── 2025/
└── audio/
    ├── 2024/
    └── 2025/
```

#### File Naming Convention
```
Format: YYYYMMDD_HHMMSS_CameraMake_Model.ext
Example: 20240315_143022_Apple_iPhone15Pro.jpg
```

### 📈 Performance Optimization

#### Multi-threading Configuration
- **Android Transfers**: Up to 8 concurrent threads per device
- **iOS Transfers**: Single thread for stability (iOS limitation)
- **Organization**: CPU core count based parallel processing
- **Memory Management**: Automatic memory monitoring and optimization

#### Large File Handling
- **Size Limits**: Configurable maximum file size handling
- **Timeout Management**: Progressive timeout increases for large files
- **Error Recovery**: Automatic retry with exponential backoff
- **Progress Tracking**: Real-time transfer speed and ETA calculation

## 🔧 Configuration Options

### GUI Configuration
- **Device Monitoring**: Automatic 3-second device scan intervals
- **Transfer Settings**: Batch size and threading configuration
- **Organization Preferences**: Folder structure and naming conventions
- **Quality Filters**: Minimum resolution and file size thresholds

### Command-Line Options
```bash
Transfer Options:
  --folder, -f           Destination folder for media files
  --no-gui              Run in command-line mode
  --organize-only       Only organize existing files
  --device-type         Monitor specific device type (android/ios/both)

Advanced Options:
  --max-workers         Number of transfer threads (default: auto)
  --timeout             Transfer timeout in seconds (default: 300)
  --min-quality         Minimum quality score (0-100, default: 25)
  --duplicate-action    Action for duplicates (skip/rename/replace)
```

### System Requirements Check
```bash
# Check system compatibility
./mobiles-media-transfer.py --check-system

# Install missing dependencies
./mobiles-media-transfer.py --install-deps
```

## 📊 Transfer Statistics

### Real-time Metrics
- **Transfer Speed**: MB/s with moving average calculation
- **File Counts**: Successful/failed transfers with error categorization
- **Duplicate Detection**: Similar file identification and space savings
- **Quality Analysis**: Average quality score and resolution distribution
- **Time Estimates**: ETA calculation based on current transfer rates

### Quality Assessment Scale
- **100 Points**: 20MP+ high-resolution images (professional quality)
- **85 Points**: 12MP+ images (excellent quality)
- **70 Points**: 8MP+ images (very good quality)
- **55 Points**: 5MP+ images (good quality)
- **40 Points**: 2MP+ images (acceptable quality)
- **25 Points**: Below 2MP (basic quality)

## 🛠️ Troubleshooting

### Device Connection Issues

**Android Device Not Detected**
```bash
# Check ADB installation
adb version

# Verify device authorization
adb devices

# Enable USB debugging on device
# Settings > Developer Options > USB Debugging

# Reset ADB server
adb kill-server && adb start-server
```

**iOS Device Not Detected**
```bash
# Check libimobiledevice installation
idevice_id -l

# Verify device trust
ideviceinfo -u [DEVICE_ID]

# Reset device trust
# Settings > General > Reset > Reset Location & Privacy
```

### Transfer Performance Issues

**Slow Transfer Speeds**
- Use USB 3.0 or higher connections
- Close unnecessary applications
- Ensure sufficient disk space (10GB+ recommended)
- Check for background sync operations

**High Memory Usage**
- Reduce number of concurrent transfers
- Process smaller batches of files
- Restart application for large transfers
- Monitor system resources during operation

### Common Error Solutions

**Permission Denied Errors**
```bash
# Make script executable
chmod +x mobiles-media-transfer.py

# Check destination folder permissions
ls -la /path/to/destination
```

**Dependency Installation Failures**
```bash
# Manual dependency installation
pip3 install pillow exifread tqdm psutil

# Update pip if needed
pip3 install --upgrade pip
```

## 📁 Output Structure

### Transfer Reports
```
Desktop/Mobile_Transfer_Logs_[timestamp]/
├── mobile_transfer_[timestamp].log    # Detailed transfer log
├── device_capabilities.json           # Device information
├── transfer_statistics.json           # Performance metrics
└── error_report.json                 # Error analysis
```

### Organized Media
```
[destination]/
├── photos/                           # Photo files organized by date
├── videos/                          # Video files organized by date
├── audio/                          # Audio files organized by date
├── transfer_reports/               # Transfer session reports
└── metadata/                      # Extracted metadata files
```

## 📚 Technical Specifications

### Supported File Formats

#### Photos
- **Standard**: JPG, JPEG, PNG, GIF, BMP, TIFF
- **Mobile**: HEIC, HEIF (iOS native formats)
- **RAW**: CR2, NEF, ARW, DNG, ORF, RW2, PEF, SRW

#### Videos
- **Standard**: MP4, MOV, AVI, MKV, WMV, FLV, WEBM
- **Mobile**: 3GP, 3G2, M4V (mobile optimized formats)
- **Professional**: M2TS, MTS, TS, VOB (high-definition formats)

#### Audio
- **Compressed**: MP3, AAC, OGG, WMA, OPUS, AMR
- **Lossless**: FLAC, WAV, APE
- **Mobile**: M4A, M4P, M4B (iTunes formats)

### System Requirements
- **OS**: macOS 10.14 or later
- **RAM**: 4GB minimum, 8GB recommended for large transfers
- **Storage**: 10GB+ free space for processing buffers
- **CPU**: Multi-core processor recommended for parallel processing
- **USB**: USB 2.0 minimum, USB 3.0+ recommended for speed

### Dependencies
```python
# Core Dependencies (auto-installed)
pillow>=9.0.0              # Image processing and EXIF handling
exifread>=3.0.0             # EXIF metadata extraction
tqdm>=4.64.0                # Progress bars and status tracking
psutil>=5.9.0               # System resource monitoring

# System Dependencies (manual installation required)
android-platform-tools      # ADB for Android device connectivity
libimobiledevice            # iOS device framework and tools

# Built-in Dependencies
tkinter                     # GUI framework (included with Python)
threading                  # Multi-threaded processing
subprocess                 # External command execution
pathlib                    # Modern path handling
logging                    # Comprehensive logging system
```

## 🎯 GET SWIFTY Methodology

This Mobile Media Transfer Station implements the GET SWIFTY development approach:

- **🎨 GET**: Professional GUI with real-time device monitoring
- **⚡ SWIFTY**: High-performance multi-threaded transfers
- **📱 Mobile-First**: Optimized for mobile device workflows
- **🔧 Reliable**: Comprehensive error handling and recovery
- **📊 Analytics**: Detailed statistics and performance metrics
- **🍎 macOS Native**: Universal compatibility with system integration

## 📞 Support and Resources

### Device Setup Guides
- **Android**: Enable Developer Options and USB Debugging
- **iOS**: Install and trust device certificates
- **macOS**: Install ADB and libimobiledevice frameworks

### Performance Tuning
- **Hardware**: Use USB 3.0+ for optimal transfer speeds
- **Software**: Close resource-intensive applications during transfers
- **Network**: Disable cloud sync during large transfers
- **Storage**: Use SSD storage for best performance

### Advanced Usage
- **Batch Processing**: Transfer from multiple devices simultaneously
- **Automation**: Command-line interface for scripted operations
- **Quality Control**: Filter transfers by resolution and file size
- **Metadata Preservation**: Maintain all original file attributes

---

**GET SWIFTY Mobile Media Transfer Station v1.0.0**  
*Professional mobile device media synchronization for macOS*