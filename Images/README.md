# Images - Media Processing & Metadata Management Suite
## Universal macOS-Compatible Image Processing Tools

### 🎯 Overview
Comprehensive collection of professional-grade image processing and metadata management tools designed for macOS. All scripts feature universal compatibility, auto-dependency installation, and native macOS GUI integration.

### ✨ Key Features
- **Universal macOS Compatibility** - Native Path.home() and desktop logging
- **Auto-Dependency Installation** - Automatic setup of Python packages and system tools
- **Parallel Processing** - Multi-core optimization for maximum performance
- **Native GUI Integration** - macOS-style file/directory selection dialogs
- **Comprehensive Metadata Support** - Full EXIF, IPTC, and XMP metadata handling
- **Professional Logging** - Desktop output with detailed operation tracking

---

## 📁 Core Scripts

### 🎯 Advanced Processing Tools
| Script | Description | Features |
|--------|-------------|----------|
| `images-media-image-processor.py` | Advanced image processing with parallel cores | Comprehensive format support, metadata extraction, year-based organization |
| `images-media-metadata-reporter.py` | Comprehensive metadata scanning and reporting | Statistical analysis, CSV export, pattern filtering |
| `images-filename-to-dto-fmd.py` | Filename to DateTime Original converter | Multi-threaded processing, pattern recognition |
| `images-manual-meta-fix.py` | Manual metadata fixing with backup | Multi-format support, progress tracking |

### 🔧 Metadata Management
| Script | Description | Capabilities |
|--------|-------------|--------------|
| `images-fix-all-metadata-tags.py` | Comprehensive metadata repair | Two-phase processing, batch operations |
| `images-extract-date-from-xmp.py` | XMP metadata processor | Dual directory selection, history parsing |
| `images-extract-date-from-xmp-inplace.py` | In-place XMP renaming | Direct file modification, pattern matching |
| `images-filename-to-date-metadata.py` | Filename to metadata converter | EXIF cleansing, batch processing |

### 📊 Analysis & Reporting
| Script | Description | Output Format |
|--------|-------------|---------------|
| `images-display-all-meta-tags.py` | Metadata tag discovery | Parallel processing, comprehensive listing |
| `images-list-date-fields.py` | DateTime field discovery | Desktop output, JSON extraction |
| `images-list-datetime-metadata.py` | Metadata datetime scanner | Field enumeration, pattern analysis |
| `images-report-unique-date-tags.py` | Unique datetime tag reporter | Regex matching, statistical summary |

### 🎨 Image Operations
| Script | Description | Processing Type |
|--------|-------------|-----------------|
| `images-convert-images-to-pdf.py` | Image to PDF converter | Layout control, quality optimization |
| `images-fix-rotation-v1.py` | PIL-based rotation correction | Single image processing |
| `images-fix-rotation-v2.py` | Advanced rotation processor | Batch operations, metadata preservation |
| `images-fix-rotation-batch.py` | External script batch processor | Integration with shell scripts |

### 📝 File Management
| Script | Description | Organization Method |
|--------|-------------|-------------------|
| `images-rename-by-creation-date.py` | ModifyDate-based renaming | Metadata-driven naming |
| `images-rename-from-modify-date.py` | FileModifyDate converter | EXIF cleansing, integrity preservation |
| `images-rename-from-original-date.py` | DateTimeOriginal converter | Original timestamp preservation |
| `images-update-all-dates-from-filename.py` | Batch date processor | Filename-to-metadata sync |

---

## 🔧 System Requirements

### Automatic Installation
All dependencies are automatically installed when running scripts:

- **Python 3.8+** (auto-verified and installed)
- **exiftool** (installed via Homebrew)
- **PIL/Pillow** (auto-installed via pip)
- **Additional packages** as needed per script

### Supported Formats
- **Image Formats**: JPG, JPEG, PNG, GIF, BMP, TIFF, HEIC, HEIF, DNG, WebP
- **Metadata Standards**: EXIF, IPTC, XMP
- **Output Formats**: PDF, CSV, JSON, Text reports

---

## 🚀 Quick Start

### Basic Usage
```bash
# Run any script directly - dependencies auto-install
python images-media-image-processor.py

# All scripts use native macOS dialogs
python images-metadata-reporter.py
```

### Batch Processing
```bash
# Process multiple directories
python images-fix-all-metadata-tags.py

# Generate comprehensive reports
python images-report-unique-date-tags.py
```

---

## 📋 Common Workflows

### 1. **New Photo Import**
```bash
python images-media-image-processor.py      # Organize by date
python images-fix-all-metadata-tags.py      # Clean metadata
python images-metadata-reporter.py          # Generate report
```

### 2. **Metadata Cleanup**
```bash
python images-filename-to-date-metadata.py  # Extract dates from filenames
python images-fix-rotation-v2.py            # Fix orientation
python images-manual-meta-fix.py            # Manual corrections
```

### 3. **Archive Analysis**
```bash
python images-list-datetime-metadata.py     # Discover date fields
python images-report-unique-date-tags.py    # Analyze patterns
python images-convert-images-to-pdf.py      # Create archives
```

---

## 🔍 Advanced Features

### Parallel Processing
- **Multi-core optimization** using all available CPU cores
- **Thread-safe operations** for batch processing
- **Memory management** for large dataset handling

### GUI Integration
- **Native macOS dialogs** for file/directory selection
- **Progress tracking** with visual feedback
- **Error reporting** with user-friendly messages

### Metadata Handling
- **Comprehensive format support** (EXIF, IPTC, XMP)
- **Non-destructive operations** with backup options
- **Pattern recognition** for automated processing

---

## 📊 Performance Notes

### Optimization Features
- **Parallel processing** utilizing all CPU cores
- **Memory-efficient** streaming for large files
- **Batch operations** for improved throughput
- **Progress tracking** for long-running operations

### Recommended Usage
- **Single operations**: Use specific scripts for targeted tasks
- **Batch processing**: Leverage parallel processing scripts
- **Large datasets**: Monitor memory usage during processing
- **Network storage**: Consider local processing for better performance

---

## 🛠️ Troubleshooting

### Common Issues
1. **Permission errors**: Ensure read/write access to target directories
2. **Missing exiftool**: Script will auto-install via Homebrew
3. **Large file processing**: Monitor system memory during batch operations
4. **Unsupported formats**: Check format support list above

### Debug Mode
Most scripts include verbose logging options for troubleshooting complex operations.

---

## 📈 Version Information
- **Version**: 1.0.0
- **Compatibility**: macOS Universal
- **Python**: 3.8+ (auto-installed)
- **Last Updated**: 2025-01-24

---

*Part of the Script.Library collection - Professional tools for media management and processing*