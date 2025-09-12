# Media Processing Tools

Video and audio processing utilities with hardware acceleration support.

## Directory Structure

### Video-Encoding/
Hardware-accelerated video encoding tools
- `media-hevc-gpu-encoder-v01.sh` - Enhanced HEVC encoder with VideoToolbox/Metal acceleration

### Audio-Tools/
Audio processing and conversion utilities

### Batch-Converters/
Batch processing tools for media conversion workflows

## Available Tools

### Video Encoding

**HEVC GPU Encoder (`media-hevc-gpu-encoder-v01.sh`)**
- **Platform:** macOS with VideoToolbox/Metal acceleration
- **Input formats:** MOV, MP4, AVI, MKV
- **Output:** HEVC/H.265 MP4 with high compression
- **Features:**
  - Hardware acceleration using VideoToolbox
  - Automatic quality scaling to 1080p
  - Comprehensive error handling and logging
  - Progress reporting and compression statistics
  - Skip existing files to resume interrupted batches
  - Detailed encoding summaries

**Usage:**
```bash
# Use default directories
./media-hevc-gpu-encoder-v01.sh

# Specify custom directories  
./media-hevc-gpu-encoder-v01.sh /path/to/input /path/to/output
```

**Technical Specifications:**
- Video codec: HEVC (hevc_videotoolbox)
- Bitrate: 8Mbps with 12Mbps max
- Profile: Main profile, Level 4.1
- Audio: Passthrough copy (no re-encoding)
- Container: MP4 with fast start enabled

## Hardware Requirements

### macOS HEVC Encoding
- **Required:** macOS with VideoToolbox support
- **GPU:** Metal-compatible GPU for hardware acceleration
- **Software:** FFmpeg with VideoToolbox support (`brew install ffmpeg`)
- **Monitor:** Use Activity Monitor > GPU History to verify hardware usage

## Performance Features
- **v01 Enhancements:**
  - Robust error handling and recovery
  - Flexible input directory detection
  - Support for multiple video formats
  - Comprehensive logging system
  - Resume capability (skips existing outputs)
  - Real-time progress monitoring
  - Detailed compression statistics

## Quality & Security
- Enhanced from original versions with security improvements
- Input validation and sanitization
- Safe file handling practices
- Comprehensive error logging
- No destructive operations on source files