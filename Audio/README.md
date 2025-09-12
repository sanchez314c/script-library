# Audio Processing Scripts Collection

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/sanchez314c/Script.Library)
[![Python](https://img.shields.io/badge/python-3.6+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://www.apple.com/macos/)

A comprehensive collection of professional-grade audio processing, management, and synthesis tools optimized for macOS. This suite provides everything needed for audio file manipulation, text-to-speech synthesis, metadata management, and more.

## 🎯 Overview

This collection includes five powerful audio processing applications:

- **🔄 Audio Batch Converter** - Multi-core M4A conversion with year-based organization
- **🔍 Audio File Deduplicator** - SHA-256 based duplicate detection and management
- **🎙️ Apple Podcasts Exporter** - Extract and organize downloaded podcast episodes
- **🎵 Spotify Lyrics Menubar** - Real-time lyrics display for Spotify tracks
- **🗣️ Text-to-Speech Universal** - Comprehensive TTS with system voices and ElevenLabs API

## ✨ Key Features

- **🍎 Native macOS Integration** - Uses native file dialogs, notifications, and menubar apps
- **⚡ Multi-core Processing** - Automatically optimizes CPU usage for maximum performance
- **🔐 Secure Credential Management** - GUI-based API key input with encrypted storage
- **📦 Auto-dependency Installation** - Scripts automatically install required packages
- **📊 Progress Tracking** - Visual progress indicators for all long-running operations
- **🎯 Standardized Interface** - Consistent CLI arguments and GUI elements across all scripts
- **🖥️ Desktop Logging** - All log files automatically placed on desktop for easy access

## 🚀 Quick Start

1. **Clone or download** this Audio folder to your local machine
2. **Run any script** - dependencies install automatically on first use
3. **Follow GUI prompts** for API credentials when needed
4. **Check your Desktop** for log files after script execution

### Example Usage

```bash
# Convert audio files to M4A format
python3 audio-batch-converter-to-m4a.py

# Find and manage duplicate audio files
python3 audio-file-deduplicator.py

# Export Apple Podcasts episodes
python3 audio-apple-podcasts-exporter.py

# Start Spotify lyrics menubar app
python3 audio-spotify-lyrics-menubar.py

# Launch comprehensive TTS application
python3 audio-tts-system-voices-gui.py
```

## 📋 Requirements

- **Python 3.6+** (macOS typically includes this)
- **macOS 10.14+** (for optimal native dialog support)
- **FFmpeg** (for audio conversion - installed via Homebrew)

### Install FFmpeg

```bash
# Install Homebrew if you haven't already
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install FFmpeg
brew install ffmpeg
```

## 📱 Applications

### 🔄 Audio Batch Converter to M4A
**File:** `audio-batch-converter-to-m4a.py`

Converts various audio formats to high-quality M4A with intelligent organization and metadata preservation.

**Features:**
- Multi-core processing for maximum speed
- Year-based folder organization (2020/, 2021/, etc.)
- Metadata preservation and enhancement
- Progress tracking with GUI
- Comprehensive error handling
- Quality settings optimization

**Usage:**
```bash
python3 audio-batch-converter-to-m4a.py [options]

Options:
  -s, --source DIR      Source directory (default: GUI prompt)
  -d, --destination DIR Destination directory (default: GUI prompt)
  -q, --quality LEVEL   Quality: high/medium/low (default: high)
  --auto-install        Install dependencies without prompting
```

**Supported Input Formats:** MP3, WAV, FLAC, OGG, AAC, WMA, AIFF

---

### 🔍 Audio File Deduplicator
**File:** `audio-file-deduplicator.py`

Identifies and manages duplicate audio files using SHA-256 hashing with intelligent collision handling.

**Features:**
- SHA-256 based duplicate detection
- Parallel processing for large collections
- Safe duplicate management options
- Collision detection and resolution
- Detailed reporting with statistics
- Preview before deletion

**Usage:**
```bash
python3 audio-file-deduplicator.py [options]

Options:
  -d, --directory DIR   Directory to scan (default: GUI prompt)
  -r, --recursive       Scan subdirectories recursively
  --dry-run            Show what would be done without making changes
  --auto-delete        Automatically delete duplicates (use with caution)
```

**Duplicate Resolution Strategy:**
1. Keeps file with longest filename (more descriptive)
2. Prefers files in root directory over subdirectories
3. Maintains file with earliest creation date
4. Provides manual selection for complex cases

---

### 🎙️ Apple Podcasts Exporter
**File:** `audio-apple-podcasts-exporter.py`

Exports downloaded Apple Podcasts episodes with full metadata preservation and organized directory structure.

**Features:**
- Direct Apple Podcasts database integration
- Complete metadata preservation (title, description, artwork)
- Organized folder structure by podcast show
- Filename sanitization for filesystem compatibility
- Episode filtering and selection
- Progress tracking for large exports

**Usage:**
```bash
python3 audio-apple-podcasts-exporter.py [options]

Options:
  -o, --output DIR      Output directory (default: GUI prompt)
  -s, --show NAME       Export specific podcast show only
  --format FORMAT       Output format: copy/mp3/m4a (default: copy)
  --include-artwork     Download and embed episode artwork
```

**Directory Structure:**
```
Export Directory/
├── Podcast Show 1/
│   ├── Episode 1.mp3
│   ├── Episode 2.mp3
│   └── artwork.jpg
└── Podcast Show 2/
    ├── Episode A.mp3
    └── Episode B.mp3
```

---

### 🎵 Spotify Lyrics Menubar
**File:** `audio-spotify-lyrics-menubar.py`

macOS menubar application that displays real-time lyrics for currently playing Spotify tracks.

**Features:**
- Real-time Spotify track monitoring
- Multi-source lyrics fetching (Genius, AZLyrics, Musixmatch)
- Native macOS menubar integration
- Automatic track change detection
- Lyrics caching for offline access
- Custom notification settings

**Setup:**
1. Create a Spotify app at [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Add `http://localhost:8888/callback` as a redirect URI
3. Run the script and enter your Client ID and Client Secret when prompted

**Usage:**
```bash
python3 audio-spotify-lyrics-menubar.py [options]

Options:
  --reset-credentials   Clear stored Spotify credentials
  --no-notifications    Disable track change notifications
  --debug              Enable debug logging
```

**Menubar Features:**
- Click icon to view current lyrics
- Right-click for preferences and controls
- Automatic startup on login (optional)
- Keyboard shortcuts for quick access

---

### 🗣️ Text-to-Speech Universal Application
**File:** `audio-tts-system-voices-gui.py`

Comprehensive text-to-speech application supporting both system voices and cloud-based synthesis.

**Features:**
- **System Voices** - Built-in macOS voice synthesis via pyttsx3
- **ElevenLabs API** - Professional-quality cloud synthesis
- **GUI & CLI Modes** - Flexible interface options
- **Batch Processing** - Handle large documents intelligently
- **Smart Text Chunking** - Optimal processing for long texts
- **Voice Preview** - Test voices before full conversion
- **Audio File Export** - Save speech as MP3/WAV files

**GUI Mode:**
```bash
python3 audio-tts-system-voices-gui.py
```

**CLI Mode:**
```bash
python3 audio-tts-system-voices-gui.py --cli [options]

Options:
  -f, --file FILE       Text file to process
  -o, --output FILE     Output audio file
  --provider PROVIDER   TTS provider: system/elevenlabs
  --voice VOICE         Voice name or ID
```

**ElevenLabs Setup:**
1. Sign up at [ElevenLabs](https://elevenlabs.io)
2. Get your API key from your profile
3. Run the script and enter your API key when prompted

**Supported Features by Provider:**

| Feature | System Voices | ElevenLabs |
|---------|---------------|------------|
| Voice Selection | ✅ | ✅ |
| Rate Control | ✅ | ❌ |
| Volume Control | ✅ | ❌ |
| Quality Settings | ❌ | ✅ |
| Custom Voices | ❌ | ✅ |
| Batch Processing | ✅ | ✅ |
| Large Documents | ❌ | ✅ |

## 🛠️ Utilities

### 📚 Audio Utils Library
**File:** `audio_utils.py`

Comprehensive utilities library providing shared functionality across all scripts.

**Core Functions:**
- `check_and_install_dependencies()` - Automatic package installation
- `setup_logging()` - Desktop logging configuration
- `select_file/folder()` - Native macOS file dialogs
- `CredentialManager` - Secure API key storage
- `ProgressTracker` - GUI progress indicators
- `ParallelExecutor` - Optimized multi-core processing

**Audio-Specific Functions:**
- `find_audio_files()` - Discover audio files in directories
- `calculate_file_hash()` - SHA-256 hash calculation
- `get_audio_duration()` - Extract audio file duration
- `sanitize_filename()` - Filesystem-safe filename conversion
- `SpotifyHelper` - Spotify API integration utilities
- `ApplePodcastsHelper` - Apple Podcasts database access
- `ElevenLabsHelper` - ElevenLabs API utilities
- `AudioProcessor` - FFmpeg-based audio processing

**Import in your scripts:**
```python
from audio_utils import *
```

## ⚙️ Configuration

### Credential Storage

All API credentials are stored securely in JSON files in your home directory:

- **Spotify:** `~/.spotify_config.json`
- **ElevenLabs:** `~/.elevenlabs_config.json`

### Configuration Files

Scripts automatically create and manage configuration files with default settings:

```json
{
  "api_key": "your_api_key_here",
  "preferred_voice": "voice_id",
  "quality_settings": {
    "stability": 0.5,
    "similarity_boost": 0.75
  }
}
```

### Reset Credentials

To reset stored credentials for any script:

```bash
python3 script-name.py --reset-credentials
```

## 🔧 Performance Optimization

### Multi-core Processing

All scripts automatically optimize CPU usage:

- **CPU-bound tasks:** Uses (cores - 1) to leave one core for system
- **I/O-bound tasks:** Uses (cores × 2) for maximum throughput
- **Memory management:** Processes files in batches to prevent memory issues

### Memory Efficiency

- Large files processed in chunks
- Automatic garbage collection
- Progress tracking with minimal memory footprint
- Streaming processing for audio conversion

## 📊 Logging and Monitoring

### Desktop Logging

All scripts place log files on your Desktop for easy access:

- **Format:** `script-name.log`
- **Content:** Timestamped operations, errors, and results
- **Rotation:** New log file created for each run

### Progress Tracking

Visual progress indicators show:
- Current operation status
- Completion percentage
- Estimated time remaining
- Processing speed metrics

## 🐛 Troubleshooting

### Common Issues

**FFmpeg not found:**
```bash
brew install ffmpeg
```

**Permission denied errors:**
```bash
# Check file permissions
ls -la /path/to/file

# Fix permissions if needed
chmod 755 /path/to/file
```

**Python package errors:**
```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Clear pip cache
python3 -m pip cache purge

# Reinstall problematic package
python3 -m pip install --force-reinstall package_name
```

**Spotify API issues:**
1. Verify your app settings in Spotify Developer Dashboard
2. Ensure redirect URI is exactly: `http://localhost:8888/callback`
3. Check that your app is not in "Development Mode" restrictions

**ElevenLabs API issues:**
1. Verify your API key is correct
2. Check your account's usage limits
3. Ensure you have sufficient credits

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python3 script-name.py --debug
```

### Log Analysis

Check desktop log files for detailed error information:

```bash
# View recent log entries
tail -f ~/Desktop/script-name.log

# Search for errors
grep -i error ~/Desktop/script-name.log
```

## 📝 Dependencies

### Automatically Installed

These packages are installed automatically when needed:

- `requests` - HTTP requests for API calls
- `pydub` - Audio file manipulation
- `pyttsx3` - System text-to-speech
- `spotipy` - Spotify API integration
- `beautifulsoup4` - Web scraping for lyrics
- `lxml` - XML parsing for metadata

### System Requirements

- **FFmpeg** - Audio/video processing (install via Homebrew)
- **SQLite3** - Database access (included with macOS)
- **tkinter** - GUI dialogs (included with Python)

### Optional Dependencies

- **mutagen** - Advanced audio metadata editing
- **pyobjc** - Enhanced macOS integration
- **playsound** - Cross-platform audio playback

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

- [Video Processing Scripts](../Video/) - Video manipulation tools
- [System Utilities](../System/) - System administration scripts
- [AI/ML Tools](../AI-ML/) - Machine learning and AI utilities

---

**Author:** sanchez314c@speedheathens.com  
**Version:** 1.0.0  
**Last Updated:** 2025-01-23  
**Platform:** macOS 10.14+  

*GET SWIFTY! 🚀*