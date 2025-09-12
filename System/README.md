# System - Universal macOS System Administration & File Management Suite
## Professional-Grade System Tools for macOS Power Users

### 🎯 Overview
Comprehensive collection of enterprise-level system administration and file management tools designed exclusively for macOS. All scripts feature universal compatibility, auto-dependency installation, and native macOS integration for maximum performance and reliability.

### ✨ Key Features
- **Universal macOS Compatibility** - Native Path.home() and desktop logging
- **Auto-Dependency Installation** - Automatic setup of Python packages and system tools
- **Multi-threaded Performance** - CPU-optimized parallel processing
- **Native GUI Integration** - macOS-style dialogs and progress tracking
- **Enterprise-Grade Security** - VPN testing, network monitoring, privacy tools
- **Professional File Management** - Advanced sorting, moving, and organization
- **Real-time Monitoring** - System inspection and folder watching capabilities

---

## 🚀 Core Categories

### 📁 **File Management & Organization**
*Advanced file processing and directory management tools*

| Script | Description | Key Features |
|--------|-------------|--------------|
| `systems-extension-scanner.py` | Parallel file extension analyzer | Multi-threaded scanning, CSV/JSON export, visualization |
| `systems-flatten-directory.py` | Directory structure flattening utility | Smart collision resolution, dry-run mode, progress tracking |
| `systems-flatten-directory-by-type.py` | File organization by extension type | Intelligent categorization, bulk processing |
| `systems-flatten-directory-first4-subs.py` | Organization by filename prefix | Advanced pattern matching, collision handling |
| `systems-batch-folder-move.py` | High-performance directory transfer | 25,000+ folder optimization, intelligent job distribution |

### 🔄 **File Movement & Transfer**
*High-performance file and directory movement systems*

| Script | Description | Capabilities |
|--------|-------------|--------------|
| `systems-turbo-move.py` | High-performance file movement | Speed-optimized transfers, error recovery |
| `systems-turbo-file-mover.py` | Advanced file management utility | Directory processing, bulk operations |
| `systems-recursive-file-mover.py` | Recursive file movement system | Multiprocessing, deep directory traversal |
| `systems-file-move-manager.py` | Advanced movement management | Intelligent routing, collision detection |
| `systems-large-folder-move.py` | Massive directory transfer system | Enterprise-scale operations, progress monitoring |
| `systems-rsync-gui.py` | Rsync with optimized GUI | Sync operations, bandwidth control, scheduling |

### 🎯 **File Discovery & Selection**
*Intelligent file finding and sampling tools*

| Script | Description | Selection Methods |
|--------|-------------|------------------|
| `systems-grab-large-files.py` | Large file detection system | Size-based filtering, threshold configuration |
| `systems-grab-timestamp-files.py` | Timestamp-based file processing | Date range selection, metadata analysis |
| `systems-grab-random-files.py` | Random file selection system | Statistical sampling, size distribution |
| `systems-random-file-selector.py` | Advanced random sampling | Weighted selection, category filtering |

### 📊 **Media & Content Processing**
*Specialized tools for media file management*

| Script | Description | Processing Type |
|--------|-------------|-----------------|
| `systems-sort-videos-by-dimensions.py` | Video sorting by resolution | Dimension-based organization, quality categorization |
| `systems-trash-recovery.py` | Trash recovery system | File restoration, recovery validation |

### 🔍 **System Monitoring & Analysis**
*Real-time system monitoring and inspection tools*

| Script | Description | Monitoring Scope |
|--------|-------------|------------------|
| `systems-folder-monitor.py` | Real-time directory monitoring | File system events, change detection |

### 🌐 **Network & Privacy Tools**
*Advanced networking and privacy verification systems*

| Script | Description | Security Features |
|--------|-------------|-------------------|
| `systems-dns-leak-test.py` | DNS privacy testing tool | Leak detection, geolocation analysis, VPN verification |
| `systems-nordvpn-fetch.py` | NordVPN server information | API integration, server optimization |
| `systems-nordvpn-servers.py` | NordVPN server listing tool | Advanced filtering, connection optimization |
| `systems-find-network-rokus.py` | Network Roku discovery | Device detection, network scanning |

---

## 🔧 System Requirements

### Automatic Installation
All dependencies are automatically installed when running scripts:

- **Python 3.8+** (auto-verified and installed)
- **rsync** (for sync operations)
- **Network tools** (for connectivity testing)
- **GUI frameworks** (tkinter, native dialogs)

### Performance Optimization
- **Multi-core processing** using all available CPU cores
- **Memory-efficient** algorithms for large dataset handling
- **Native macOS integration** for optimal performance
- **Intelligent caching** for repeated operations

---

## 🚀 Quick Start Guide

### Basic File Management
```bash
# Analyze file extensions in a directory
python systems-extension-scanner.py

# Flatten complex directory structures
python systems-flatten-directory.py

# Move large collections of folders
python systems-batch-folder-move.py
```

### Advanced Operations
```bash
# High-performance file transfers
python systems-turbo-move.py

# Sync directories with GUI
python systems-rsync-gui.py

# Monitor directory changes in real-time
python systems-folder-monitor.py
```

### Network & Privacy Testing
```bash
# Test for DNS leaks
python systems-dns-leak-test.py

# Find optimal NordVPN servers
python systems-nordvpn-fetch.py

# Discover network devices
python systems-find-network-rokus.py
```

---

## 📋 Common Workflows

### 1. **Large-Scale File Organization**
```bash
python systems-extension-scanner.py        # Analyze file types
python systems-flatten-directory.py        # Flatten structure
python systems-batch-folder-move.py        # Organize by category
```

### 2. **System Cleanup & Optimization**
```bash
python systems-grab-large-files.py         # Find space hogs
python systems-sort-videos-by-dimensions.py # Organize media
python systems-trash-recovery.py           # Recover important files
```

### 3. **Network Security Verification**
```bash
python systems-dns-leak-test.py            # Test DNS privacy
python systems-nordvpn-servers.py          # Find optimal servers
python systems-find-network-rokus.py       # Inventory devices
```

### 4. **Advanced File Management**
```bash
python systems-folder-monitor.py           # Monitor changes
python systems-recursive-file-mover.py     # Deep reorganization
python systems-rsync-gui.py               # Sync with remote systems
```

---

## 🎯 Advanced Features

### Multi-Threading Performance
- **Intelligent core utilization** based on system capabilities
- **Load balancing** for optimal resource distribution
- **Memory management** for large-scale operations
- **Progress tracking** with ETA calculations

### Native macOS Integration
- **Path.home()** compatibility for universal user directory access
- **Desktop logging** with native notification support
- **Native dialogs** using tkinter and system APIs
- **Spotlight integration** for enhanced file discovery

### Error Handling & Recovery
- **Graceful degradation** when dependencies are missing
- **Automatic retry mechanisms** for network operations
- **Transaction-safe** file operations with rollback capability
- **Comprehensive logging** with desktop output support

---

## 🛡️ Security & Privacy

### Privacy Protection
- **DNS leak detection** with real-time monitoring
- **VPN verification** tools for privacy validation
- **Network scanning** with security awareness
- **Local processing** to avoid data transmission

### Data Safety
- **Non-destructive operations** with backup options
- **Dry-run modes** for safe testing
- **Collision detection** and resolution strategies
- **Recovery mechanisms** for operation failures

---

## 📈 Performance Benchmarks

### File Operations
- **Large directory processing**: Optimized for 25,000+ items
- **Multi-threaded transfers**: Up to 80% performance improvement
- **Memory efficiency**: Handles datasets larger than available RAM
- **Progress tracking**: Real-time ETA with accuracy

### Network Operations
- **DNS testing**: Sub-second leak detection
- **Server discovery**: Comprehensive network scanning
- **Connection optimization**: Intelligent server selection
- **Bandwidth awareness**: Adaptive performance tuning

---

## 🛠️ Troubleshooting

### Common Issues
1. **Permission errors**: Scripts auto-request necessary permissions
2. **Large file handling**: Memory-efficient streaming algorithms
3. **Network connectivity**: Automatic fallback and retry mechanisms
4. **GUI display**: Native macOS dialog compatibility

### Performance Optimization
- **SSD optimization**: Enhanced algorithms for solid-state drives
- **Network storage**: Intelligent caching for remote operations
- **Multi-monitor**: Full support for extended desktop setups
- **Background processing**: Non-blocking operations where possible

---

## 📊 Shell Script Utilities

### Included Shell Scripts
*These scripts complement the Python tools and provide system-level operations:*

- `backup-docker-containers.sh` - Docker container backup automation
- `force-trash-empty.sh` - Force empty macOS trash safely
- `grab-random-100gb.sh` - Random 100GB file selection
- `grab-underscore-files.sh` - Underscore filename processing
- `network-bond.sh` / `network-unbond.sh` - Network interface bonding
- `nordvpn-speedtest.sh` - NordVPN speed testing automation
- `rename-by-oldest-date.sh` - Date-based file renaming
- `restore-docker-containers.sh` - Docker container restoration
- `system-inspection.sh` - Comprehensive system analysis
- `tag-files-red.sh` - macOS file tagging automation
- `unpack-appimage.sh` - AppImage extraction utility

---

## 📈 Version Information
- **Version**: 1.0.0
- **Compatibility**: macOS Universal (Intel & Apple Silicon)
- **Python**: 3.8+ (auto-installed)
- **Last Updated**: 2025-01-24
- **License**: Professional Use

---

*Part of the Script.Library collection - Enterprise-grade tools for system administration and file management*