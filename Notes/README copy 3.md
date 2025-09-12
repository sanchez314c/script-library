# System Installers

Core system installation and configuration scripts for Ubuntu/Linux systems.

## Directory Structure

### Core-v01/ (Latest - April 2025)
- `system-ubuntu-essentials-v01.sh` - Complete Ubuntu setup with modern tools
- `system-nvidia-470-driver-v01.sh` - NVIDIA 470 driver installer  
- `system-master-installer-v01.sh` - Interactive master installer with enhanced security

### Core-v00/ (Original - March 2025)
- `system-ubuntu-essentials-v00.sh` - Original Ubuntu essentials installer
- Archive versions for reference

### Deprecated/
- Old backups and broken installation scripts
- Historical versions no longer recommended

## Usage

**⚠️ Important Security Notes:**
- All scripts have been audited and enhanced with security improvements
- Use Core-v01 versions for new installations
- Scripts include logging, error handling, and validation
- Never run as root directly - scripts will use sudo when needed

## Features Added in v01
- Enhanced error handling and logging
- Flexible directory detection
- Comprehensive prerequisite checking  
- Progress reporting and summaries
- Security validation of scripts before execution
- Timeout protection against hanging installations

## Installation Order (Recommended)
1. `system-ubuntu-essentials-v01.sh` - Base system setup
2. `system-nvidia-470-driver-v01.sh` - GPU drivers (if needed)
3. Use `system-master-installer-v01.sh` for guided installation of additional components