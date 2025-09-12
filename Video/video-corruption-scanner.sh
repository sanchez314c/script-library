#!/bin/bash
#
# video-corruption-scanner - Video File Corruption Detection and Isolation
# ----------------------------------------------------
# Author: sanchez314c@speedheathens.com
# Date: 2025-01-24
# Version: 1.0.0
#
# Description:
#     Scans video files for corruption using FFmpeg error detection and moves
#     corrupted files to error directories. Supports hardware acceleration and
#     multiple video formats with batch processing capabilities.
#
# Features:
#     - Hardware accelerated corruption detection
#     - Multiple FFmpeg detection methods
#     - Automatic error file isolation
#     - Support for multiple video formats
#     - Subdirectory processing
#     - Configurable target directories
#
# Requirements:
#     - bash 4.0+
#     - ffmpeg with hardware acceleration support
#     - Write permissions to target directories
#
# Usage:
#     ./video-corruption-scanner.sh
#

# Configuration - Edit these paths as needed
TOPDIR="/Volumes/SLAB_RAID/*SLAB_SYNC/Movies/ASSETS/LANDSCAPE"

# Option 1: Hardware acceleration with compliant error detection
echo "Running corruption scan with hardware acceleration..."
shopt -s nullglob nocaseglob

for subdir in "$TOPDIR"/*/; do
    for mov in "$subdir"/*.{mov,MOV,mp4,MP4,mpg,MPG,mpeg,MPEG,mts,MTS,m4v,M4V}; do
        if [ -f "$mov" ] && ! ffmpeg -hwaccel auto -err_detect compliant -i "$mov" -f null /dev/null 2>/dev/null; then
            mkdir -p "$subdir"/00error &&
            mv "$mov" "$subdir"/00error
            echo "Moved corrupted file: $mov"
        fi
    done
done

echo "Corruption scan complete."