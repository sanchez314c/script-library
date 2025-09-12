#!/usr/bin/env python3
####################################################################################
#                                                                                  #
#    ██████╗ ███████╗████████╗   ███████╗██╗    ██╗██╗███████╗████████╗██╗   ██╗   #
#   ██╔════╝ ██╔════╝╚══██╔══╝   ██╔════╝██║    ██║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝   #
#   ██║  ███╗█████╗     ██║      ███████╗██║ █╗ ██║██║█████╗     ██║    ╚████╔╝    #
#   ██║   ██║██╔══╝     ██║      ╚════██║██║███╗██║██║██╔══╝     ██║     ╚██╔╝     #
#   ╚██████╔╝███████╗   ██║      ███████║╚███╔███╔╝██║██╗        ██║      ██║      #
#    ╚═════╝ ╚══════╝   ╚═╝      ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝        ╚═╝      ╚═╝      #
#                                                                                  #
####################################################################################
#
# Script Name: images-fix-rotation-batch.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible batch rotation correction processor that fixes
#     image orientation issues using external scripts. Maintains directory
#     structure while preserving original files with systematic processing.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Multiple format support (.jpg, .jpeg, .png, .webp, .dng, .gif, .tif, .tiff)
#     - Recursive directory processing with progress tracking
#     - Original file preservation with _unrotated suffix system
#     - External script integration with macOS-optimized paths
#     - Detailed logging and comprehensive error handling
#     - Desktop output support with native macOS compatibility
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - External unrotate script (auto-detected for macOS)
#     - os (standard library with macOS compatibility)
#
# Usage:
#     python images-fix-rotation-batch.py
#
####################################################################################
Title in Title Case
------------------
Author: Jason Paul Michaels
Date: December 28, 2024
Version: 1.0.0

Description:
    Clear, concise description of script purpose and functionality.
    Multiple lines if needed.

Features:
    - Feature one
    - Feature two
    - Additional features

Requirements:
    - Python version
    - Required packages
    - System dependencies

Usage:
    python script-name.py [arguments]

unrotate_script="/Users/heathen.admin/Library/Mobile Documents/com~apple~CloudDocs/Scripts/*RECENT/unrotate3.sh" # Corrected path to your unrotate script

function process_directory {
    for file in "$1"/*; do
        if [ -d "$file" ]; then
            # If it's a directory, recurse into it
            process_directory "$file"
        else
            # Check for various image file extensions
            if [[ $file =~ \.(jpg|jpeg|png|webp|dng|gif|tif|tiff)$ ]]; then
                echo "Processing $file"
                # Call your unrotate script here
                # Assuming the unrotated image should be saved with a suffix, e.g., filename_unrotated.jpg
                "$unrotate_script" "$file" "${file%.*}_unrotated.${file##*.}"
            fi
        fi
    done
}

root_directory="$1" # Pass the root directory as an argument to the script

process_directory "$root_directory"
