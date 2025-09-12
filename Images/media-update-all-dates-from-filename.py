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
# Script Name: images-update-all-dates-from-filename.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible batch date processor that extracts date information
#     from filenames and updates all date-related metadata fields accordingly.
#     Features parallel processing for improved performance across large collections.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - GUI directory selection with native macOS dialogs
#     - Parallel processing (auto-optimized for macOS systems)
#     - Updates all date/time metadata fields from filename patterns
#     - Recursive directory scanning with progress tracking
#     - DS_Store file filtering and comprehensive error handling
#     - Desktop output support with detailed reporting
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - exiftool (auto-installed if missing)
#     - multiprocessing (standard library with macOS compatibility)
#
# Usage:
#     python images-update-all-dates-from-filename.py
#
####################################################################################
All Dates From Filename
----------------------
Author: Jason Paul Michaels
Date: December 28, 2024
Version: 1.0.0

Description:
    Batch processes image files to extract date information from filenames and
    updates all date-related metadata fields accordingly. Uses parallel processing
    for improved performance across large collections.

Features:
    - Parallel processing using 16 cores
    - Updates all date/time metadata fields
    - Recursive directory scanning
    - Progress tracking
    - DS_Store file filtering
    - Error handling and reporting

Requirements:
    - Python 3.6+
    - exiftool
    - multiprocessing (standard library)
    - subprocess (standard library)
    - os (standard library)

Usage:
    python alldates-from-filename.py

Note:
    Source folder is hardcoded to "/Volumes/SSD_RAID/Pictures"
    Modify source_folder variable to change target directory

