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
# Script Name: images-rename-from-original-date.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible DateTimeOriginal to filename converter that uses
#     DTO as the filename and synchronizes FileModifyDate to match. Features EXIF
#     cleansing, batch processing, and integrity preservation for media libraries.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - GUI-based folder selection (supports multiple folders) with native macOS dialogs
#     - Parallel processing (auto-optimized for macOS systems)
#     - EXIF data cleansing with automatic file renaming
#     - Duplicate filename handling and metadata integrity preservation
#     - Comprehensive error handling and desktop output reporting
#     - Batch processing support with detailed progress tracking
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - exiftool (auto-installed if missing)
#     - tkinter (standard library with macOS compatibility)
#
# Usage:
#     python images-rename-from-original-date.py
#
####################################################################################
DTO to FMD Filename Converter
---------------------------
Author: Jason Paul Michaels
Date: December 28, 2024
Version: 1.0.0

Description:
    Converts files to use DateTimeOriginal (DTO) as the filename and updates
    FileModifyDate (FMD) to match. Includes EXIF data cleansing and integrity
    checks. Supports batch processing with GUI folder selection.

Features:
    - GUI-based folder selection (up to 25 folders)
    - Parallel processing (18 cores)
    - EXIF data cleansing
    - Automatic file renaming
    - Duplicate filename handling
    - Metadata integrity preservation
    - Error handling and reporting

Requirements:
    - Python 3.6+
    - exiftool
    - tkinter (standard library)
    - multiprocessing (standard library)
    - subprocess (standard library)
    - os (standard library)

Usage:
    python dto-to-fmd-filename.py
    Then select folders through GUI dialog

